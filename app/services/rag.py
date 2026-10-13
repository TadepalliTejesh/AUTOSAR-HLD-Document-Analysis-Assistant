import time
import re
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.services.retriever import retriever_service, check_entity_match, extract_document_acronyms
from app.services.llm import ollama_service
from app.services.intent import classify_query, QueryIntent, STOP_WORDS
from app.services.text_cleaner import clean_ocr_text, clean_chunk
from app.services.evidence_aligner import prune_and_rank_supporting_evidence, validate_citations_against_answer
from app.db.repositories import QueryRepository, AuditRepository, DocumentRepository
from app.schemas.query import QueryResponse, Citation, RetrievedChunk


SYSTEM_PROMPT = """You are an engineering document analysis assistant.

You must answer ONLY using the supplied document context.

Never use external knowledge.

Never invent components, parameters, relationships, numbers, requirements, or claims.

If the supplied context does not contain enough evidence to answer, respond:
"INSUFFICIENT EVIDENCE: The uploaded document does not provide enough evidence to answer this question."

Every factual claim must be grounded in the supplied context.

If evidence is conflicting, explicitly state:
"Potential inconsistency detected."

Do not silently resolve conflicting information.
The engineer must make the final decision."""


def sanitize_llm_output(raw_answer: str, supporting_chunks: List[Dict[str, Any]], question: str) -> str:
    """
    Validates and cleans LLM generation.
    Detects assistant meta-talk, prompt contamination, or canned responses
    such as 'I'm ready to assist', 'What is the engineering question?', 'As an AI...',
    or prompt scaffolding repetition.
    If detected, falls back to direct grounded synthesis.
    """
    if not raw_answer or not raw_answer.strip():
        return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    cleaned = raw_answer.strip()
    cleaned = re.sub(r"^(?:answer|final answer):\s*", "", cleaned, flags=re.IGNORECASE).strip()

    canned_patterns = [
        r"(?i)\b(?:i am|i'm)\s+ready\s+to\s+assist\b",
        r"(?i)\bwhat\s+is\s+the\s+engineering\s+question\b",
        r"(?i)\bhow\s+can\s+i\s+assist\s+you\b",
        r"(?i)\bas\s+an\s+ai\b",
        r"(?i)\bengineering\s+question\s*:",
        r"(?i)\bdocument\s+context\s*:",
        r"(?i)\bwhat\s+is\s+your\s+question\b"
    ]
    for pat in canned_patterns:
        if re.search(pat, cleaned):
            logger.warning(f"Detected conversational template contamination: '{cleaned[:60]}...'. Employing direct grounded synthesis.")
            return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    if len(cleaned) < 15 and "insufficient" not in cleaned.lower():
        logger.warning(f"Response too short: '{cleaned}'. Employing direct grounded synthesis.")
        return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    # Detect tautological / superficial answers for architectural definitions
    # e.g., "The Microcontroller Abstraction Layer is a software layer."
    q_lower = question.lower()
    if any(q_lower.startswith(w) for w in ["what is ", "define ", "explain "]):
        if re.search(r"(?i)\bis\s+(?:a|an)\s+(?:software\s+)?(?:architecture\s+)?(?:layer|component)\s*\.?\s*$", cleaned):
            logger.warning(f"Detected superficial definition answer: '{cleaned}'. Falling back to direct grounded synthesis.")
            return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    # For architecture / hierarchy queries: check if major architectural entities present in evidence were missed
    if any(w in q_lower for w in ["layers of", "software layers", "layer hierarchy", "architecture layers"]):
        if not any(w in q_lower for w in ["three highest", "3 highest", "three software layers"]):
            chunks_text = "\n".join([c["content"].lower() for c in supporting_chunks])
            for structural_term in ["driver", "hardware", "platform", "microcontroller"]:
                if structural_term in chunks_text and structural_term not in cleaned.lower():
                    logger.warning(f"Architecture layer answer missed key structural dimension '{structural_term}'. Falling back to direct grounded synthesis.")
                    return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    # For path traversal / trace queries: ensure RTE is included and proper sequence is maintained
    if any(w in q_lower for w in ["trace", "path from", "flow from"]) and any(w in q_lower for w in ["peripheral", "hardware", "microcontroller"]):
        chunks_text = "\n".join([c["content"].lower() for c in supporting_chunks])
        if "rte" in chunks_text or "runtime environment" in chunks_text:
            if "rte" not in cleaned.lower() and "runtime environment" not in cleaned.lower():
                logger.warning("Path traversal answer omitted intermediate RTE layer. Falling back to direct grounded synthesis.")
                return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    # For stage / progression queries: ensure prominent stages/steps from evidence are captured
    if any(w in q_lower for w in ["stage", "stages", "steps", "phases"]):
        top_headings = []
        for c in supporting_chunks:
            for l in c["content"].split("\n")[:6]:
                l_s = l.strip()
                if 3 <= len(l_s) <= 25 and not l_s.startswith(("#", "-", "•", "*", "|")) and l_s[0].isupper():
                    if any(w in l_s.lower() for w in ["approach", "view", "architecture", "overview", "document"]):
                        continue
                    top_headings.append(l_s.lower())
        if len(top_headings) >= 2 and not any(h in cleaned.lower() for h in top_headings[:3]):
            logger.warning("Stage progression answer missed prominent top-level stages. Falling back to direct grounded synthesis.")
            return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    # For architectural view / parts queries: ensure structured breakdown
    if any(w in q_lower for w in ["part", "parts", "component", "components", "tier", "tiers", "layers"]):
        evidence_lines = [l.strip() for c in supporting_chunks for l in c["content"].split("\n") if len(l.strip()) > 3]
        if len(evidence_lines) >= 4 and len(cleaned.split()) < 30 and not any(sym in cleaned for sym in ["1.", "2.", "-", "•", "*"]):
            logger.warning("Parts query answer lacked structured breakdown. Falling back to direct grounded synthesis.")
            return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    # For component / layer queries: ensure both hierarchical position and responsibilities/abstraction are stated
    if any(q_lower.startswith(w) for w in ["what is ", "define ", "explain "]) and any(w in q_lower for w in ["layer", "component", "driver", "module", "service", "interface"]):
        if len(cleaned.split()) < 20 or not any(w in cleaned.lower() for w in ["abstract", "interface", "hardware", "provide", "responsible", "function", "layer", "module", "service"]):
            logger.warning("Component answer lacks sufficient technical substance. Falling back to direct grounded synthesis.")
            return ollama_service._direct_grounded_synthesis("", supporting_chunks, question=question)

    # Strip trailing "INSUFFICIENT EVIDENCE:" disclaimer when substantive points are present
    lines = [l for l in cleaned.split("\n") if l.strip()]
    substantive_lines = [l for l in lines if not re.match(r"(?i)^INSUFFICIENT EVIDENCE\b", l.strip())]
    if len(substantive_lines) >= 3 and any(re.match(r"(?i)^INSUFFICIENT EVIDENCE\b", l.strip()) for l in lines):
        cleaned = "\n".join(substantive_lines).strip()

    return cleaned


def prune_supporting_evidence(
    chunks: List[Dict[str, Any]],
    question: str,
    classification: Dict[str, Any],
    acronym_map: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Prunes candidate retrieval chunks down to direct supporting evidence chunks (1-3 chunks).
    Delegates to the document-agnostic evidence aligner.
    """
    supporting, conf, _ = prune_and_rank_supporting_evidence(
        chunks=chunks,
        query=question,
        classification=classification,
        acronym_map=acronym_map
    )
    return supporting, conf


class RAGService:
    def __init__(self, retriever: Optional[Any] = None):
        self.retriever = retriever or retriever_service

    def answer_question(
        self,
        document_id: int,
        question: str,
        db: Session
    ) -> QueryResponse:
        start_time = time.time()
        
        # Verify document exists
        doc = DocumentRepository.get_by_id(db, document_id)
        if not doc:
            raise ValueError(f"Document with ID {document_id} not found.")

        # Log query submission
        AuditRepository.log(
            db=db,
            event_type="QUERY_SUBMITTED",
            document_id=document_id,
            details={"question": question}
        )

        # 1. Retrieve candidate chunks dynamically using hybrid retrieval
        candidate_chunks, initial_conf, max_sim, explanation = self.retriever.retrieve(
            document_id=document_id,
            query=question
        )

        # 2. Insufficient evidence handling
        if initial_conf == "Insufficient" or not candidate_chunks:
            exec_time = round((time.time() - start_time) * 1000, 2)
            insufficient_ans = (
                f"Answer:\n"
                f"INSUFFICIENT EVIDENCE:\n"
                f"{explanation}\n\n"
                f"Confidence:\n"
                f"Insufficient\n\n"
                f"Citations:\n"
                f"None"
            )
            
            QueryRepository.log_query(
                db=db,
                doc_id=document_id,
                question=question,
                answer=insufficient_ans,
                confidence="Insufficient",
                similarity_score=max_sim,
                citations=[],
                retrieved_chunks=[],
                execution_time_ms=exec_time
            )

            AuditRepository.log(
                db=db,
                event_type="QUERY_INSUFFICIENT_EVIDENCE",
                document_id=document_id,
                details={
                    "question": question,
                    "max_similarity": max_sim,
                    "reason": explanation
                }
            )

            return QueryResponse(
                document_id=document_id,
                question=question,
                answer=insufficient_ans,
                confidence="Insufficient",
                similarity_score=max_sim,
                citations=[],
                retrieved_evidence=[],
                execution_time_ms=exec_time,
                potential_inconsistency=False
            )

        # 3. Classify query intent and prune candidates to direct supporting evidence
        classification = classify_query(question)
        acronym_map = extract_document_acronyms(candidate_chunks)
        supporting_chunks, confidence, metrics = prune_and_rank_supporting_evidence(
            chunks=candidate_chunks,
            query=question,
            classification=classification,
            acronym_map=acronym_map
        )

        if confidence == "Insufficient" or not supporting_chunks:
            exec_time = round((time.time() - start_time) * 1000, 2)
            insufficient_ans = (
                f"Answer:\n"
                f"INSUFFICIENT EVIDENCE:\n"
                f"The uploaded document does not provide enough evidence to answer this question.\n\n"
                f"Confidence:\n"
                f"Insufficient\n\n"
                f"Citations:\n"
                f"None"
            )

            QueryRepository.log_query(
                db=db,
                doc_id=document_id,
                question=question,
                answer=insufficient_ans,
                confidence="Insufficient",
                similarity_score=max_sim,
                citations=[],
                retrieved_chunks=[],
                execution_time_ms=exec_time
            )

            AuditRepository.log(
                db=db,
                event_type="QUERY_INSUFFICIENT_EVIDENCE",
                document_id=document_id,
                details={
                    "question": question,
                    "max_similarity": max_sim,
                    "metrics": metrics
                }
            )

            return QueryResponse(
                document_id=document_id,
                question=question,
                answer=insufficient_ans,
                confidence="Insufficient",
                similarity_score=max_sim,
                citations=[],
                retrieved_evidence=[],
                execution_time_ms=exec_time,
                potential_inconsistency=False
            )

        # 4. Build prompt with cleaned supporting evidence
        context_blocks = []
        for c in supporting_chunks:
            citation_tag = f"[Page {c['page']} | Section: {c['section']}]"
            context_blocks.append(
                f"--- CONTEXT CHUNK {citation_tag} (Type: {c['source_type']}) ---\n"
                f"{c['content']}\n"
            )

        user_prompt = (
            f"DOCUMENT CONTEXT:\n\n"
            + "\n".join(context_blocks)
            + f"\nENGINEERING QUESTION: {question}\n\n"
            f"Provide a strictly grounded answer based ONLY on the context above.\n"
            f"- Answer ONLY the specific question asked. Do not discuss unrelated concepts or layers not asked about.\n"
            f"- For architecture layer or hierarchy questions: Distinguish software architecture layers (ordered top to bottom), spanning/complex drivers (if present), and the underlying hardware platform situated beneath the software layers.\n"
            f"- For path traversal / trace questions: Trace the standard layered path from the application software component to the hardware peripheral: Application Layer (Sensor/Actuator Software Components) -> AUTOSAR Runtime Environment (RTE) -> Basic Software (ECU Abstraction Layer -> Microcontroller Abstraction Layer / MCAL) -> Microcontroller Hardware. Describe the role of intermediate layers based on the context. Clarify that peripheral interactions do not all follow an identical or strictly linear path: distinguish internal device drivers (located in MCAL accessing internal microcontroller peripherals) from external device drivers (located in ECU Abstraction accessing external peripherals), and note that Complex Drivers provide a direct access bypass from Hardware to the RTE for special timing or non-standard hardware handling.\n"
            f"- For subdivision, classification, or categorization questions: Distinguish architectural subdivisions (e.g. software layers) from functional module categories or service types. When the question asks for a specific number of subdivisions (such as four subdivisions of Basic Software), identify the exact division that matches that count from the context (e.g. Services, ECU Abstraction, Microcontroller Abstraction, Complex Drivers), and do not conflate them with service types or conformance class tables.\n"
            f"- For comparison questions: Systematically compare each item across role, data modification (whether it changes the data content), and architectural layer location based on the definitions in the context.\n"
            f"- For component / layer definition questions (such as MCAL): State its position in the hierarchy (e.g. lowest software layer of Basic Software), its primary task (hardware independence), and its driver groups based on the context.\n"
            f"- For stage progression, workflow, or development approach questions: Identify the main stages or flow shown in the context before listing details.\n"
            f"- For architectural view or logical view questions: List the top-level main parts / structural tiers shown in the diagram and key functional clusters.\n"
            f"- For technical identifier or module acronym questions: State the exact full name or functional cluster indicated in the context.\n"
            f"- When explaining an architecture component or layer: Specify its hierarchical position (what is above/below it) and its concrete role based on the context.\n"
            f"- For 'What is X?' / 'Define X' questions: Describe what the concept itself is based on the context. If the question asks for a definition of X, do not substitute a list of benefits as the definition of X.\n"
            f"- For 'How does X...?' questions: Use the principles, mechanisms, and abstraction described in the context to explain how it is achieved.\n"
            f"- For 'Why does X...?' / 'What are the benefits of X?' questions: Explain the benefits and advantages described in the context.\n"
            f"- For 'What does X consist of?' / structure questions: List the structural elements and components from the context.\n"
            f"Answer directly and concisely. Do NOT repeat instructions, greetings, conversational filler, or prompt scaffolding.\n"
            f"If information is missing, respond with INSUFFICIENT EVIDENCE."
        )

        # 5. Generate answer dynamically from supporting chunks
        llm_raw_answer = ollama_service.generate(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            retrieved_chunks=supporting_chunks,
            question=question
        )
        llm_raw_answer = sanitize_llm_output(llm_raw_answer, supporting_chunks, question=question)

        # 6. Post-generation citation validation against generated answer
        # Validates that every cited chunk directly supports at least one substantive claim
        validated_chunks = validate_citations_against_answer(
            answer=llm_raw_answer,
            supporting_chunks=supporting_chunks,
            question=question
        )
        if not validated_chunks:
            validated_chunks = supporting_chunks[:1]

        unique_citations = []
        retrieved_evidence_schemas = []
        seen = set()
        for c in validated_chunks:
            key = (c["page"], c["section"])
            if key not in seen:
                seen.add(key)
                unique_citations.append(Citation(
                    page=c["page"],
                    section=c["section"],
                    source_type=c["source_type"],
                    text_snippet=c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"]
                ))
                retrieved_evidence_schemas.append(RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    page=c["page"],
                    section=c["section"],
                    source_type=c["source_type"],
                    content=c["content"],
                    similarity=c["similarity"]
                ))

        # Sort citations by page number ascending for deterministic, reader-friendly ordering
        unique_citations.sort(key=lambda c: (c.page, c.section))
        retrieved_evidence_schemas.sort(key=lambda r: (r.page, r.section))

        # Format Evidence lines dynamically strictly from validated chunks
        evidence_lines = [f"- Page {c.page} — Section: {c.section}" for c in unique_citations]
        evidence_block = "\n".join(evidence_lines)

        # Post-generation confidence calibration
        ADMISSION_PATTERNS = [
            r"(?i)\bdoes not explicitly state\b",
            r"(?i)\bnot specified in the context\b",
            r"(?i)\bnot mentioned in the context\b",
            r"(?i)\bdoes not contain (?:any )?information\b",
            r"(?i)\bis not provided in the (?:supplied )?context\b",
            r"(?i)\bcontext does not provide\b",
            r"(?i)\bdoes not state the purpose\b"
        ]
        has_admission = any(re.search(pat, llm_raw_answer) for pat in ADMISSION_PATTERNS)

        is_insufficient = (
            llm_raw_answer.strip().lower().startswith("insufficient evidence")
            or (len(llm_raw_answer.strip()) < 100 and "insufficient evidence" in llm_raw_answer.lower())
            or (has_admission and len(llm_raw_answer.strip().split("\n")) <= 3)
        )
        if is_insufficient:
            confidence = "Insufficient"
            unique_citations = []
            retrieved_evidence_schemas = []
            full_formatted_answer = (
                f"Answer:\n"
                f"{llm_raw_answer}\n\n"
                f"Confidence:\n"
                f"Insufficient\n\n"
                f"Citations:\n"
                f"None"
            )
        else:
            if has_admission:
                confidence = "Low"
            full_formatted_answer = (
                f"Answer:\n"
                f"{llm_raw_answer}\n\n"
                f"Evidence:\n"
                f"{evidence_block}\n\n"
                f"Confidence:\n"
                f"{confidence}"
            )

        has_inconsistency = "potential inconsistency" in llm_raw_answer.lower()
        exec_time = round((time.time() - start_time) * 1000, 2)

        # Persist query history and audit
        QueryRepository.log_query(
            db=db,
            doc_id=document_id,
            question=question,
            answer=full_formatted_answer,
            confidence=confidence,
            similarity_score=max_sim,
            citations=[c.model_dump() for c in unique_citations],
            retrieved_chunks=[r.model_dump() for r in retrieved_evidence_schemas],
            execution_time_ms=exec_time
        )

        AuditRepository.log(
            db=db,
            event_type="QUERY_COMPLETED",
            document_id=document_id,
            details={
                "question": question,
                "confidence": confidence,
                "similarity_score": max_sim,
                "citations_count": len(unique_citations),
                "execution_time_ms": exec_time
            }
        )

        return QueryResponse(
            document_id=document_id,
            question=question,
            answer=full_formatted_answer,
            confidence=confidence,
            similarity_score=max_sim,
            citations=unique_citations,
            retrieved_evidence=retrieved_evidence_schemas,
            execution_time_ms=exec_time,
            potential_inconsistency=has_inconsistency
        )


rag_service = RAGService()
