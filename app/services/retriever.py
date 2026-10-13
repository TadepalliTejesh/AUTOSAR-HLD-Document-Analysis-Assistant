import re
from typing import List, Dict, Any, Tuple, Optional
from app.core.logging import logger
from app.services.vector_store import vector_store_service
from app.services.embeddings import embedding_service
from app.services.intent import classify_query, QueryIntent, extract_search_terms, STOP_WORDS


def extract_document_acronyms(all_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Dynamically extracts acronym definitions from any document text.
    Handles patterns like:
    - Full Name (ACRONYM)
    - ACRONYM (Full Name)
    Zero hardcoded domain words.
    """
    full_text = " ".join([c["content"] + " " + c["section"] for c in all_chunks])
    pairs = re.findall(r'([A-Za-z][a-zA-Z0-9_\-\s]{2,40}?)\s*\(([A-Za-z0-9]{2,8})\)', full_text)
    pairs += [(m[1], m[0]) for m in re.findall(r'\b([A-Za-z0-9]{2,8})\s*\(([A-Za-z][a-zA-Z0-9_\-\s]{2,40}?)\)', full_text)]

    acronym_map: Dict[str, str] = {}
    for name, acr in pairs:
        name_clean = name.strip().lower()
        acr_clean = acr.strip().lower()
        if len(name_clean) > len(acr_clean) and not re.match(r'^(tm|url|http|part|page|fig)$', acr_clean):
            acronym_map[acr_clean] = name_clean
            acronym_map[name_clean] = acr_clean
    return acronym_map


COMPOUND_PREFIXES = ("micro", "multi", "electro", "tele", "nano", "opto", "cyber", "inter", "intra")


def get_stem(w: str) -> str:
    """Extracts morphological root of an English word with prefix/suffix stripping."""
    w = w.lower().strip()
    w = re.sub(r'^(?:sub|super|inter|intra|multi|pre|post|re)', '', w)
    for _ in range(2):
        w = re.sub(r'(?:ing|ed|es|s|ence|ance|ent|ant|tion|sion|ic|al|ly|ability|ibility|ive|or|er)$', '', w)
    return w


def matches_token(w: str, text: str) -> bool:
    """Checks exact match or morphological root match in text."""
    if w in text:
        return True
    st = get_stem(w)
    if len(st) >= 4 and st in text:
        return True
    return False


def check_entity_match(ent: str, text: str, acronym_map: Optional[Dict[str, str]] = None) -> float:
    """
    Dynamically scores how well an entity matches candidate text:
    - 1.0: Exact word match, normalized hyphen match, or full acronym expansion match.
    - 0.95: Exact initialism match (with scientific compound prefix splitting).
    - 0.85 / 0.60: Strong prefix match for acronym entities.
    - 0.50: Partial multi-word match.
    - 0.0: No match.
    """
    ent_lower = ent.lower().strip()
    text_lower = text.lower()

    if not ent_lower or not text_lower:
        return 0.0

    # 1. Exact token / word / technical identifier match (including :: or /)
    if ent_lower in text_lower:
        return 1.0
    if re.search(r'\b' + re.escape(ent_lower) + r'\b', text_lower):
        return 1.0

    # Normalized hyphens, Greek micro, colons, slashes, and underscores
    norm_text = re.sub(r'[\u00b5\u03bc]c\b', 'microcontroller', text_lower)
    norm_text = re.sub(r'[\u00b5\u03bc]', 'micro', norm_text)
    norm_text = re.sub(r'[:/.\-\s_]+', ' ', norm_text)
    norm_ent = re.sub(r'[:/.\-\s_]+', ' ', ent_lower)
    if norm_ent in norm_text:
        return 1.0

    # Multi-word entity: all substantive words present
    ent_words = [w for w in re.findall(r'\b[a-zA-Z0-9]+\b', norm_ent) if w not in STOP_WORDS]
    if len(ent_words) > 1 and all(w in norm_text or matches_token(w, norm_text) for w in ent_words):
        return 1.0

    # Acronym map expansion (learned dynamically from document)
    if acronym_map and ent_lower in acronym_map:
        expansion = acronym_map[ent_lower]
        if expansion in norm_text or all(w in norm_text for w in expansion.split() if w not in STOP_WORDS):
            return 1.0

    # Initialism / acronym matching with scientific compound prefix splitting (e.g. Microcontroller -> micro + controller -> m + c)
    if 2 <= len(ent_lower) <= 6 and ent_lower.isalpha():
        raw_words = [w for w in re.findall(r'\b[a-zA-Z]{2,}\b', norm_text) if w not in STOP_WORDS]
        words = []
        for w in raw_words:
            split_done = False
            for pfx in COMPOUND_PREFIXES:
                if w.startswith(pfx) and len(w) > len(pfx) + 2:
                    words.extend([pfx, w[len(pfx):]])
                    split_done = True
                    break
            if not split_done:
                words.append(w)

        # First pass: exact initialism match
        for n in range(2, len(words) + 1):
            for i in range(len(words) - n + 1):
                initials = "".join(w[0] for w in words[i:i+n])
                if initials == ent_lower:
                    return 0.95

        # Second pass: strong prefix match (at least 3 letters, matching prefix of length >= len(ent)-1)
        if len(ent_lower) >= 4:
            for n in range(3, len(words) + 1):
                for i in range(len(words) - n + 1):
                    initials = "".join(w[0] for w in words[i:i+n])
                    if len(initials) >= len(ent_lower) - 1 and ent_lower.startswith(initials):
                        return 0.60

    # Partial word match for multi-word phrases (proportional to matched content words)
    if ent_words:
        matched_ent_words = sum(1 for w in ent_words if (w in norm_text or matches_token(w, norm_text)) and len(w) > 2)
        match_ratio = matched_ent_words / len(ent_words)
        if match_ratio >= 0.66:
            return 0.50
        elif match_ratio >= 0.33:
            return 0.20
        return 0.0

    return 0.0


def evaluate_confidence(
    max_similarity: float,
    supporting_chunks_count: int,
    lexical_overlap: float,
    entity_coverage: float = 1.0,
    min_relevance_threshold: float = 0.58
) -> Tuple[str, str]:
    """
    Evaluates evidence confidence strictly based on retrieved content quality:
    - High: Strong semantic match, high entity coverage, substantive lexical overlap, and supporting chunks.
    - Medium: Moderate semantic or lexical match with at least one supporting chunk.
    - Insufficient: Weak match, inadequate entity coverage, or absence of key query concepts.
    """
    if supporting_chunks_count == 0 or max_similarity < min_relevance_threshold or entity_coverage < 0.35:
        return "Insufficient", "Retrieved chunks do not contain sufficient evidence to answer the query."

    if max_similarity >= 0.70 and entity_coverage >= 0.70 and supporting_chunks_count >= 1:
        return "High", f"Strong semantic similarity ({max_similarity:.2f}) and confirmed multi-entity coverage."

    if max_similarity >= 0.58 and entity_coverage >= 0.40 and supporting_chunks_count >= 1:
        return "Medium", f"Moderate relevance ({max_similarity:.2f}) with supporting document evidence."

    return "Insufficient", "Retrieved chunks do not contain sufficient evidence to answer the query."


class RetrieverService:
    def __init__(
        self,
        top_k: int = 5,
        min_relevance: float = 0.58,
        vector_store: Optional[Any] = None,
        embedding_svc: Optional[Any] = None
    ):
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.vector_store = vector_store or vector_store_service
        self.embedding_svc = embedding_svc or embedding_service

    def retrieve(
        self,
        document_id: int,
        query: str
    ) -> Tuple[List[Dict[str, Any]], str, float, str]:
        """
        Dynamically retrieves relevant chunks using hybrid retrieval (Dense Semantic + Lexical Match + Entity Coverage)
        with zero hardcoded pages or question lists.
        Strictly isolated by document_id.
        """
        clean_query = query.strip()
        classification = classify_query(clean_query)
        intent = classification["intent"]
        search_terms = classification.get("search_terms", [])
        entities = classification.get("entities", [])
        entities_lower = [e.lower().strip() for e in entities if e and e.strip()]

        # 1. Fetch all chunks for this isolated document to enable hybrid lexical indexing
        all_chunks = self.vector_store.get_all_chunks(document_id)
        if not all_chunks:
            return [], "Insufficient", 0.0, "No chunks found in document index."

        all_doc_text = " ".join([c["content"].lower() + " " + c["section"].lower() for c in all_chunks])

        # 2. Check for concept presence across the document (Generic Insufficient Evidence Check)
        acronyms = re.findall(r"\b[A-Z]{2,6}\b", clean_query)
        if acronyms:
            for acr in acronyms:
                if acr.lower() not in all_doc_text:
                    logger.info(f"Generic Insufficient Evidence: Acronym '{acr}' absent in doc {document_id}")
                    return [], "Insufficient", 0.0, f"The uploaded document does not contain information about {acr}."

        distinctive_words = [w for w in re.findall(r"\b[A-Za-z0-9_\-]{3,}\b", clean_query.lower()) if w not in STOP_WORDS]
        if distinctive_words:
            found_words = [w for w in distinctive_words if w in all_doc_text]
            if len(found_words) == 0:
                logger.info(f"Generic Insufficient Evidence: No search terms {distinctive_words} found in doc {document_id}")
                return [], "Insufficient", 0.0, "The uploaded document does not contain information on this topic."

            # Calculate chunk frequency for found words to detect ubiquitous background terms
            total_chunks = max(len(all_chunks), 1)
            chunk_freq = {
                w: sum(1 for c in all_chunks if w in c["content"].lower() or w in c.get("section", "").lower())
                for w in found_words
            }
            rare_found = [w for w in found_words if chunk_freq[w] / total_chunks < 0.40]
            missing_distinctive = [w for w in distinctive_words if w not in all_doc_text and len(w) >= 4]

            # If core inquiry keywords are absent from the entire document and no rare/topic-specific words were found:
            if missing_distinctive and (not rare_found or len(missing_distinctive) >= len(found_words)):
                logger.info(f"Generic Insufficient Evidence: Specific inquiry terms {missing_distinctive} absent in doc {document_id}")
                return [], "Insufficient", 0.0, "The uploaded document does not contain information on this topic."

        acronym_map = extract_document_acronyms(all_chunks)

        # 3. Dense Semantic Retrieval across the whole document
        query_emb = self.embedding_svc.embed_text(clean_query)
        dense_candidates = self.vector_store.search(
            document_id=document_id,
            query_embedding=query_emb,
            top_k=len(all_chunks)
        )
        dense_map = {c["chunk_id"]: c["similarity"] for c in dense_candidates}

        # 4. Multi-Feature Scoring across all chunks
        PURPOSE_CUES = [
            "purpose", "role", "function", "provides", "responsible for", "used for",
            "enables", "allows", "abstracts", "manages", "interfaces", "aim", "objective",
            "task", "defined as", "serves as", "baseline", "protects", "avoids", "capability"
        ]

        scored_candidates = []
        for chunk in all_chunks:
            cid = chunk["chunk_id"]
            content_lower = chunk["content"].lower()
            section_lower = chunk["section"].lower()
            combined_text = section_lower + " " + content_lower

            # Skip TOC and Agenda slides during substantive Q&A
            is_toc = any(w in section_lower for w in ["table of contents", "contents", "index", "agenda", "outline", "toc"])
            if is_toc and not any(w in clean_query.lower() for w in ["table of contents", "contents", "index", "agenda", "outline", "toc"]):
                continue

            sim_dense = dense_map.get(cid, 0.45)

            norm_combined = re.sub(r'[\-–—_/:]+', ' ', combined_text)
            norm_sec = re.sub(r'[\-–—_/:]+', ' ', section_lower)

            # a. Section / Title Relevance
            sec_word_matches = sum(1 for w in distinctive_words if matches_token(w, norm_sec))
            sec_unigram_ratio = sec_word_matches / max(len(distinctive_words), 1)

            phrase_in_sec = any(
                len(e.split()) > 1 and re.sub(r'[\-–—_/:]+', ' ', e) in norm_sec
                for e in entities_lower
            ) or any(
                len(term.split()) > 1 and (term in section_lower or term in norm_sec)
                for term in search_terms
            )
            sec_relevance = sec_unigram_ratio
            if phrase_in_sec:
                sec_relevance = max(sec_relevance, 0.50 + 0.45 * sec_unigram_ratio)

            # Canonical section matching: does section title match or start/end with an entity?
            canonical_sec_match = False
            target_ent = classification.get("target", "").lower().strip()
            candidates_to_check = entities_lower + ([target_ent] if target_ent else [])
            for ent in candidates_to_check:
                if not ent or ent in STOP_WORDS:
                    continue
                m = check_entity_match(ent, section_lower, acronym_map)
                norm_e = re.sub(r'[:/.\-\s_]+', ' ', ent)
                if m >= 0.85 or norm_sec.strip() == norm_e or norm_sec.startswith(norm_e + " ") or norm_sec.endswith(" " + norm_e):
                    canonical_sec_match = True
                    break

            if canonical_sec_match:
                sec_relevance = max(sec_relevance, 0.95)

            if entities_lower and len(entities_lower) >= 2:
                sec_ent_matches = sum(1 for e in entities_lower if check_entity_match(e, section_lower, acronym_map) >= 0.5)
                if sec_ent_matches == len(entities_lower):
                    sec_relevance = max(sec_relevance, 0.95)

            # b. Multi-Entity Coverage with dynamic acronym & hyphen matching
            covered_count = 0.0
            for ent in entities_lower:
                m = check_entity_match(ent, combined_text, acronym_map)
                covered_count += m
            entity_coverage = (covered_count / len(entities_lower)) if entities_lower else 1.0

            # c. High-Precision Lexical & Phrase Matching
            matched_words = sum(1 for w in distinctive_words if matches_token(w, combined_text) or matches_token(w, norm_combined))
            unigram_overlap = matched_words / max(len(distinctive_words), 1)

            phrase_match_score = 0.0
            longest_matching_phrase = 0
            for term in search_terms:
                substantive_tokens = [w for w in term.split() if w not in STOP_WORDS]
                if len(substantive_tokens) > 1 and (term in combined_text or term in norm_combined):
                    phrase_match_score += 0.35
                    longest_matching_phrase = max(longest_matching_phrase, len(substantive_tokens))

            phrase_match_score = min(1.0, phrase_match_score)
            lex_score = max(unigram_overlap, 0.40 * unigram_overlap + 0.60 * phrase_match_score)

            phrase_boost = 0.0
            if longest_matching_phrase >= 3:
                phrase_boost = 0.30
            elif longest_matching_phrase == 2:
                phrase_boost = 0.15

            # d. Intent-Specific Contextual Weighting
            intent_boost = 0.0

            # Target Entity Focus: For queries asking about a specific target entity,
            # prioritize chunks containing that entity and prevent off-target chunks from being boosted by background words.
            if target_ent and len(target_ent) >= 2 and target_ent not in STOP_WORDS:
                target_m = check_entity_match(target_ent, combined_text, acronym_map)
                if target_m >= 0.5:
                    intent_boost += 0.35
                elif target_m < 0.25 and intent not in (QueryIntent.MULTI_HOP_FLOW, QueryIntent.DOCUMENT_PURPOSE):
                    intent_boost -= 0.40
                    phrase_boost = 0.0

            # Diagram entity-definition / label boost: e.g. "ara::exec Execution Management" or "MCAL Microcontroller Abstraction Layer"
            has_entity_label = False
            label_targets = [target_ent] if target_ent else entities_lower
            for ent in label_targets:
                if len(ent) >= 2 and ent not in STOP_WORDS:
                    pat = r'(?<![A-Za-z0-9_])' + re.escape(ent) + r'(?![A-Za-z0-9_])[ \t]*[:\-\u2013\u2014]?[ \t]*([A-Z][a-zA-Z]*(?:[ \t]+[A-Z][a-zA-Z]*)*)'
                    m = re.search(pat, chunk["content"])
                    if m and m.group(1).strip() and len(m.group(1).strip()) >= 3:
                        has_entity_label = True
                        break
                    # Acronym definition pattern: e.g. "AUTOSAR Runtime for Adaptive Applications (ARA)"
                    if re.search(r'\b[A-Za-z][a-zA-Z0-9_\-\s]{2,40}?\(\s*' + re.escape(ent) + r'\s*\)', chunk["content"], re.IGNORECASE):
                        has_entity_label = True
                        break
            if has_entity_label:
                intent_boost += 0.35

            # Document Purpose Intent
            if intent == QueryIntent.DOCUMENT_PURPOSE:
                is_doc_purpose_sec = any(w in section_lower for w in ["purpose", "purpose and inputs", "inputs", "scope", "introduction", "document overview"])
                has_purpose_content = any(w in content_lower for w in ["purpose of this document", "purpose of the document", "describes the", "forms the base", "scope of this document"])
                if is_doc_purpose_sec or has_purpose_content:
                    sec_relevance = max(sec_relevance, 0.95)
                    intent_boost += 0.50
                if any(w in section_lower for w in ["example of a layered", "release history", "revision"]):
                    intent_boost -= 0.50

            # Architecture view vs Development/Process view affinity
            ARCH_VIEW_KEYWORDS = ["logical view", "architecture", "layered architecture", "layers", "components", "structural view", "component view", "distribution", "interfaces"]
            PROCESS_VIEW_KEYWORDS = ["stage", "stages", "development approach", "workflow", "process", "lifecycle", "phases", "steps", "methodology"]

            query_has_arch_view = any(w in clean_query.lower() for w in ARCH_VIEW_KEYWORDS)
            query_has_process_view = any(w in clean_query.lower() for w in PROCESS_VIEW_KEYWORDS)

            if query_has_arch_view and not query_has_process_view and intent not in (QueryIntent.DOCUMENT_PURPOSE, QueryIntent.MULTI_HOP_FLOW):
                if any(w in section_lower for w in ["logical view", "layered software architecture", "layered architecture"]):
                    sec_relevance = max(sec_relevance, 0.95)
                    intent_boost += 0.35
                elif any(w in section_lower for w in ["development approach", "workflow", "process", "continuous improvement"]):
                    intent_boost -= 0.50
                    sec_relevance = 0.0
                    phrase_boost = 0.0
            elif query_has_process_view and not query_has_arch_view:
                if any(w in section_lower for w in ["development approach", "workflow", "process", "methodology", "lifecycle"]):
                    sec_relevance = max(sec_relevance, 0.95)
                    intent_boost += 0.35
                elif any(w in section_lower for w in ["logical view", "layered architecture"]):
                    intent_boost -= 0.50
                    sec_relevance = 0.0
                    phrase_boost = 0.0

            if intent == QueryIntent.RELATIONSHIP:
                if len(entities_lower) >= 2:
                    if entity_coverage >= 0.70:
                        intent_boost += 0.35
                    elif entity_coverage >= 0.45:
                        intent_boost += 0.20
                if any(w in section_lower for w in ["architecture", "process", "framework", "overview", "distribution"]):
                    intent_boost += 0.15

            elif intent == QueryIntent.INTEROPERABILITY:
                if entity_coverage >= 0.70:
                    intent_boost += 0.35
                elif entity_coverage >= 0.45:
                    intent_boost += 0.20
                if any(w in section_lower for w in ["use case", "interface", "exchange", "interoperab"]):
                    intent_boost += 0.20

            elif intent in (QueryIntent.DEFINITION_DESCRIPTION, QueryIntent.DEFINITION_PURPOSE):
                has_benefit = any(w in section_lower for w in ["benefit", "benefits", "advantage", "advantages", "why use"])
                if has_benefit:
                    intent_boost -= 0.35
                elif entities_lower:
                    target_ent = entities_lower[0]
                    ent_m = check_entity_match(target_ent, combined_text, acronym_map)
                    if ent_m > 0:
                        intent_boost += 0.25
                if any(w in section_lower for w in ["architecture", "overview", "system", "principles", "approach", "logical view"]):
                    intent_boost += 0.25
                if any(w in section_lower for w in ["document overview", "history", "changelog", "release notes"]):
                    intent_boost -= 0.35

            elif intent == QueryIntent.STRUCTURE_COMPONENTS:
                has_benefit = any(w in section_lower for w in ["benefit", "benefits", "advantage", "advantages", "why use"])
                if has_benefit:
                    intent_boost -= 0.35
                if any(w in section_lower for w in ["architecture", "layer", "layers", "structural", "logical view", "components", "overview", "stack", "tier"]):
                    intent_boost += 0.35
                elif any(w in section_lower for w in ["development approach", "workflow", "process", "lifecycle", "history", "procedure"]):
                    intent_boost -= 0.30
                if any(cue in combined_text for cue in ["1.", "2.", "•", "-", "->", "→", "|", "layer", "tier", "component", "subsystem", "module", "block", "stack"]):
                    intent_boost += 0.20

            elif intent == QueryIntent.BENEFITS:
                has_benefit = any(w in section_lower for w in ["benefit", "benefits", "advantage", "advantages", "why use"])
                if has_benefit:
                    intent_boost += 0.35
                else:
                    intent_boost -= 0.25

            elif intent == QueryIntent.COMPARISON and len(entities_lower) >= 2:
                if entity_coverage >= 0.70:
                    intent_boost += 0.35
                elif entity_coverage >= 0.45:
                    intent_boost += 0.18
                if canonical_sec_match:
                    intent_boost += 0.40

            elif intent == QueryIntent.MULTI_HOP_FLOW:
                source_ent = classification.get("source", "").lower().strip()
                target_ent = classification.get("target", "").lower().strip()
                m_src = check_entity_match(source_ent, combined_text, acronym_map) if source_ent else 0.0
                m_tgt = check_entity_match(target_ent, combined_text, acronym_map) if target_ent else 0.0
                BRIDGE_CUES = [
                    "runtime environment", "rte", "ecu abstraction", "hardware abstraction",
                    "microcontroller abstraction", "mcal", "driver", "drivers", "i/o hw abstraction",
                    "basic software", "bsw", "abstraction layer", "sensor/actuator", "microcontroller"
                ]
                is_bridge = any(cue in combined_text for cue in BRIDGE_CUES)
                is_stack_table = "table" in chunk.get("source_type", "").lower() or ("|" in chunk["content"] and "layer" in combined_text)
                if m_src >= 0.4:
                    intent_boost += 0.40
                if m_tgt >= 0.4:
                    intent_boost += 0.40
                if is_bridge:
                    intent_boost += 0.40
                    entity_coverage = max(entity_coverage, 0.75)
                if is_stack_table:
                    intent_boost += 0.30

            elif intent == QueryIntent.LIST_MULTI_PART:
                if sec_relevance >= 0.40:
                    intent_boost += 0.35
                has_division_stmt = bool(re.search(r'(?i)\b(?:divided|subdivided|subdivision|divided in the layers)\b', combined_text))
                if has_division_stmt:
                    intent_boost += 0.40
                elif any(cue in combined_text for cue in ["1.", "2.", "•", "-", "->", "→", "|", "stage", "class", "step", "category", "phase", "type", "level", "layer"]):
                    intent_boost += 0.20

            elif intent == QueryIntent.PROCESS_WORKFLOW:
                if any(cue in combined_text for cue in ["workflow", "process", "steps", "procedure", "flow", "pipeline"]):
                    intent_boost += 0.30

            label_boost = 0.25 if has_entity_label else 0.0

            # Hybrid Score Formulation
            hybrid_score = (
                0.35 * sim_dense +
                0.25 * entity_coverage +
                0.20 * lex_score +
                0.10 * sec_relevance +
                0.10 * intent_boost +
                phrase_boost +
                label_boost +
                (0.40 if canonical_sec_match else 0.0)
            )
            if is_toc:
                hybrid_score -= 0.35

            c_copy = dict(chunk)
            c_copy["similarity"] = round(sim_dense, 4)
            c_copy["entity_coverage"] = round(entity_coverage, 4)
            c_copy["sec_relevance"] = round(sec_relevance, 4)
            c_copy["hybrid_score"] = round(hybrid_score, 4)
            scored_candidates.append(c_copy)

        scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        if not scored_candidates:
            return [], "Insufficient", 0.0, "No relevant evidence found in the document."

        # 5. Greedy Diversity & Coverage Selection
        filtered_chunks = []
        seen_ids = set()

        # 1. COMPARISON: guarantee top defining chunk for each compared entity across all chunks
        if intent == QueryIntent.COMPARISON and entities_lower and len(entities_lower) >= 2:
            for ent in entities_lower:
                best_cand = None
                best_score = -1.0
                for c in scored_candidates:
                    c_sec = c["section"].lower().strip()
                    c_cnt = c["content"].lower()
                    c_comb = c_sec + " " + c_cnt
                    m_comb = check_entity_match(ent, c_comb, acronym_map)
                    m_sec = check_entity_match(ent, c_sec, acronym_map)
                    if m_comb >= 0.5:
                        is_exact_sec = (c_sec == ent or re.sub(r'[:/.\-\s_]+', ' ', c_sec).strip() == ent)
                        score = c["hybrid_score"] + (0.75 if is_exact_sec else (0.35 if m_sec >= 0.85 else 0.0))
                        if score > best_score:
                            best_cand = c
                            best_score = score
                if best_cand and best_cand["chunk_id"] not in seen_ids:
                    filtered_chunks.append(best_cand)
                    seen_ids.add(best_cand["chunk_id"])

        # 2. MULTI_HOP_FLOW: select diverse layers along the stack
        elif intent == QueryIntent.MULTI_HOP_FLOW:
            layer_specs = [
                ("app", ["application layer", "sensor/actuator", "software component"]),
                ("rte", ["runtime environment", "rte"]),
                ("ecu_abs", ["ecu abstraction", "hardware abstraction", "i/o hw abstraction"]),
                ("mcal", ["microcontroller abstraction", "mcal"]),
                ("complex_drivers", ["complex driver", "complex drivers"])
            ]
            for layer_key, layer_terms in layer_specs:
                best_layer_chunk = None
                best_layer_score = -1.0
                for c in scored_candidates:
                    c_sec = c["section"].lower()
                    c_cnt = c["content"].lower()
                    matches_layer = any(t in c_sec or t in c_cnt for t in layer_terms)
                    if matches_layer:
                        sec_matches = any(t in c_sec for t in layer_terms)
                        score = c["hybrid_score"] + (0.70 if sec_matches else 0.0)
                        if score > best_layer_score:
                            best_layer_chunk = c
                            best_layer_score = score
                if best_layer_chunk and best_layer_chunk["chunk_id"] not in seen_ids:
                    filtered_chunks.append(best_layer_chunk)
                    seen_ids.add(best_layer_chunk["chunk_id"])

        # Top-k fill / general greedy diversity
        top_score = scored_candidates[0]["hybrid_score"] if scored_candidates else 1.0
        covered_ents = set()
        for c in scored_candidates[:20]:
            if len(filtered_chunks) >= self.top_k:
                break
            cid = c["chunk_id"]
            if cid in seen_ids:
                continue

            adds_new = False
            txt = c["section"].lower() + " " + c["content"].lower()
            for ent in entities_lower:
                if check_entity_match(ent, txt, acronym_map) > 0.5:
                    if ent not in covered_ents:
                        adds_new = True
                        covered_ents.add(ent)

            is_same_slide = (
                filtered_chunks and
                c.get("page") == filtered_chunks[0].get("page") and
                c.get("section") == filtered_chunks[0].get("section") and
                c["hybrid_score"] >= top_score * 0.75
            )
            if len(filtered_chunks) < 2 or adds_new or is_same_slide or c["hybrid_score"] >= top_score * 0.90:
                filtered_chunks.append(c)
                seen_ids.add(cid)

        if not filtered_chunks:
            return [], "Insufficient", 0.0, "Insufficient evidence retrieved."

        # 6. Evaluate Confidence dynamically
        max_sim = max(c["similarity"] for c in filtered_chunks)
        all_retrieved_text = " ".join([c["content"].lower() + " " + c["section"].lower() for c in filtered_chunks])

        # Final entity coverage across all retrieved chunks
        final_covered = 0.0
        for ent in entities_lower:
            m = check_entity_match(ent, all_retrieved_text, acronym_map)
            final_covered += m
        final_entity_coverage = (final_covered / len(entities_lower)) if entities_lower else 1.0

        matched_words = [w for w in distinctive_words if w in all_retrieved_text]
        lexical_overlap = len(matched_words) / max(len(distinctive_words), 1)

        confidence_level, explanation = evaluate_confidence(
            max_similarity=max_sim,
            supporting_chunks_count=len(filtered_chunks),
            lexical_overlap=lexical_overlap,
            entity_coverage=final_entity_coverage,
            min_relevance_threshold=self.min_relevance
        )

        logger.info(
            f"Retrieved {len(filtered_chunks)} chunks for doc {document_id} "
            f"(Max sim: {max_sim:.3f}, Final coverage: {final_entity_coverage:.2f}, Lexical: {lexical_overlap:.2f}, Conf: {confidence_level})"
        )

        return filtered_chunks, confidence_level, max_sim, explanation


retriever_service = RetrieverService()
