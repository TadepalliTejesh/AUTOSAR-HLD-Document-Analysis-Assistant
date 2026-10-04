import re
from typing import Dict, Any, List, Optional, Tuple
from app.core.logging import logger
from app.services.intent import QueryIntent, STOP_WORDS
from app.services.retriever import check_entity_match
from app.services.text_cleaner import clean_ocr_text


# Positive cues for structural / component intent
STRUCTURAL_KEYWORDS = {
    "consist", "consists", "consisting", "comprise", "comprises", "comprising",
    "component", "components", "layer", "layers", "sublayer", "sublayers",
    "part", "parts", "structure", "structures", "structural", "architecture",
    "block", "blocks", "diagram", "diagrams", "element", "elements", "stack",
    "framework", "module", "modules", "composition", "divided into", "contains",
    "approach", "hardware", "software", "middleware", "virtualization",
    "interface", "interfaces", "application software", "standardized middleware"
}

# Explicit benefit / advantage regex pattern
BENEFIT_PHRASE_PAT = re.compile(
    r'\b(?:reduc(?:e|ed|ing|tion)\s+(?:\w+\s+)?(?:time|cost|costs|effort|complexity|overhead)|'
    r'(?:enhanc|improv|increas)(?:e|ed|ing|ement)\s+(?:\w+\s+)?(?:quality|efficiency|performance|productivity|reliability)|'
    r'softwares?\s+reuse|reusab(?:le|ility)|parallel\s+development|distributed\s+development|'
    r'widely\s+independent|lower\s+(?:cost|costs|time)|faster\s+time|benefits?|advantages?|gain(?:s)?)\b',
    re.IGNORECASE
)

QUESTION_META_WORDS = {
    "stage", "stages", "step", "steps", "phase", "phases",
    "part", "parts", "element", "elements", "component", "components",
    "layer", "layers", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "two", "one",
    "main", "major", "primary", "key", "shown", "listed", "described", "contain", "contains",
    "demonstrated", "demonstrate", "demonstrates", "demonstration",
    "illustrated", "illustrate", "illustrates", "illustration",
    "using", "use", "uses", "used", "case", "cases",
    "provide", "provides", "provided", "providing", "provision",
    "across", "between", "among", "along", "defined", "define", "defines"
}


def score_intent_alignment(
    chunk_content: str,
    chunk_section: str,
    intent: str,
    entities: List[str],
    acronym_map: Optional[Dict[str, str]] = None
) -> float:
    """
    Evaluates whether a chunk's content and section align with the user's specific query intent.
    Returns a score between 0.0 (irrelevant / conflicting role) and 1.0 (strongly aligned).
    """
    cleaned = clean_ocr_text(chunk_content)
    sec_lower = chunk_section.lower().strip()
    text_lower = cleaned.lower()
    combined = sec_lower + " " + text_lower
    chunk_lines = [l.strip() for l in cleaned.split("\n") if l.strip()]

    # Check whether the section or the chunk's headings explicitly focus on benefits
    has_benefit_heading = any(w in sec_lower for w in ["benefit", "benefits", "advantage", "advantages", "why use"])
    if not has_benefit_heading and chunk_lines:
        for l in chunk_lines[:4]:
            if re.match(r"^(?:[\#\-\*•\d\.]*\s*)?(?:benefits?|advantages?)\s+(?:of\b)?", l, re.IGNORECASE):
                has_benefit_heading = True
                break

    # 1. Structure / Components Intent
    if intent == QueryIntent.STRUCTURE_COMPONENTS:
        benefit_matches = BENEFIT_PHRASE_PAT.findall(text_lower)
        benefit_term_count = len(benefit_matches)
        struct_term_count = sum(1 for kw in STRUCTURAL_KEYWORDS if kw in text_lower)

        # Severe penalty if the chunk has a benefit heading or is predominantly discussing benefits
        if has_benefit_heading:
            return 0.15
        if benefit_term_count >= 2 and struct_term_count <= 2:
            return 0.15

        # Positive structural signals
        score = 0.50

        # Section or chunk first-line title match for approach / structure / architecture
        first_line = cleaned.split("\n")[0].lower() if cleaned else ""
        combined_hdr = sec_lower + " " + first_line
        if any(w in combined_hdr for w in ["approach", "architecture", "structure", "layered", "overview", "components", "diagram", "principles"]):
            score += 0.20

        # Explicit structural layout: lines with layers/components, bullet items, arrows, or tables
        has_bullets = bool(re.search(r"^[\-–—•\*\d]\s+[A-Z]", cleaned, re.MULTILINE))
        has_arrows = any(sym in cleaned for sym in ["->", "→", "↓", "←", "|"])
        has_layer_blocks = bool(re.search(r"(?i)\b(?:software|hardware|middleware|firmware|application|platform|driver|service|runtime)\s+(?:layer|tier|stack|level|component|module|architecture|approach)\b", text_lower))
        short_diagram_lines = sum(1 for l in cleaned.split("\n") if 3 <= len(l.strip()) <= 45 and l.strip()[:1].isupper())

        if has_layer_blocks or short_diagram_lines >= 4:
            score += 0.25
        elif has_bullets or has_arrows:
            score += 0.15

        if struct_term_count >= 3:
            score += 0.15

        return min(max(score, 0.10), 1.0)

    # 2. Benefits / Advantages Intent
    elif intent == QueryIntent.BENEFITS:
        benefit_matches = BENEFIT_PHRASE_PAT.findall(text_lower)
        benefit_count = len(benefit_matches)

        if has_benefit_heading:
            return 1.0
        if benefit_count >= 2:
            return 0.95
        if benefit_count == 1:
            return 0.80

        # Chunks containing no benefit phrases or headings receive low alignment for benefits queries
        return 0.20

    # 3. Definition / Purpose Intent
    elif intent in (QueryIntent.DEFINITION_DESCRIPTION, QueryIntent.DEFINITION_PURPOSE):
        is_tool_gui = any(w in sec_lower for w in ["gui", "ui", "window", "screenshot", "menu", "dialog", "editor example", "errors & warnings"])
        if is_tool_gui:
            return 0.20

        # CRITICAL: If the chunk is a benefits section, heavily penalize it for definition queries!
        if has_benefit_heading:
            return 0.15

        benefit_matches = BENEFIT_PHRASE_PAT.findall(text_lower)
        if len(benefit_matches) >= 2 and not any(w in text_lower for w in ["defined as", "consists of", "is a", "architecture", "provides", "structure"]):
            return 0.20

        has_defining_label = any(
            bool(re.search(r'(?<![A-Za-z0-9_])' + re.escape(e.lower()) + r'(?![A-Za-z0-9_])[ \t]*[:\-\u2013\u2014]?[ \t]*([A-Z][a-zA-Z]*(?:[ \t]+[A-Z][a-zA-Z]*)*)', cleaned))
            for e in entities if len(e) >= 2 and e.lower() not in STOP_WORDS
        )
        if has_defining_label:
            return 1.0

        has_purpose = any(w in combined for w in ["purpose", "provides", "role", "aim", "responsible", "layer", "interface", "middleware", "architecture", "connects", "approach", "framework", "standardized"])
        has_entity_in_sec = any(check_entity_match(e, sec_lower, acronym_map) > 0.6 for e in entities)

        if has_entity_in_sec and has_purpose:
            return 1.0
        elif has_purpose:
            return 0.85
        return 0.45

    # 4. Comparison Intent
    elif intent == QueryIntent.COMPARISON:
        if len(entities) >= 2:
            matches = sum(1 for e in entities[:2] if check_entity_match(e, combined, acronym_map) > 0.4)
            if matches >= 2:
                return 1.0
            elif matches == 1:
                return 0.70
        return 0.50

    # 5. Mechanism / How It Works Intent
    elif intent == QueryIntent.MECHANISM_HOW:
        has_mech = any(w in text_lower for w in ["how", "mechanism", "workflow", "operation", "interacts", "processes", "executes", "triggers", "sequence", "handles"])
        if has_mech:
            return 0.90
        return 0.45

    # 6. Transformation / Workflow Intent
    elif intent == QueryIntent.TRANSFORMATION_WORKFLOW:
        has_gen = any(w in combined for w in ["generator", "generators", "generate", "generates"])
        has_io = any(w in combined for w in ["input", "output", "definition", "description", "xml", "epd", ".c", ".h", "source files", "header files"])
        if has_gen and has_io:
            return 1.0
        elif has_gen:
            return 0.70
        return 0.20

    # 7. Interoperability Intent
    elif intent == QueryIntent.INTEROPERABILITY:
        has_interop = any(w in combined for w in ["interoperab", "standard", "generic", "gce", "compatible", "compliant", "format", "exchange", "interface", "use case"])
        target_entity = entities[0] if entities else ""
        has_target = check_entity_match(target_entity, combined, acronym_map) >= 0.45 if target_entity else True
        if has_interop and has_target:
            return 1.0
        elif has_interop:
            return 0.85
        elif not has_target:
            return 0.20
        return 0.50

    # 7b. Relationship Intent
    elif intent == QueryIntent.RELATIONSHIP:
        score = 0.50
        if len(entities) >= 2:
            matches = sum(1 for e in entities[:2] if check_entity_match(e, combined, acronym_map) >= 0.45)
            if matches >= 2:
                score += 0.35
            elif matches == 1:
                score += 0.15
        if any(w in combined for w in ["distribution", "distribute", "distributed", "across", "bus", "network", "connect", "relationship", "interaction"]):
            score += 0.15
        if any(w in sec_lower for w in ["distribution", "relationship", "connection", "interaction"]):
            score += 0.15
        return min(max(score, 0.10), 1.0)

    # 8. List / Multi-Part Intent
    elif intent == QueryIntent.LIST_MULTI_PART:
        has_cat = any(w in combined for w in ["categories", "classes", "types", "groups", "drivers", "layers", "scalability", "stages"])
        has_bullets = bool(re.search(r"^[\-–—•\*\d]\s+[A-Z]", cleaned, re.MULTILINE))
        if has_cat or has_bullets:
            return 0.95
        return 0.50

    # 9. Process / Workflow Intent
    elif intent == QueryIntent.PROCESS_WORKFLOW:
        has_steps = any(w in combined for w in ["process", "workflow", "steps", "procedure", "sequence", "phase", "stage", "methodology"])
        if has_steps:
            return 0.90
        return 0.45

    # 10. General / Factual Intent
    else:
        # Base neutral alignment
        return 0.50


def get_word_root(w: str) -> str:
    """Gets generic morphological root / stem of an English word."""
    w_clean = w.lower().strip()
    return re.sub(r'(?:ing|ed|es|s|ence|ance|ent|ant|tion|sion|ic|al|ly|ability|ibility)$', '', w_clean)


def is_word_in_text(w: str, text: str) -> bool:
    """Checks whether a word, its morphological root, or compound constituent is present in text."""
    w_low = w.lower().strip()
    if not w_low:
        return False
    if w_low in text:
        return True
    if "-" in w_low or "_" in w_low:
        w_space = re.sub(r'[\-_]+', ' ', w_low)
        if w_space in text:
            return True
        sub_tokens = [t for t in re.split(r'[\-_]+', w_low) if t and t not in STOP_WORDS]
        if sub_tokens and all(is_word_in_text(sub, text) for sub in sub_tokens):
            return True
        RELATIONAL_SUFFIXES = {"defined", "based", "driven", "specific", "related", "oriented", "centric", "level", "aware"}
        primary_subs = [t for t in sub_tokens if t not in RELATIONAL_SUFFIXES]
        if primary_subs and all(is_word_in_text(sub, text) for sub in primary_subs):
            return True
    root = get_word_root(w_low)
    if len(root) >= 4 and root in text:
        return True
    return False


def evaluate_evidence_confidence(
    supporting_chunks: List[Dict[str, Any]],
    query: str,
    classification: Dict[str, Any],
    acronym_map: Optional[Dict[str, str]] = None
) -> Tuple[str, Dict[str, float]]:
    """
    Evaluates grounding confidence across 5 distinct factors:
    1. Semantic similarity (dense cosine similarity)
    2. Lexical overlap (query non-stopword tokens in evidence)
    3. Entity coverage (explicit query entities matched in evidence)
    4. Intent alignment (role match: structure vs benefits vs purpose)
    5. Evidence completeness & noise ratio

    Returns:
        (confidence_str: "High" | "Medium" | "Low" | "Insufficient", metrics_dict)
    """
    if not supporting_chunks:
        return "Insufficient", {
            "semantic_sim": 0.0,
            "lexical_overlap": 0.0,
            "entity_coverage": 0.0,
            "intent_alignment": 0.0,
            "completeness": 0.0,
            "noise_ratio": 1.0,
            "composite_score": 0.0
        }

    q_lower = query.lower()
    q_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", q_lower)) - STOP_WORDS
    entities = [e.lower().strip() for e in classification.get("entities", []) if e.strip()]

    combined_text = " ".join([c.get("section", "").lower() + " " + clean_ocr_text(c["content"]).lower() for c in supporting_chunks])
    top_chunk = supporting_chunks[0]

    # 1. Semantic Similarity
    semantic_sim = max(c.get("similarity", 0.5) for c in supporting_chunks)

    # 2. Lexical Overlap with morphological root matching
    matched_words = [w for w in q_words if is_word_in_text(w, combined_text)]
    lexical_overlap = len(matched_words) / max(len(q_words), 1)

    # 3. Entity Coverage
    if entities:
        ent_scores = [check_entity_match(e, combined_text, acronym_map) for e in entities]
        entity_coverage = sum(ent_scores) / len(ent_scores)
    else:
        entity_coverage = lexical_overlap

    # 4. Intent Alignment
    intent = classification.get("intent", QueryIntent.GENERAL)
    intent_alignment = score_intent_alignment(
        chunk_content=top_chunk["content"],
        chunk_section=top_chunk.get("section", ""),
        intent=intent,
        entities=entities,
        acronym_map=acronym_map
    )

    # 5. Completeness & Noise Ratio
    total_lines = sum(len(c["content"].split("\n")) for c in supporting_chunks)
    noisy_lines = sum(1 for c in supporting_chunks for l in c["content"].split("\n") if len(l.strip()) < 3 or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", l))
    noise_ratio = noisy_lines / max(total_lines, 1)

    completeness = 0.8
    if intent == QueryIntent.STRUCTURE_COMPONENTS:
        has_multiple_items = len(re.findall(r"^[\-–—•\*\d]\s+[A-Z]", top_chunk["content"], re.MULTILINE)) >= 2 or "\n" in top_chunk["content"]
        completeness = 1.0 if has_multiple_items else 0.5
    elif intent == QueryIntent.COMPARISON:
        completeness = 1.0 if (entities and all(check_entity_match(e, combined_text, acronym_map) > 0.4 for e in entities[:2])) else 0.4

    # Critical gate: If query specifically asks for an entity (e.g. "CGPA") and it is completely missing
    if entities and any(len(e) >= 3 and e not in STOP_WORDS and check_entity_match(e, combined_text, acronym_map) < 0.25 for e in entities):
        # Specific missing entity check
        return "Insufficient", {
            "semantic_sim": semantic_sim,
            "lexical_overlap": lexical_overlap,
            "entity_coverage": 0.0,
            "intent_alignment": intent_alignment,
            "completeness": 0.0,
            "noise_ratio": noise_ratio,
            "composite_score": 0.1
        }

    # Substantive content term absence gate:
    # If key content words in the query (length >= 5) are absent in the supporting text
    missing_substantive = [w for w in q_words if not is_word_in_text(w, combined_text) and len(w) >= 5 and w not in QUESTION_META_WORDS]
    if missing_substantive and lexical_overlap < 0.65:
        substantive_q_words = [w for w in q_words if len(w) >= 5 and w not in QUESTION_META_WORDS]
        if len(missing_substantive) >= 2 or (substantive_q_words and len(missing_substantive) / len(substantive_q_words) >= 0.40):
            return "Insufficient", {
                "semantic_sim": semantic_sim,
                "lexical_overlap": lexical_overlap,
                "entity_coverage": 0.0,
                "intent_alignment": intent_alignment,
                "completeness": 0.0,
                "noise_ratio": noise_ratio,
                "composite_score": 0.15
            }

    # Intent alignment gate: if retrieved chunks do not answer the requested intent
    if intent_alignment < 0.25:
        return "Insufficient", {
            "semantic_sim": semantic_sim,
            "lexical_overlap": lexical_overlap,
            "entity_coverage": entity_coverage,
            "intent_alignment": intent_alignment,
            "completeness": completeness,
            "noise_ratio": noise_ratio,
            "composite_score": 0.2
        }

    composite_score = (
        0.25 * semantic_sim +
        0.20 * lexical_overlap +
        0.20 * entity_coverage +
        0.25 * intent_alignment +
        0.10 * (1.0 - min(noise_ratio, 1.0))
    )

    if composite_score >= 0.70 and intent_alignment >= 0.70 and entity_coverage >= 0.65:
        confidence = "High"
    elif composite_score >= 0.48 and intent_alignment >= 0.40:
        confidence = "Medium"
    elif composite_score < 0.35 or entity_coverage < 0.30:
        confidence = "Insufficient"
    else:
        confidence = "Low"

    metrics = {
        "semantic_sim": round(semantic_sim, 3),
        "lexical_overlap": round(lexical_overlap, 3),
        "entity_coverage": round(entity_coverage, 3),
        "intent_alignment": round(intent_alignment, 3),
        "completeness": round(completeness, 3),
        "noise_ratio": round(noise_ratio, 3),
        "composite_score": round(composite_score, 3)
    }

    return confidence, metrics


def prune_and_rank_supporting_evidence(
    chunks: List[Dict[str, Any]],
    query: str,
    classification: Dict[str, Any],
    acronym_map: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """
    Ranks candidate chunks using multi-factor evidence-to-intent alignment and prunes
    down to 1-3 direct supporting chunks. Eliminates noisy, off-intent, or conflicting chunks.
    """
    if not chunks:
        return [], "Insufficient", {}

    entities = [e.lower().strip() for e in classification.get("entities", []) if e.strip()]
    intent = classification.get("intent", QueryIntent.GENERAL)
    q_lower = query.lower()
    q_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", q_lower)) - STOP_WORDS
    is_ui_query = any(w in q_lower for w in ["window", "gui", "screenshot", "menu", "editor", "dialog", "view", "display"])

    scored = []
    for c in chunks:
        cleaned_content = clean_ocr_text(c["content"])
        sec_lower = c.get("section", "").lower()
        combined_text = sec_lower + " " + cleaned_content.lower()

        # 1. Section Title Affinity
        sec_match_score = 0.0
        for ent in entities:
            if ent in sec_lower or check_entity_match(ent, sec_lower, acronym_map) > 0.6:
                sec_match_score = max(sec_match_score, 0.9)

        # Multi-word phrase or substantive keyword match between query and section title
        q_phrases = [p.strip() for p in re.findall(r"\b[a-zA-Z0-9_\-]{3,}(?:\s+[a-zA-Z0-9_\-]+)+\b", q_lower)]
        for p in q_phrases:
            if p in sec_lower:
                sec_match_score = max(sec_match_score, 0.95)

        # 2. UI screenshot penalty for conceptual queries
        is_tool_gui = any(w in sec_lower for w in ["gui", "ui", "window", "screenshot", "menu", "dialog", "editor example", "errors & warnings"])
        gui_penalty = 0.35 if (is_tool_gui and not is_ui_query) else 1.0

        # View mismatch penalty (e.g. asking for logical view / architecture vs development approach / process)
        is_view_mismatch = (
            (any(w in q_lower for w in ["logical view", "architecture", "layered architecture", "layers"]) and any(w in sec_lower for w in ["development approach", "workflow", "process"])) or
            (any(w in q_lower for w in ["development approach", "workflow", "process", "lifecycle"]) and any(w in sec_lower for w in ["logical view", "layered architecture"]))
        )
        view_factor = 0.35 if is_view_mismatch else 1.0

        # 3. Content word overlap
        matched_words = [w for w in q_words if w in combined_text]
        word_overlap = len(matched_words) / max(len(q_words), 1)

        # 4. Entity match
        if entities:
            ent_scores = [check_entity_match(e, combined_text, acronym_map) for e in entities]
            max_ent = max(ent_scores)
        else:
            max_ent = 1.0

        # 5. Dynamic Intent Alignment Score
        intent_align = score_intent_alignment(
            chunk_content=c["content"],
            chunk_section=c.get("section", ""),
            intent=intent,
            entities=entities,
            acronym_map=acronym_map
        )

        # 6. Exact Multi-Word Phrase Boost
        longest_matching_phrase = 0
        for term in classification.get("search_terms", []):
            term_words = term.split()
            if len(term_words) > 1 and term in combined_text:
                longest_matching_phrase = max(longest_matching_phrase, len(term_words))

        phrase_boost = 0.30 if longest_matching_phrase >= 3 else (0.15 if longest_matching_phrase == 2 else 0.0)

        target_ent = classification.get("target", "").lower().strip()
        label_targets = [target_ent] if target_ent else entities
        has_defining_label = any(
            bool(re.search(r'(?<![A-Za-z0-9_])' + re.escape(e.lower()) + r'(?![A-Za-z0-9_])[ \t]*[:\-\u2013\u2014]?[ \t]*([A-Z][a-zA-Z]*(?:[ \t]+[A-Z][a-zA-Z]*)*)', cleaned_content))
            for e in label_targets if len(e) >= 2 and e.lower() not in STOP_WORDS
        )
        label_boost = 0.25 if has_defining_label else 0.0

        support_score = (
            0.30 * intent_align +
            0.20 * word_overlap +
            0.15 * max_ent +
            0.10 * sec_match_score +
            0.10 * c.get("similarity", 0.5) +
            0.15 * phrase_boost +
            label_boost
        ) * gui_penalty * view_factor

        c_copy = dict(c)
        c_copy["content"] = cleaned_content
        c_copy["support_score"] = support_score
        c_copy["intent_alignment"] = intent_align
        scored.append(c_copy)

    scored.sort(key=lambda x: x["support_score"], reverse=True)
    top_chunk = scored[0]
    top_score = top_chunk["support_score"]

    # Check if top candidate is insufficient
    if top_score < 0.35 or top_chunk["intent_alignment"] < 0.25:
        return [], "Insufficient", {"top_score": top_score, "top_intent_alignment": top_chunk["intent_alignment"]}

    # --- Evidence Minimality & Sufficiency Gate ---
    # Check if top_chunk completely and sufficiently answers the query on its own
    is_top_sufficient = False
    if (
        intent not in {QueryIntent.COMPARISON, QueryIntent.MULTI_HOP_FLOW}
        and top_chunk["intent_alignment"] >= 0.70
        and top_chunk.get("word_overlap", 0.0) >= 0.70
        and len(top_chunk["content"].strip()) >= 50
    ):
        if entities and len(entities) >= 2:
            top_comb = top_chunk.get("section", "").lower() + " " + top_chunk["content"].lower()
            all_entities_covered = all(check_entity_match(e, top_comb, acronym_map) >= 0.45 for e in entities)
            is_top_sufficient = all_entities_covered
        else:
            is_top_sufficient = True

    if is_top_sufficient:
        # Minimum sufficient evidence set: cite ONLY the complete top chunk
        supporting = [top_chunk]
    else:
        # Multi-page or multi-aspect evidence required (e.g. comparison or multi-entity query)
        supporting = [top_chunk]
        top_comb = top_chunk.get("section", "").lower() + " " + top_chunk["content"].lower()
        for c in scored[1:]:
            if len(supporting) >= 3:
                break
            c_comb = c.get("section", "").lower() + " " + c["content"].lower()

            # Check if c provides novel entity coverage not covered by top_chunk
            has_novel_entity = False
            if entities:
                for e in entities:
                    if check_entity_match(e, top_comb, acronym_map) < 0.40 and check_entity_match(e, c_comb, acronym_map) >= 0.45:
                        has_novel_entity = True
                        break

            # Multi-part list or transformation workflow expansion
            is_multi_part_expansion = (
                intent in {QueryIntent.LIST_MULTI_PART, QueryIntent.TRANSFORMATION_WORKFLOW, QueryIntent.COMPARISON, QueryIntent.MULTI_HOP_FLOW}
                and c["support_score"] >= top_score * 0.75
                and c["intent_alignment"] >= 0.45
            )

            # Same page and same section expansion (e.g. multi-part diagram split across chunks on the same slide)
            is_same_page_section = (
                c.get("page") == top_chunk.get("page")
                and c.get("section") == top_chunk.get("section")
                and c["support_score"] >= top_score * 0.60
            )

            # Reject neighboring chunks with conflicting subtopic headings
            chunk_lines = [l.strip() for l in c["content"].split("\n") if l.strip()]
            has_conflicting_heading = False
            if chunk_lines:
                first_lines_text = " ".join(chunk_lines[:3]).lower()
                for q_term in q_words:
                    if len(q_term) >= 4 and q_term in top_comb and q_term not in c_comb:
                        # Top chunk contains key inquiry subject not in c; check if c has a distinct heading
                        top_heading_terms = set(re.findall(r"\b[a-zA-Z]{4,}\b", top_chunk.get("section", "").lower() + " " + chunk_lines[0].lower()))
                        c_heading_terms = set(re.findall(r"\b[a-zA-Z]{4,}\b", first_lines_text))
                        distinct_heading = c_heading_terms - top_heading_terms - q_words
                        if distinct_heading and not any(w in q_words for w in c_heading_terms):
                            has_conflicting_heading = True
                            break

            if (has_novel_entity or is_multi_part_expansion or is_same_page_section) and not has_conflicting_heading:
                supporting.append(c)

    confidence, metrics = evaluate_evidence_confidence(
        supporting_chunks=supporting,
        query=query,
        classification=classification,
        acronym_map=acronym_map
    )

    if confidence == "Insufficient":
        return [], "Insufficient", metrics

    return supporting, confidence, metrics


def validate_citations_against_answer(
    answer: str,
    supporting_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Validates supporting chunks against the final generated answer.
    Guarantees that every cited chunk directly supports at least one substantive claim
    in the final answer. Removes topically related but unused neighboring chunks.
    """
    if not supporting_chunks or len(supporting_chunks) <= 1:
        return supporting_chunks

    # Exclude formulaic template lines and meta-text; split by sentence boundaries and newlines
    clean_lines = [l.strip() for l in re.split(r"(?<=[.!?])\s+|\n+", answer) if l.strip()]
    claims = []
    for l in clean_lines:
        l_lower = l.lower()
        if l_lower.startswith(("based on", "according to", "answer:", "evidence:", "confidence:", "citations:", "potential inconsistency")):
            continue
        c = re.sub(r"^[•\-\*\d\.\)\s\|]+", "", l).strip()
        if len(c) > 20:
            claims.append(c)

    if not claims:
        return supporting_chunks[:1]

    # For each claim, find the best supporting chunk
    best_supporting_chunk_indices = set()
    for claim in claims:
        claim_tokens = {w for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", claim.lower()) if w not in STOP_WORDS and len(w) > 2}
        if not claim_tokens:
            continue
        best_idx = -1
        best_score = 0.0
        for i, chunk in enumerate(supporting_chunks):
            c_text = clean_ocr_text(chunk["content"]).lower()
            matched = {w for w in claim_tokens if w in c_text}
            score = len(matched) / len(claim_tokens)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0 and best_score >= 0.35:
            best_supporting_chunk_indices.add(best_idx)

    if not best_supporting_chunk_indices:
        return supporting_chunks[:1]

    return [supporting_chunks[i] for i in sorted(best_supporting_chunk_indices)]

