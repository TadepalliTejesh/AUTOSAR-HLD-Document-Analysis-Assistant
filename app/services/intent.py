import re
from typing import Dict, Any, List, Optional


class QueryIntent:
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    MULTI_HOP_FLOW = "multi_hop_flow"
    TRANSFORMATION_WORKFLOW = "transformation_workflow"
    INTEROPERABILITY = "interoperability"
    LIST_MULTI_PART = "list_multi_part"
    DEFINITION_DESCRIPTION = "definition_description"
    DEFINITION_PURPOSE = "definition_description"
    STRUCTURE_COMPONENTS = "structure_components"
    BENEFITS = "benefits"
    MECHANISM_HOW = "mechanism_how"
    PROCESS_WORKFLOW = "process_workflow"
    GENERAL = "general"


def split_query_entities(raw_str: str) -> List[str]:
    """Splits conjunctions and comma lists of entities dynamically."""
    raw = re.sub(r"^(?:the|a|an)\s+", "", raw_str.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(
        r"^(?:the\s+)?(?:three\s+|four\s+|five\s+|main\s+|major\s+|primary\s+|key\s+)?(?:stages|steps|phases|parts|elements|components|categories|layers|pillars|principles|levels)\s+(?:shown\s+in|described\s+in|of|in|for)\s+",
        "",
        raw,
        flags=re.IGNORECASE
    )
    parts = re.split(r",\s*and\s*|,\s*|\s+and\s+", raw, flags=re.IGNORECASE)
    clean = []
    for p in parts:
        c = re.sub(r"^(?:the|a|an)\s+", "", p.strip(), flags=re.IGNORECASE)
        c = re.sub(r"\s+releases?$", "", c, flags=re.IGNORECASE).strip()
        c = re.sub(r"\s+documents?$", "", c, flags=re.IGNORECASE).strip()
        if c and len(c) > 1 and c.lower() not in STOP_WORDS:
            clean.append(c)
    return clean


STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "of", "in", "on", "at", "to", "for",
    "a", "an", "and", "or", "by", "with", "from", "does", "do", "did", "how",
    "which", "who", "whom", "whose", "why", "where", "can", "could", "would",
    "should", "between", "difference", "differences", "compare", "contrast",
    "list", "tell", "explain", "describe", "mention", "provide", "show", "shown",
    "document", "say", "says", "about", "uploaded", "according", "this", "that", "it", "any", "be",
    "using", "use", "across", "role", "roles", "purpose", "purposes", "function", "functions",
    "responsibility", "responsibilities", "meaning", "definition"
}


def classify_query(query: str) -> Dict[str, Any]:
    """
    Dynamically analyzes user question structure without hardcoding any domain entities.
    Identifies query intent (relationship, comparison, multi-hop flow, transformation,
    interoperability, list, definition, general) and extracts core semantic entities
    and search phrases dynamically.
    """
    q = query.strip()
    q_lower = q.lower()

    # 1. Multi-entity Relationship / Connection Intent
    rel_match = re.search(
        r"(?:what is the relationship between|relationship between|explain the relationship between|relation between|interaction between)\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if rel_match:
        ents = split_query_entities(rel_match.group(1))
        return {
            "intent": QueryIntent.RELATIONSHIP,
            "entities": ents,
            "entity1": ents[0] if len(ents) > 0 else "",
            "entity2": ents[1] if len(ents) > 1 else "",
            "search_terms": extract_search_terms(q)
        }

    conn_match = re.search(
        r"(?:how does.*?connect|how do.*?connect|connection between)\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if conn_match:
        ents = split_query_entities(conn_match.group(1))
        return {
            "intent": QueryIntent.RELATIONSHIP,
            "entities": ents,
            "entity1": ents[0] if len(ents) > 0 else "",
            "entity2": ents[1] if len(ents) > 1 else "",
            "search_terms": extract_search_terms(q)
        }

    # Distribution across entities / ECUs / nodes (e.g. How is X distributed across Y?)
    dist_match = re.search(
        r"(?:how\s+(?:is|are)\s+(?:the\s+|an?\s+)?(.*?)\s+distributed\s+(?:across|on|between|among|in)\s+(?:the\s+|an?\s+)?(.*?)|"
        r"distribution\s+of\s+(?:the\s+|an?\s+)?(.*?)\s+(?:across|on|between|among|in)\s+(?:the\s+|an?\s+)?(.*?))"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if dist_match:
        ents_raw = [g for g in dist_match.groups() if g is not None]
        subj = ents_raw[0].strip()
        obj = ents_raw[1].strip()
        subj = re.sub(r"^(?:the|a|an)\s+", "", subj, flags=re.IGNORECASE).strip()
        obj = re.sub(r"^(?:the|a|an)\s+", "", obj, flags=re.IGNORECASE).strip()
        ents = [subj, obj]
        return {
            "intent": QueryIntent.RELATIONSHIP,
            "entities": ents,
            "entity1": subj,
            "entity2": obj,
            "target": subj,
            "search_terms": extract_search_terms(q)
        }

    # 2. Transformation / Workflow Intent (e.g. How does X transform Y into Z?)
    trans_match = re.search(
        r"(?:how does|how do)\s+(?:the\s+|an?\s+)?(.*?)\s+(?:transform|convert|generate|produce|translate)\s+(?:the\s+|an?\s+)?(.*?)(?:\s+into\s+|\s+to\s+)(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if trans_match:
        process_name = trans_match.group(1).strip()
        inputs = trans_match.group(2).strip()
        outputs = trans_match.group(3).strip()
        ents = [process_name, inputs, outputs]
        return {
            "intent": QueryIntent.TRANSFORMATION_WORKFLOW,
            "entities": ents,
            "process": process_name,
            "inputs": inputs,
            "outputs": outputs,
            "search_terms": extract_search_terms(q)
        }

    # 3. Interoperability / Enabling Mechanism Intent (e.g. How does X enable interoperability with Y?)
    mech_match = re.search(
        r"(?:how does|how do)\s+(?:the\s+|an?\s+)?(.*?)\s+(?:enable|allow|support|facilitate|provide|interact with|work with|interoperate with)\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if mech_match:
        subj = mech_match.group(1).strip()
        obj = mech_match.group(2).strip()
        ents = [subj, obj]
        acrs = re.findall(r"\b[A-Z]{2,6}\b", subj)
        for a in acrs:
            if a not in ents:
                ents.append(a)
        return {
            "intent": QueryIntent.INTEROPERABILITY,
            "entities": ents,
            "subject": subj,
            "capability": obj,
            "search_terms": extract_search_terms(q)
        }

    # Demonstration / Realization Intent (e.g. How is X demonstrated using Y?)
    demo_match = re.search(
        r"(?:how\s+(?:is|are)\s+(?:the\s+|an?\s+)?(.*?)\s+(?:demonstrated|illustrated|realized|shown|modeled|implemented)\s+using\s+(?:the\s+|an?\s+)?(.*?))"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if demo_match:
        subj = demo_match.group(1).strip()
        obj = demo_match.group(2).strip()
        subj = re.sub(r"^(?:the|a|an)\s+", "", subj, flags=re.IGNORECASE).strip()
        obj = re.sub(r"^(?:the|a|an)\s+", "", obj, flags=re.IGNORECASE).strip()
        ents = [subj, obj]
        return {
            "intent": QueryIntent.INTEROPERABILITY,
            "entities": ents,
            "subject": subj,
            "capability": obj,
            "target": subj,
            "search_terms": extract_search_terms(q)
        }

    # 4. Comparison / Difference Intent
    comp_match = re.search(
        r"(?:difference between|differences between|compare|contrast)\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if not comp_match:
        comp_match = re.search(
            r"(?:what does|how does|what do|how do)\s+(.*?)\s+(?:provide that makes it different from|differ from|provide compared to|compare with|compare to)\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
            q,
            re.IGNORECASE
        )
    if not comp_match:
        comp_match = re.search(
            r"(?:what|how)\s+(?:is|are)\s+(?:the\s+)?(.*?)\s+compared to\s+(?:the\s+)?(.*?)(?:\?|$|\.)",
            q,
            re.IGNORECASE
        )
    if comp_match:
        if comp_match.lastindex == 2:
            ents = [comp_match.group(1).strip(), comp_match.group(2).strip()]
        else:
            ents = split_query_entities(comp_match.group(1))
        return {
            "intent": QueryIntent.COMPARISON,
            "entities": ents,
            "entity1": ents[0] if len(ents) > 0 else "",
            "entity2": ents[1] if len(ents) > 1 else "",
            "search_terms": extract_search_terms(q)
        }

    # 5. Multi-hop Flow / Pathway Intent
    flow_match = re.search(
        r"(?:how does|how do)\s+(?:an?\s+|the\s+)?(.*?)\s+access\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if not flow_match:
        flow_match = re.search(
            r"(?:communication|data|execution|control|call)?\s*(?:path|flow|chain|stack|traversal)\s+(?:from|between)\s+(?:the\s+|an?\s+)?(.*?)\s+(?:to|and)\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
            q,
            re.IGNORECASE
        )

    if flow_match:
        source = flow_match.group(1).strip()
        target = flow_match.group(2).strip()
        source = re.sub(r"^(?:the|a|an)\s+", "", source, flags=re.IGNORECASE).strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+in\s+.*$", "", target, flags=re.IGNORECASE).strip()

        return {
            "intent": QueryIntent.MULTI_HOP_FLOW,
            "source": source,
            "target": target,
            "entities": [source, target],
            "search_terms": extract_search_terms(q)
        }

    # 6. Definition / Purpose Intent
    purpose_match = re.search(
        r"(?:what is the purpose of|what is the role of|what is the function of|what is the responsibility of)\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if not purpose_match:
        purpose_match = re.search(
            r"(?:purpose of|role of|function of)\s+(?:the\s+|an?\s+)?(.*?)(?:\?|$|\.)",
            q,
            re.IGNORECASE
        )
    if not purpose_match:
        purpose_match = re.search(
            r"(?:which|what)\s+(?:layer|component|module|service|entity|interface)\s+provides\s+(?:the\s+)?(.*?)(?:\?|$|\.)",
            q,
            re.IGNORECASE
        )

    if purpose_match:
        target = purpose_match.group(1).strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(?:in|within|for)\s+(?:the\s+)?.*$", "", target, flags=re.IGNORECASE).strip()
        return {
            "intent": QueryIntent.DEFINITION_PURPOSE,
            "target": target,
            "entities": [target],
            "search_terms": extract_search_terms(q)
        }

    # Provision / Deliverables Intent (e.g. What does X provide/offer/deliver/define/specify?)
    prov_match = re.search(
        r"(?:what\s+(?:does|do)\s+(?:the\s+|an?\s+)?(.*?)\s+(?:provide|offer|deliver|define|specify|bring)|"
        r"what\s+(?:is|are)\s+provided\s+by\s+(?:the\s+|an?\s+)?(.*?))"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if prov_match:
        target = [g for g in prov_match.groups() if g is not None][0].strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", target, flags=re.IGNORECASE).strip()
        ents = split_query_entities(target)
        if not ents and target:
            ents = [target]
        return {
            "intent": QueryIntent.DEFINITION_PURPOSE,
            "target": target,
            "entities": ents,
            "search_terms": extract_search_terms(q)
        }

    # 7. Structure / Components Intent (e.g. What does X consist of?, What are the components/parts/layers of X?)
    struct_match = re.search(
        r"(?:what\s+(?:does|do)\s+(.*?)\s+consist\s+of|"
        r"what\s+(?:is|are)\s+(?:the\s+)?(?:main\s+|major\s+|primary\s+|key\s+)?(?:components|parts|elements|constituents|layers|building\s+blocks|composition|subsystems)\s+of\s+(.*?)|"
        r"what\s+(?:is|are)\s+(.*?)\s+(?:composed\s+of|made\s+up\s+of)|"
        r"what\s+comprises\s+(?:the\s+)?(.*?)|"
        r"what\s+does\s+(.*?)\s+comprise)"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if struct_match:
        target = [g for g in struct_match.groups() if g is not None][0].strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", target, flags=re.IGNORECASE).strip()
        ents = split_query_entities(target)
        if not ents and target:
            ents = [target]
        return {
            "intent": QueryIntent.STRUCTURE_COMPONENTS,
            "target": target,
            "entities": ents,
            "search_terms": extract_search_terms(q)
        }

    # 8. Benefits / Advantages Intent (e.g. Why does X...?, What are the benefits/advantages of X?)
    benefit_match = re.search(
        r"(?:what\s+(?:are|is)\s+(?:the\s+)?(?:benefits?|advantages?|gains?|value\s+propositions?)\s+of\s+(.*?)|"
        r"what\s+advantages?\s+does\s+(.*?)\s+(?:provide|offer|bring)|"
        r"why\s+(?:should|would|do|does|is|are)\s+(?:.*?\s+)?(?:use|used|adopt|adopted|choose|chosen|leverage|leveraged|prefer|preferred|implement|implemented)\s*(.*?)|"
        r"why\s+is\s+(.*?)\s+used|"
        r"why\s+(?:does|do|is|are)\s+(.*?)|"
        r"what\s+value\s+does\s+(.*?)\s+bring)"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if benefit_match:
        target = [g for g in benefit_match.groups() if g is not None][0].strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", target, flags=re.IGNORECASE).strip()
        ents = split_query_entities(target)
        if not ents and target:
            ents = [target]
        return {
            "intent": QueryIntent.BENEFITS,
            "target": target,
            "entities": ents,
            "search_terms": extract_search_terms(q)
        }

    # 9. Mechanism / How it works Intent (e.g. How does X work?, What is the working principle of X?)
    mech_work_match = re.search(
        r"(?:how\s+(?:does|do)\s+(.*?)\s+(?:work|operate|function)|"
        r"what\s+is\s+the\s+(?:mechanism|working\s+principle|operating\s+principle)\s+of\s+(.*?))"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if mech_work_match:
        target = [g for g in mech_work_match.groups() if g is not None][0].strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", target, flags=re.IGNORECASE).strip()
        ents = split_query_entities(target)
        if not ents and target:
            ents = [target]
        return {
            "intent": QueryIntent.MECHANISM_HOW,
            "target": target,
            "entities": ents,
            "search_terms": extract_search_terms(q)
        }

    # 10. Process / Workflow Intent
    proc_match = re.search(
        r"(?:what\s+(?:is|are)\s+(?:the\s+)?(?:top-down\s+|bottom-up\s+)?(?:process|workflow|steps|procedure|sequence|lifecycle)\s+(?:of|for|to)\s+(.*?)|"
        r"how\s+to\s+(?:implement|configure|deploy|execute|generate)\s+(.*?))"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if proc_match:
        target = [g for g in proc_match.groups() if g is not None][0].strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", target, flags=re.IGNORECASE).strip()
        ents = split_query_entities(target)
        if not ents and target:
            ents = [target]
        return {
            "intent": QueryIntent.PROCESS_WORKFLOW,
            "target": target,
            "entities": ents,
            "search_terms": extract_search_terms(q)
        }

    # 11. Category / Classification List Intent
    cat_match = re.search(
        r"(?:which|what)\s+(?:are the\s+)?(.*?)\s+(?:categories|classes|types|groups|kinds)\s+(?:are\s+)?(?:shown|defined|listed|described|available)?(?:\s+in\s+.*?)?(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if cat_match:
        concept = cat_match.group(1).strip()
        ents = [concept]
        acrs = re.findall(r"\b[A-Z]{2,6}\b", concept)
        for a in acrs:
            if a not in ents:
                ents.append(a)
        return {
            "intent": QueryIntent.LIST_MULTI_PART,
            "target": concept,
            "entities": ents,
            "search_terms": extract_search_terms(q)
        }

    # 12. List / Multi-part Intent
    list_match = re.search(
        r"(?:what are the|list (?:all )?the|which are the)\s+(?:main\s+|major\s+)?(.*?)\s+and\s+(?:what is|what are)\s+(?:the\s+)?(?:responsibility|responsibilities|role|roles|purpose|function)s?\s+of each(?:\s+\w+)?(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if not list_match:
        list_match = re.search(
            r"(?:what are the|list (?:all )?the|which are the)\s+(?:four\s+|three\s+|main\s+|major\s+)?(.*?)(?:\?|$|\.)",
            q,
            re.IGNORECASE
        )

    if list_match:
        concept = list_match.group(1).strip()
        concept = re.sub(
            r'^(?:three\s+|four\s+|five\s+|main\s+|major\s+|primary\s+|key\s+)?(?:stages|steps|phases|parts|elements|components|categories|layers|pillars|principles|levels)\s+(?:shown\s+in|described\s+in|of|in|for)\s+',
            '',
            concept,
            flags=re.IGNORECASE
        ).strip()
        concept_ents = split_query_entities(concept)
        if not concept_ents and concept:
            concept_ents = [concept]
        return {
            "intent": QueryIntent.LIST_MULTI_PART,
            "target": concept,
            "entities": concept_ents,
            "search_terms": extract_search_terms(q)
        }

    # 13. Technical Identifier / Representation / Meaning Intent (e.g. What does X represent / stand for / mean?)
    rep_match = re.search(
        r"(?:what\s+does\s+(?:the\s+|an?\s+)?(.*?)\s+(?:represent|stand\s+for|mean|refer\s+to|signify|denote)|"
        r"what\s+is\s+represented\s+by\s+(?:the\s+|an?\s+)?(.*?))"
        r"(?:\s+(?:in|for|within|under)\s+(?:the\s+)?(.*?))?"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if rep_match:
        target = [g for g in rep_match.groups()[:2] if g is not None][0].strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", target, flags=re.IGNORECASE).strip()
        scope = rep_match.group(3).strip() if len(rep_match.groups()) >= 3 and rep_match.group(3) else ""
        scope = re.sub(r"^(?:the|a|an)\s+", "", scope, flags=re.IGNORECASE).strip()
        scope = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", scope, flags=re.IGNORECASE).strip()
        ents = [target]
        if scope and scope.lower() not in STOP_WORDS:
            ents.append(scope)
        return {
            "intent": QueryIntent.DEFINITION_PURPOSE,
            "target": target,
            "scope": scope,
            "entities": ents,
            "search_terms": extract_search_terms(q)
        }

    # 14. Definition / Description Intent (e.g. What is X?, What is the X approach?, Define X, Describe X, Explain X)
    def_match = re.search(
        r"(?:define\s+(?:the\s+|an?\s+)?(.*?)|"
        r"definition\s+of\s+(?:the\s+|an?\s+)?(.*?)|"
        r"describe\s+(?:the\s+|an?\s+)?(.*?)|"
        r"explain\s+(?:the\s+|an?\s+)?(.*?)|"
        r"what\s+is\s+(?:the\s+|an?\s+)?(.*?)\s+approach|"
        r"what\s+is\s+(?:the\s+|an?\s+)?(.*?)|"
        r"what\s+are\s+(?:the\s+|an?\s+)?(.*?))"
        r"(?:\?|$|\.)",
        q,
        re.IGNORECASE
    )
    if def_match:
        target = [g for g in def_match.groups() if g is not None][0].strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE).strip()
        target = re.sub(r"\s+(?:shown\s+)?in\s+(?:the\s+)?document.*$", "", target, flags=re.IGNORECASE).strip()
        ents = split_query_entities(target)
        if not ents and target:
            ents = [target]
        return {
            "intent": QueryIntent.DEFINITION_DESCRIPTION,
            "target": target,
            "entities": ents,
            "search_terms": extract_search_terms(q)
        }

    # 14. General Factual Query
    return {
        "intent": QueryIntent.GENERAL,
        "entities": [],
        "search_terms": extract_search_terms(q)
    }


def extract_search_terms(query: str) -> List[str]:
    """
    Extracts substantive non-stopword tokens, n-grams, and multi-word phrases from any query.
    Preserves contiguous natural language phrase spans for high-precision generic phrase retrieval.
    Preserves exact technical identifiers (e.g. ara::exec, sys/io.h).
    """
    terms = []

    # Preserve technical tokens with scopes/delimiters
    tech_tokens = re.findall(r"\b[A-Za-z0-9_]+::[A-Za-z0-9_]+\b", query)
    tech_tokens += re.findall(r"\b[A-Za-z0-9_]+/[A-Za-z0-9_]+\b", query)
    for tok in tech_tokens:
        tok_low = tok.lower().strip()
        terms.append(tok_low)
        terms.append(re.sub(r"[:/]+", " ", tok_low))

    clean = re.sub(r"[^\w\s\-]", " ", query)
    tokens = [w.lower() for w in clean.split() if w.strip()]
    content_tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 2]

    # 1. Contiguous natural language phrase spans (2 to 5 tokens from original query)
    # Only keep phrase spans that contain at least 2 content tokens
    for n in range(5, 1, -1):
        for i in range(len(tokens) - n + 1):
            span_tokens = tokens[i:i + n]
            content_in_span = [w for w in span_tokens if w not in STOP_WORDS and len(w) > 2]
            if len(content_in_span) >= 2:
                phrase = " ".join(span_tokens)
                terms.append(phrase)
                if "-" in phrase:
                    terms.append(phrase.replace("-", " "))

    # 2. Substantive content token n-grams (without stopwords)
    for i in range(len(content_tokens) - 1):
        terms.append(f"{content_tokens[i]} {content_tokens[i+1]}")
    for i in range(len(content_tokens) - 2):
        terms.append(f"{content_tokens[i]} {content_tokens[i+1]} {content_tokens[i+2]}")

    # 3. Individual content words
    for w in content_tokens:
        terms.append(w)
        if "-" in w:
            terms.append(w.replace("-", " "))

    # Remove duplicates preserving priority order (longer phrases first)
    seen = set()
    ordered_terms = []
    for t in terms:
        t_clean = t.strip()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            ordered_terms.append(t_clean)

    return ordered_terms
