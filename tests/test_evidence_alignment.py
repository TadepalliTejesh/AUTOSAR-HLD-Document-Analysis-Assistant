import pytest
from app.services.intent import classify_query, QueryIntent
from app.services.evidence_aligner import (
    score_intent_alignment,
    prune_and_rank_supporting_evidence,
    evaluate_evidence_confidence,
    validate_citations_against_answer
)


def test_intent_classification():
    """Verify generic intent classification across various query phrasing."""
    # Structure / Components
    cl1 = classify_query("What does the AUTOSAR middleware approach consist of?")
    assert cl1["intent"] == QueryIntent.STRUCTURE_COMPONENTS
    assert "middleware" in cl1["target"].lower()

    cl2 = classify_query("What are the components of the ECU Abstraction Layer?")
    assert cl2["intent"] == QueryIntent.STRUCTURE_COMPONENTS

    cl3 = classify_query("What is the system composed of?")
    assert cl3["intent"] == QueryIntent.STRUCTURE_COMPONENTS

    # Benefits
    cl4 = classify_query("What are the benefits of the AUTOSAR middleware approach?")
    assert cl4["intent"] == QueryIntent.BENEFITS
    assert "middleware" in cl4["target"].lower()

    cl5 = classify_query("What are the advantages of BSW modules?")
    assert cl5["intent"] == QueryIntent.BENEFITS

    # Mechanism
    cl6 = classify_query("How does the scheduler work?")
    assert cl6["intent"] == QueryIntent.MECHANISM_HOW

    # Definition / Purpose
    cl7 = classify_query("What is the purpose of the RTE?")
    assert cl7["intent"] == QueryIntent.DEFINITION_PURPOSE

    # Comparison
    cl8 = classify_query("What is the difference between MCAL and ECU Abstraction Layer?")
    assert cl8["intent"] == QueryIntent.COMPARISON


def test_intent_alignment_scoring():
    """Verify that structure queries prioritize structural layouts and penalize benefit chunks."""
    structural_chunk = (
        "Proprietary vs. AUTOSAR Middleware Approach\n"
        "Application Software\n"
        "Standardized Middleware\n"
        "Virtualization / OS / Hardware\n"
        "Hardware Specific ECU\n"
        "Standardized Methodology"
    )

    benefits_chunk = (
        "Benefits of the AUTOSAR Middleware Approach\n"
        "Hardware and software widely independent of each other.\n"
        "Distributed, parallel development through horizontal layers;\n"
        "therefore, reduced development time and costs.\n"
        "Enhanced quality and efficiency through software reuse."
    )

    # 1. Structure Query Scoring
    struct_score_p25 = score_intent_alignment(
        chunk_content=structural_chunk,
        chunk_section="AUTOSAR Basic Principles",
        intent=QueryIntent.STRUCTURE_COMPONENTS,
        entities=["AUTOSAR middleware approach"]
    )
    struct_score_p26 = score_intent_alignment(
        chunk_content=benefits_chunk,
        chunk_section="AUTOSAR Basic Principles",
        intent=QueryIntent.STRUCTURE_COMPONENTS,
        entities=["AUTOSAR middleware approach"]
    )

    # Structural chunk should score very high, benefits chunk should be heavily penalized
    assert struct_score_p25 >= 0.85
    assert struct_score_p26 <= 0.25

    # 2. Benefits Query Scoring
    benefit_score_p25 = score_intent_alignment(
        chunk_content=structural_chunk,
        chunk_section="AUTOSAR Basic Principles",
        intent=QueryIntent.BENEFITS,
        entities=["AUTOSAR middleware approach"]
    )
    benefit_score_p26 = score_intent_alignment(
        chunk_content=benefits_chunk,
        chunk_section="AUTOSAR Basic Principles",
        intent=QueryIntent.BENEFITS,
        entities=["AUTOSAR middleware approach"]
    )

    # Benefits chunk should score very high, structural chunk without benefits should be low
    assert benefit_score_p26 >= 0.90
    assert benefit_score_p25 <= 0.30


def test_pruning_filters_off_intent_chunks():
    """
    Verify prune_and_rank_supporting_evidence selects only direct supporting evidence
    and prunes nearby semantic chunks that discuss off-intent information.
    """
    candidates = [
        {
            "chunk_id": "c_benefits",
            "page": 26,
            "section": "AUTOSAR Basic Principles",
            "content": "Benefits of the AUTOSAR Middleware Approach\nReduced development time and costs\nEnhanced quality and efficiency through software reuse",
            "source_type": "text",
            "similarity": 0.87
        },
        {
            "chunk_id": "c_structure",
            "page": 25,
            "section": "AUTOSAR Basic Principles",
            "content": "Proprietary vs. AUTOSAR Middleware Approach\nApplication Software\nStandardized Middleware\nVirtualization / OS / Hardware\nHardware Specific ECU",
            "source_type": "text",
            "similarity": 0.79
        }
    ]

    # Query: Consist of (Structure)
    classification_struct = classify_query("What does the AUTOSAR middleware approach consist of?")
    supporting, conf, metrics = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query="What does the AUTOSAR middleware approach consist of?",
        classification=classification_struct
    )

    # Must select ONLY the structural chunk (page 25) and prune the benefits chunk (page 26)
    assert len(supporting) == 1
    assert supporting[0]["page"] == 25
    assert conf in ["High", "Medium"]

    # Query: Benefits
    classification_benefit = classify_query("What are the benefits of the AUTOSAR middleware approach?")
    supporting_ben, conf_ben, metrics_ben = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query="What are the benefits of the AUTOSAR middleware approach?",
        classification=classification_benefit
    )

    assert len(supporting_ben) >= 1
    assert supporting_ben[0]["page"] == 26
    assert conf_ben in ["High", "Medium"]


def test_insufficient_evidence_on_absent_entity():
    """Verify that asking about an entity not present in retrieved text results in Insufficient confidence."""
    candidates = [
        {
            "chunk_id": "c1",
            "page": 1,
            "section": "General Overview",
            "content": "This document outlines the guidelines for student project submissions.",
            "source_type": "text",
            "similarity": 0.55
        }
    ]

    classification = classify_query("What is the minimum CGPA required for students to apply?")
    classification["entities"] = ["minimum CGPA", "CGPA"]

    supporting, conf, metrics = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query="What is the minimum CGPA required for students to apply?",
        classification=classification
    )

    assert conf == "Insufficient"
    assert len(supporting) == 0


def test_one_page_sufficient_evidence():
    """Requirement: If one page completely supports the answer, cite only that page."""
    candidates = [
        {
            "chunk_id": "doc34_p8",
            "page": 8,
            "section": "AUTOSAR Mission",
            "content": (
                "AUTOSAR is a global partnership of leading companies in the automotive and software industry "
                "to develop and establish the standardized software framework and open E/E system architecture for intelligent mobility."
            ),
            "source_type": "text",
            "similarity": 0.88
        },
        {
            "chunk_id": "doc34_p9",
            "page": 9,
            "section": "AUTOSAR Vision",
            "content": (
                "AUTOSAR will be the global established standard for software and methodology enabling open E/E system architectures "
                "for future intelligent mobility supporting high levels of dependability, especially safety and security."
            ),
            "source_type": "text",
            "similarity": 0.84
        }
    ]

    query = "What is the AUTOSAR Mission?"
    classification = classify_query(query)

    supporting, conf, metrics = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query=query,
        classification=classification
    )

    # Pre-generation minimality: Page 8 completely supports the query; Page 9 must not be included
    assert len(supporting) == 1
    assert supporting[0]["page"] == 8
    assert conf in ["High", "Medium"]

    # Post-generation validation
    answer = (
        "The AUTOSAR Mission is to develop and establish a standardized software framework "
        "and open E/E system architecture for intelligent mobility through a global partnership."
    )
    validated = validate_citations_against_answer(answer, supporting)
    assert len(validated) == 1
    assert validated[0]["page"] == 8


def test_multi_page_necessary_evidence():
    """Requirement: Multi-page necessary evidence (e.g. comparison or multi-aspect)."""
    candidates = [
        {
            "chunk_id": "doc17_p24",
            "page": 24,
            "section": "Microcontroller Abstraction Layer",
            "content": (
                "The Microcontroller Abstraction Layer (MCAL) is the lowest software layer of the Basic Software. "
                "It contains internal drivers with direct access to the microcontroller and internal peripherals."
            ),
            "source_type": "text",
            "similarity": 0.85
        },
        {
            "chunk_id": "doc17_p25",
            "page": 25,
            "section": "ECU Abstraction Layer",
            "content": (
                "The ECU Abstraction Layer interfaces the drivers of the Microcontroller Abstraction Layer and contains "
                "drivers for external devices. It offers an API for access to peripherals regardless of their location."
            ),
            "source_type": "text",
            "similarity": 0.84
        }
    ]

    query = "What is the difference between the Microcontroller Abstraction Layer and ECU Abstraction Layer?"
    classification = classify_query(query)

    supporting, conf, metrics = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query=query,
        classification=classification
    )

    # Multi-page necessary: Both pages must be selected
    assert len(supporting) == 2
    pages = [c["page"] for c in supporting]
    assert 24 in pages and 25 in pages

    # Post-generation validation preserves both pages when both claims are in the answer
    answer = (
        "The Microcontroller Abstraction Layer contains internal drivers with direct access to the microcontroller. "
        "In contrast, the ECU Abstraction Layer interfaces MCAL drivers and offers an API for external peripherals."
    )
    validated = validate_citations_against_answer(answer, supporting)
    assert len(validated) == 2
    val_pages = [c["page"] for c in validated]
    assert 24 in val_pages and 25 in val_pages


def test_related_but_unsupported_neighboring_page():
    """Requirement: Do not cite neighboring pages merely because they discuss the same topic."""
    candidates = [
        {
            "chunk_id": "doc34_p26",
            "page": 26,
            "section": "AUTOSAR Basic Principles",
            "content": (
                "Benefits of the AUTOSAR Middleware Approach\n"
                "Hardware and software widely independent of each other.\n"
                "Distributed, parallel development (by abstraction) through horizontal layers; therefore, reduced development time and costs."
            ),
            "source_type": "text",
            "similarity": 0.84
        },
        {
            "chunk_id": "doc34_p25",
            "page": 25,
            "section": "AUTOSAR Basic Principles",
            "content": (
                "Proprietary vs. AUTOSAR Middleware Approach\n"
                "Standardized Middleware\n"
                "Virtualization / OS / Hardware\n"
                "Hardware Specific ECU"
            ),
            "source_type": "text",
            "similarity": 0.81
        },
        {
            "chunk_id": "doc34_p9",
            "page": 9,
            "section": "AUTOSAR Vision",
            "content": (
                "AUTOSAR Vision\n"
                "AUTOSAR will be the global established standard for software and methodology enabling open E/E system architectures."
            ),
            "source_type": "text",
            "similarity": 0.78
        }
    ]

    query = "How does AUTOSAR achieve hardware and software independence?"
    classification = classify_query(query)

    supporting, conf, metrics = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query=query,
        classification=classification
    )

    # Page 26 is completely sufficient. Pages 25 and 9 are topically related neighbors and must be excluded.
    assert len(supporting) == 1
    assert supporting[0]["page"] == 26
    assert 25 not in [c["page"] for c in supporting]
    assert 9 not in [c["page"] for c in supporting]


def test_negative_evidence():
    """Requirement: If evidence is absent/insufficient, return Insufficient and zero supporting citations."""
    candidates = [
        {
            "chunk_id": "doc_eng_1",
            "page": 3,
            "section": "System Requirements",
            "content": "The system shall operate within a voltage range of 9V to 16V DC across all operating temperatures.",
            "source_type": "text",
            "similarity": 0.48
        }
    ]

    query = "What is the minimum CGPA required for students to apply?"
    classification = classify_query(query)

    supporting, conf, metrics = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query=query,
        classification=classification
    )

    assert conf == "Insufficient"
    assert len(supporting) == 0


def test_definition_vs_benefits():
    """
    Regression Test:
    - Definition queries ('What is X?', 'What is the X approach?') must select the definition/concept chunk (Page 25)
      and reject the benefits chunk (Page 26).
    - Benefits queries ('Why does X...?', 'What are the benefits of X?') must select the benefits chunk (Page 26).
    - If only benefits are available for a definition query, must return Insufficient.
    """
    candidates = [
        {
            "chunk_id": "p26_ben",
            "page": 26,
            "section": "AUTOSAR Basic Principles",
            "content": (
                "Benefits of the AUTOSAR Middleware Approach\n"
                "• Hardware and software widely independent of each other\n"
                "• Distributed, parallel development (by abstraction) through horizontal layers; therefore, reduced development time and costs\n"
                "• Enhanced quality and efficiency through software reuse"
            ),
            "source_type": "text",
            "similarity": 0.88
        },
        {
            "chunk_id": "p25_def",
            "page": 25,
            "section": "AUTOSAR Basic Principles",
            "content": (
                "Proprietary vs. AUTOSAR Middleware Approach\n"
                "Standardized Middleware\n"
                "Virtualization / OS / Hardware\n"
                "Hardware Specific ECU\n"
                "Standardized Methodology\n"
                "Application Software"
            ),
            "source_type": "text",
            "similarity": 0.82
        }
    ]

    # 1. Definition Query: "What is the AUTOSAR middleware approach?"
    q_def = "What is the AUTOSAR middleware approach?"
    cl_def = classify_query(q_def)
    assert cl_def["intent"] == QueryIntent.DEFINITION_DESCRIPTION

    supp_def, conf_def, _ = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query=q_def,
        classification=cl_def
    )
    assert len(supp_def) == 1
    assert supp_def[0]["page"] == 25, f"Expected Page 25 for definition, got Page {supp_def[0]['page']}"

    # 2. Benefits Query: "Why does AUTOSAR use a middleware approach?"
    q_why = "Why does AUTOSAR use a middleware approach?"
    cl_why = classify_query(q_why)
    assert cl_why["intent"] == QueryIntent.BENEFITS

    supp_why, conf_why, _ = prune_and_rank_supporting_evidence(
        chunks=candidates,
        query=q_why,
        classification=cl_why
    )
    assert len(supp_why) == 1
    assert supp_why[0]["page"] == 26, f"Expected Page 26 for benefits, got Page {supp_why[0]['page']}"

    # 3. Only benefits chunk available for definition question -> Insufficient evidence
    supp_only_ben, conf_only_ben, _ = prune_and_rank_supporting_evidence(
        chunks=[candidates[0]],  # only Page 26 benefits
        query=q_def,
        classification=cl_def
    )
    assert conf_only_ben == "Insufficient"
    assert len(supp_only_ben) == 0


def test_benefits_vs_structure():
    """
    Regression Test:
    - Benefits query ('What are the benefits of X?') selects Page 26.
    - Structure query ('What does X consist of?') selects Page 25.
    """
    candidates = [
        {
            "chunk_id": "p26_ben",
            "page": 26,
            "section": "AUTOSAR Basic Principles",
            "content": "Benefits of the AUTOSAR Middleware Approach\n• Reduced development time and costs\n• Software reuse",
            "source_type": "text",
            "similarity": 0.89
        },
        {
            "chunk_id": "p25_struct",
            "page": 25,
            "section": "AUTOSAR Basic Principles",
            "content": "Proprietary vs. AUTOSAR Middleware Approach\nStandardized Middleware\nVirtualization / OS / Hardware\nHardware Specific ECU",
            "source_type": "text",
            "similarity": 0.84
        }
    ]

    # Structure
    q_struct = "What does the AUTOSAR middleware approach consist of?"
    cl_struct = classify_query(q_struct)
    assert cl_struct["intent"] == QueryIntent.STRUCTURE_COMPONENTS
    supp_struct, _, _ = prune_and_rank_supporting_evidence(candidates, q_struct, cl_struct)
    assert len(supp_struct) == 1
    assert supp_struct[0]["page"] == 25

    # Benefits
    q_ben = "What are the benefits of the AUTOSAR middleware approach?"
    cl_ben = classify_query(q_ben)
    assert cl_ben["intent"] == QueryIntent.BENEFITS
    supp_ben, _, _ = prune_and_rank_supporting_evidence(candidates, q_ben, cl_ben)
    assert len(supp_ben) == 1
    assert supp_ben[0]["page"] == 26


def test_paraphrased_phrase_retrieval():
    """
    Regression Test:
    Important phrases from query (e.g. 'top-down workflow for distributed development')
    must match the exact evidence (Page 36) over adjacent distributed development mentions (Page 21).
    """
    candidates = [
        {
            "chunk_id": "p21_suppliers",
            "page": 21,
            "section": "AUTOSAR Basic Principles",
            "content": "Benefits of Applying the AUTOSAR SW Framework\n• Supports distributed development among suppliers\n• Standardizes non-competitive SW",
            "source_type": "text",
            "similarity": 0.81
        },
        {
            "chunk_id": "p36_adaptive",
            "page": 36,
            "section": "AUTOSAR Adaptive Platform",
            "content": (
                "AUTOSAR Adaptive Platform\nThree Pillars for ADAS Applications\n"
                "• E/E Architecture development\n"
                "• Top-down workflow for distributed development\n"
                "• Formal exchange formats"
            ),
            "source_type": "text",
            "similarity": 0.77
        }
    ]

    query = "What is the top-down workflow for distributed development in AUTOSAR?"
    classification = classify_query(query)
    supp, conf, _ = prune_and_rank_supporting_evidence(candidates, query, classification)

    assert len(supp) >= 1
    assert supp[0]["page"] == 36, f"Expected Page 36 with exact phrase, got Page {supp[0]['page']}"
    assert conf in ["High", "Medium"]


def test_proprietary_claim_negative_test():
    """
    Regression Test:
    Ensures that questions regarding proprietary claims or architecture correctly contrast
    'Proprietary Solution' with AUTOSAR 'Standardized Middleware' and reject unsupported claims.
    """
    chunk_p25 = {
        "chunk_id": "p25_prop_vs_autosar",
        "page": 25,
        "section": "AUTOSAR Basic Principles",
        "content": (
            "Proprietary vs. AUTOSAR Middleware Approach\n"
            "Proprietary Solution:\n"
            "Application Software\n"
            "Basic Software\n"
            "Hardware\n\n"
            "AUTOSAR:\n"
            "Application Software\n"
            "Standardized Middleware\n"
            "Virtualization / OS / Hardware\n"
            "Hardware Specific ECU\n"
            "Standardized Methodology"
        ),
        "source_type": "text",
        "similarity": 0.85
    }

    # Query asking if AUTOSAR is proprietary
    query = "Does AUTOSAR use a proprietary middleware approach?"
    cl = classify_query(query)
    supp, conf, _ = prune_and_rank_supporting_evidence([chunk_p25], query, cl)
    assert len(supp) == 1
    assert supp[0]["page"] == 25
    assert "Standardized Middleware" in supp[0]["content"]

    # Unsupported query: proprietary encryption protocol
    unsupported_query = "What proprietary encryption algorithm is implemented in the AUTOSAR middleware?"
    cl_un = classify_query(unsupported_query)
    supp_un, conf_un, _ = prune_and_rank_supporting_evidence([chunk_p25], unsupported_query, cl_un)
    # The chunk contains no encryption algorithm evidence
    assert conf_un == "Insufficient"


def test_stage_progression_question_meta_words():
    """
    Regression Test:
    Ensures question meta words like 'three' and 'stages' are filtered out from
    missing_substantive check so diagrams showing progression boxes evaluate to High confidence.
    """
    chunk_p25 = {
        "chunk_id": "doc52_p25",
        "page": 25,
        "section": "AUTOSAR Adaptive Platform Development Approach",
        "content": (
            "AuToSAR Adaptive Platform Development Approach\n"
            "Specification\n"
            "Implementation\n"
            "Demonstration\n"
            "Identify needs & use-cases:\n"
            "1) Concepts 2) Features 3) Requirements\n"
            "RS Documents, SWS Documents\n"
            "Gain speed:\n"
            "1) Spec validation 2) Reduce room for spec interpretation 3) Training / dissemination of AP\n"
            "Attracting environment for coders:\n"
            "Appealing technology (C++, Yocto, Git, …)\n"
            "Gain trust: 1) Advertises the progress 2) Highlights some specific features\n"
            "Quality: TF-ARC approval, Cross team review\n"
            "Show AUTOSAR interoperability of classic and adaptive platforms"
        ),
        "source_type": "text",
        "similarity": 0.85
    }

    query = "What are the three main stages shown in the AUTOSAR Adaptive Platform Development Approach?"
    cl = classify_query(query)
    supp, conf, metrics = prune_and_rank_supporting_evidence([chunk_p25], query, cl)

    assert len(supp) == 1
    assert supp[0]["page"] == 25
    assert conf in ["High", "Medium"]
    assert metrics["composite_score"] >= 0.50


def test_logical_view_part_retrieval_and_section_preference():
    """
    Regression Test:
    Ensures query for 'logical view' prioritizes logical view section over development approach,
    and includes multi-part chunks from the same page.
    """
    chunk_p9_1 = {
        "chunk_id": "doc52_p9_c1",
        "page": 9,
        "section": "Logical view",
        "content": (
            "User Applications\n"
            "AUTOSAR Runtime for Adaptive Applications (ARA)\n"
            "(Virtual) Machine / Container / Hardware\n"
            "ara::com Communication Management"
        ),
        "source_type": "text",
        "similarity": 0.78
    }
    chunk_p9_2 = {
        "chunk_id": "doc52_p9_c2",
        "page": 9,
        "section": "Logical view",
        "content": (
            "Signal PDU\n"
            "ara::crypto Cryptography\n"
            "Adaptive Application\n"
            "POSIX PSE51 / C++ STL Operating System Interface\n"
            "ara::exec Execution Management\n"
            "API Platform Foundation FCs\n"
            "SERVICE Platform Service FCs"
        ),
        "source_type": "text",
        "similarity": 0.73
    }
    chunk_p25 = {
        "chunk_id": "doc52_p25_dev",
        "page": 25,
        "section": "AUTOSAR Adaptive Platform Development Approach",
        "content": "AuToSAR Adaptive Platform Development Approach Specification Implementation Demonstration",
        "source_type": "text",
        "similarity": 0.82
    }

    query = "What are the main parts of the AUTOSAR Adaptive Platform logical view?"
    cl = classify_query(query)
    supp, conf, metrics = prune_and_rank_supporting_evidence([chunk_p9_2, chunk_p25, chunk_p9_1], query, cl)

    # Must prioritize page 9 and include both parts on page 9, excluding page 25
    assert len(supp) >= 1
    assert all(c["page"] == 9 for c in supp)
    assert conf in ["High", "Medium"]


def test_technical_identifier_resolution():
    """
    Regression Test:
    Ensures technical identifiers like ara::exec are resolved to their full cluster name.
    """
    chunk_p9 = {
        "chunk_id": "doc52_p9_c2",
        "page": 9,
        "section": "Logical view",
        "content": "AUTOSAR Adaptive Platform\nLogical View\nara::exec Execution Management\nara::com Communication Management\nara::diag Diagnostics",
        "source_type": "text",
        "similarity": 0.81
    }

    query = "What does ara::exec represent in the AUTOSAR Adaptive Platform?"
    cl = classify_query(query)
    supp, conf, _ = prune_and_rank_supporting_evidence([chunk_p9], query, cl)

    assert len(supp) == 1
    assert supp[0]["page"] == 9
    assert conf in ["High", "Medium"]


def test_tautological_definition_and_layer_separation_sanitization():
    """
    Regression Test:
    Ensures sanitize_llm_output detects tautological definitions and layer separation absence,
    triggering full direct grounded synthesis.
    """
    from app.services.rag import sanitize_llm_output

    chunk_p6 = {
        "chunk_id": "doc52_p6",
        "page": 6,
        "section": "Layered Software Architecture (1/2)",
        "content": (
            "The layered architecture of the classic platform basically supports\n"
            "• Hardware abstraction\n"
            "Microcontroller\n"
            "Microcontroller Abstraction Layer\n"
            "ECU Abstraction Layer\n"
            "Services Layer\n"
            "Runtime Environment\n"
            "Application Layer\n"
            "Complex Drivers"
        ),
        "source_type": "text",
        "similarity": 0.85
    }

    # 1. Tautological definition
    tautological_ans = "The Microcontroller Abstraction Layer is a software layer."
    sanitized_def = sanitize_llm_output(tautological_ans, [chunk_p6], question="What is the Microcontroller Abstraction Layer?")
    assert "hardware" in sanitized_def.lower()
    assert "abstraction" in sanitized_def.lower()

    # 2. Incomplete layer hierarchy (missing complex drivers or hardware platform)
    incomplete_layers = "1. Application Layer\n2. Runtime Environment\n3. Services Layer\n4. ECU Abstraction Layer\n5. Microcontroller Abstraction Layer"
    sanitized_layers = sanitize_llm_output(incomplete_layers, [chunk_p6], question="What are the layers of the AUTOSAR Classic Platform?")
    assert "software architecture layers" in sanitized_layers.lower()
    assert "complex drivers" in sanitized_layers.lower()
    assert "microcontroller" in sanitized_layers.lower()


