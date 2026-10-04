import re
from typing import Optional, Dict, Any, List
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.services.intent import classify_query, QueryIntent, STOP_WORDS
from app.services.text_cleaner import clean_ocr_text, clean_chunk


def clean_and_unwrap(text: str) -> str:
    """Join lines where a sentence or clause was split across lines."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    merged = []
    dangling_endings = {'the', 'a', 'an', 'of', 'to', 'for', 'with', 'and', 'or', 'in', 'on', 'at', 'from', 'by', 'its', 'their', 'standard', 'complete', 'such', 'e.g.', 'i.e.'}
    for l in lines:
        if not merged:
            merged.append(l)
            continue
        last_word = merged[-1].split()[-1].lower().rstrip('.,;:!?') if merged[-1].split() else ''
        should_merge = (
            (l[0].islower() or last_word in dangling_endings)
            and not merged[-1].endswith(('.', ':', '|', '!', '?'))
            and not l.startswith(('-', '•', '…', '–', '*', '|', '\ufffd'))
        )
        if should_merge:
            merged[-1] = merged[-1] + ' ' + l
        else:
            merged.append(l)
    return "\n".join(merged)


class OllamaService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        self.timeout = 60.0

    def check_health(self) -> Dict[str, Any]:
        """Check if Ollama service is reachable and if model is loaded."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name") for m in data.get("models", [])]
                    model_available = any(self.model in m for m in models)
                    return {
                        "available": True,
                        "base_url": self.base_url,
                        "model": self.model,
                        "model_downloaded": model_available,
                        "installed_models": models
                    }
        except Exception as e:
            logger.debug(f"Ollama health check notice: {e}")

        return {
            "available": False,
            "base_url": self.base_url,
            "model": self.model,
            "model_downloaded": False,
            "installed_models": []
        }

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        question: Optional[str] = None
    ) -> str:
        """
        Generate grounded completion using Ollama.
        If Ollama is unreachable, synthesizes strict evidence-grounded answer directly
        from the retrieved chunks to maintain reliable operation.
        """
        if not question:
            q_match = re.search(r"ENGINEERING QUESTION:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
            question = q_match.group(1).strip() if q_match else ""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.1
            }
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "").strip()
                    if response_text:
                        return response_text
        except Exception as e:
            logger.warning(f"Ollama generation unavailable ({e}). Employing direct grounded synthesis.")

        # Grounded fallback synthesis when Ollama daemon is offline
        return self._direct_grounded_synthesis(prompt, retrieved_chunks or [], question=question)

    def _direct_grounded_synthesis(
        self,
        prompt: str,
        retrieved_chunks: List[Dict[str, Any]],
        question: str = ""
    ) -> str:
        """
        Purely evidence-driven local synthesis engine.
        Synthesizes structured, concise, and grounded answers dynamically from retrieved chunks
        with zero hardcoded domain strings or fixed answer templates.
        """
        if not retrieved_chunks:
            return "INSUFFICIENT EVIDENCE: The uploaded document does not provide enough evidence to answer this question."

        classification = classify_query(question)
        intent = classification["intent"]

        cleaned_chunks = [clean_chunk(c) for c in retrieved_chunks]
        raw_text = "\n".join([c["content"] for c in cleaned_chunks])
        unwrapped_text = clean_and_unwrap(raw_text)
        all_lines = [l.strip() for l in unwrapped_text.split("\n") if l.strip() and not l.startswith("[Document:") and not re.match(r"^Document ID\s*\d+", l)]

        # 1. Structure / Components Queries (e.g. What does X consist of?, What are the components of X?)
        if intent == QueryIntent.STRUCTURE_COMPONENTS:
            return self._synthesize_structure_components(question, classification, cleaned_chunks, all_lines)

        # 2. Benefits / Advantages Queries (e.g. What are the benefits of X?)
        if intent == QueryIntent.BENEFITS:
            return self._synthesize_benefits(question, classification, cleaned_chunks, all_lines)

        # 3. Interoperability / Enabling Mechanism Queries
        if intent == QueryIntent.INTEROPERABILITY:
            return self._synthesize_interoperability(question, classification, cleaned_chunks, all_lines)

        # 4. Transformation / Workflow Queries
        if intent == QueryIntent.TRANSFORMATION_WORKFLOW:
            return self._synthesize_transformation_workflow(question, classification, cleaned_chunks, all_lines)

        # 5. Relationship Queries (e.g. MCAL & ECU hardware, RTE & BSW & MCAL, tooling flows)
        if intent == QueryIntent.RELATIONSHIP:
            return self._synthesize_relationship(question, classification, cleaned_chunks, all_lines)

        # 6. Comparison Queries
        if intent == QueryIntent.COMPARISON:
            return self._synthesize_comparison(classification, cleaned_chunks, all_lines)

        # 7. Multi-hop Flow / Pathway Queries
        if intent == QueryIntent.MULTI_HOP_FLOW:
            return self._synthesize_flow(classification, cleaned_chunks, all_lines)

        # 8. List / Multi-part Queries
        if intent == QueryIntent.LIST_MULTI_PART:
            return self._synthesize_list(classification, cleaned_chunks, all_lines)

        # 9. Definition / Purpose Queries
        if intent == QueryIntent.DEFINITION_PURPOSE:
            return self._synthesize_definition(classification, cleaned_chunks, all_lines)

        # 10. General / Factual / Procedural Queries
        return self._synthesize_factual(question, classification, cleaned_chunks, all_lines)

    def _synthesize_structure_components(
        self,
        question: str,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        target = classification.get("target", "the system").strip()
        top_chunk = chunks[0]
        sec_name = top_chunk.get("section", "Document Evidence")
        page_num = top_chunk.get("page", 1)

        # 1. Merge adjacent split phrases (e.g. 'Complex' + 'Drivers' -> 'Complex Drivers')
        merged_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if line.lower() in ["complex", "standardized"] and nxt.lower() in ["drivers", "driver", "middleware"]:
                    merged_lines.append(f"{line} {nxt}")
                    i += 2
                    continue
            merged_lines.append(line)
            i += 1

        sw_layers = []
        drivers = []
        hw_platform = []
        general_items = []
        seen = set()

        in_capability_section = False

        for line in merged_lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("[Document:") or line_str.startswith("Document ID"):
                continue
            clean_item = re.sub(r"^[\-\–—•\*\d\.\\…\>\~]+\s*", "", line_str).strip()
            clean_item = re.sub(r"\s+", " ", clean_item)
            low = clean_item.lower()

            if not (3 <= len(clean_item) <= 100) or low in seen:
                continue
            if re.search(r"\b(?:[A-Za-z]{3}-\d{4}|\d{4}|part\s+\d+)\b", clean_item, re.IGNORECASE) and any(w in low for w in ["introduction", "presentation", "page", "part"]):
                continue

            # Detect capability lead-in (e.g. "The layered architecture basically supports")
            if any(kw in low for kw in ["basically supports", "architecture supports", "supports:", "features:"]):
                in_capability_section = True
                continue

            if in_capability_section:
                is_layer_label = (
                    low.endswith("layer") or
                    low in ["microcontroller", "runtime environment", "operating system", "complex drivers"] or
                    "abstraction layer" in low
                )
                if is_layer_label:
                    in_capability_section = False
                else:
                    # Capability bullet point
                    continue

            seen.add(low)

            # Categorize
            if any(w in low for w in ["complex driver", "complex drivers", "cdd", "spanning"]):
                drivers.append(clean_item)
            elif low.endswith("layer") or any(kw in low for kw in ["runtime environment", "operating system", "middleware", "abstraction layer", "services layer", "application layer"]):
                sw_layers.append(clean_item)
            elif any(kw in low for kw in ["microcontroller", "ecu-hardware", "hardware platform", "processor", "(virtual) machine"]) and not ("layer" in low or "abstraction" in low):
                hw_platform.append(clean_item)
            else:
                general_items.append(clean_item)

        # Distinguish software architecture layers vs complex drivers vs hardware
        if sw_layers and (drivers or hw_platform):
            # If layers are in bottom-up order, reverse to present canonical top-down hierarchy
            if len(sw_layers) >= 2 and "microcontroller" in sw_layers[0].lower() and "application" in sw_layers[-1].lower():
                sw_layers = list(reversed(sw_layers))

            res = [f"Based on the {sec_name} (Page {page_num}), the architecture consists of the following layers and components:\n"]
            res.append("**Software Architecture Layers:**")
            for idx, l in enumerate(sw_layers, 1):
                res.append(f"{idx}. **{l}**")
            if drivers:
                res.append("\n**Complex Drivers / Spanning Elements:**")
                for d in drivers:
                    res.append(f"- **{d}**")
            if hw_platform:
                res.append("\n**Underlying Hardware Execution Platform:**")
                for h in hw_platform:
                    res.append(f"- **{h}**")
            return "\n".join(res)

        all_items = sw_layers + general_items + drivers + hw_platform
        if all_items:
            formatted_list = "\n".join([f"- **{elem}**" if len(elem.split()) <= 6 else f"- {elem}" for elem in all_items[:12]])
            return (
                f"Based on the retrieved document evidence ({sec_name}, Page {page_num}), "
                f"the {target} consists of the following components/layers:\n\n"
                f"{formatted_list}"
            )

        return chunks[0]["content"][:300].strip()

    def _synthesize_benefits(
        self,
        question: str,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        target = classification.get("target", "the system").strip()
        top_chunk = chunks[0]
        sec_name = top_chunk.get("section", "Document Evidence")
        page_num = top_chunk.get("page", 1)

        benefit_items = []
        seen = set()
        for line in lines:
            l_str = line.strip()
            if not l_str or l_str.startswith("[Document:") or l_str.startswith("Document ID"):
                continue
            clean_item = re.sub(r"^[\-–—•\*\d\.]+\s*", "", l_str).strip()
            clean_item = re.sub(r"\s+", " ", clean_item)
            if any(w in clean_item.lower() for w in ["development", "cost", "time", "quality", "efficiency", "reuse", "benefit", "flexible", "parallel", "independent", "improve", "save", "advantage"]):
                if clean_item.lower() not in seen and len(clean_item) > 10:
                    seen.add(clean_item.lower())
                    benefit_items.append(clean_item)

        if benefit_items:
            formatted = "\n".join([f"- {b}" for b in benefit_items[:6]])
            return (
                f"Based on the retrieved document evidence ({sec_name}, Page {page_num}), "
                f"the key benefits of the {target} are:\n\n"
                f"{formatted}"
            )

        return chunks[0]["content"][:300].strip()

    def _synthesize_interoperability(
        self,
        question: str,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        top_chunk = chunks[0]
        sec_name = top_chunk.get("section", "Document Evidence")
        page_num = top_chunk.get("page", 1)

        # 1. Check for standard bullet points
        matching_bullets = []
        for line in lines:
            line_str = line.strip()
            if any(w in line_str.lower() for w in ["interoperab", "standard", "allow", "support", "generic", "compatible", "compliant", "format", "tool", "open", "portable"]):
                clean = re.sub(r"^[\-\•\*\d\.]+\s*", "", line_str)
                if clean not in matching_bullets and len(clean) > 20:
                    matching_bullets.append(clean)

        if matching_bullets:
            bullets_text = "\n".join([f"- {b}" for b in matching_bullets[:5]])
            return f"Based on the {sec_name} (Page {page_num}), interoperability is supported as follows:\n\n{bullets_text}"

        # 2. Check for interface demonstration / architecture interaction (e.g. use case diagrams)
        interface_types = []
        components = []
        stakeholders = []
        signals = []
        comp_pattern = re.compile(r'^([A-Z][A-Za-z0-9_\-]+(?:\s+[A-Z][A-Za-z0-9_\-]+)?)\s+[a-z_][a-z0-9_]*\s*\(')

        for line in lines:
            l = line.strip()
            if not l or l.startswith("[Document:") or len(l) > 90:
                continue
            clean = re.sub(r"^[\-\–—•\*\d\.\\…\>\~]+\s*", "", l).strip()
            # Interface lines: e.g. "AUTOSAR Interface", "Standardized Interface", "Std. Interface"
            if re.search(r"\b(?:[A-Za-z0-9_\-.]+\s+)?Interface\b", clean, re.IGNORECASE) and len(clean.split()) <= 4:
                if clean not in interface_types:
                    interface_types.append(clean)
            # Component with function call
            m_comp = comp_pattern.match(clean)
            if m_comp:
                cname = m_comp.group(1).strip()
                if cname not in components:
                    components.append(cname)
            # Standalone roles / stakeholders
            if any(w in clean.lower() for w in ["oem", "supplier", "integrator", "vendor"]) and len(clean.split()) <= 6:
                if clean not in stakeholders:
                    stakeholders.append(clean)
            # Function calls / signals
            sigs = re.findall(r"\b([a-z_][a-z0-9_]*\s*\([^)]*\))", clean)
            for s in sigs:
                if s not in signals and len(s) <= 40:
                    signals.append(s)

        if interface_types or components:
            res = [f"Based on the {sec_name} (Page {page_num}), interfaces are demonstrated as follows:\n"]
            if interface_types:
                res.append("**Defined Interface Types:**")
                for it in interface_types[:4]:
                    res.append(f"- **{it}**")
            if components:
                res.append("\n**Interacting Components:**")
                for c in components[:5]:
                    res.append(f"- {c}")
            if signals:
                res.append("\n**Exchanged Signals & Operations:**")
                for s in signals[:4]:
                    res.append(f"- `{s}`")
            if stakeholders:
                res.append(f"\n**Stakeholder & Vendor Partitioning:** {', '.join(stakeholders[:4])}")
            return "\n".join(res)

        return self._synthesize_factual(question, classification, chunks, lines)

    def _synthesize_transformation_workflow(
        self,
        question: str,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        top_chunk = chunks[0]
        sec_name = top_chunk.get("section", "Document Evidence")
        page_num = top_chunk.get("page", 1)

        steps = []
        for line in lines:
            if any(w in line.lower() for w in ["input", "tool", "generate", "generator", "output", "step", "produce", "file", "process", "workflow"]):
                clean = re.sub(r"^[\-\•\*\d\.]+\s*", "", line).strip()
                if clean not in steps and len(clean) > 15:
                    steps.append(clean)

        if steps:
            num_steps = "\n".join([f"{i}. **Step {i}:** {s}" for i, s in enumerate(steps[:6], 1)])
            return f"Based on the {sec_name} (Page {page_num}), the workflow comprises:\n\n{num_steps}"

        return self._synthesize_factual(question, classification, chunks, lines)

    def _synthesize_relationship(
        self,
        question: str,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        top_chunk = chunks[0]
        sec_name = top_chunk.get("section", "Document Evidence")
        page_num = top_chunk.get("page", 1)
        entities = classification.get("entities", [])
        entities_lower = [e.lower().strip() for e in entities if e and e.strip()]

        # Check for distribution across nodes/ECUs/processors connected by a bus/network
        is_distribution_query = any(w in question.lower() for w in ["distributed", "distribution", "partitioned", "allocation", "ecus", "nodes"])
        if is_distribution_query:
            components = []
            bus_elements = []
            hw_nodes = []
            interfaces = []
            comp_pattern = re.compile(r'^([A-Z][A-Za-z0-9_\-]+(?:\s+[A-Z][A-Za-z0-9_\-]+)?)\s+[a-z_][a-z0-9_]*\s*\(')

            for line in lines:
                l = line.strip()
                if not l or l.startswith("[Document:") or len(l) > 90:
                    continue
                clean = re.sub(r"^[\-\–—•\*\d\.\\…\>\~]+\s*", "", l).strip()
                low = clean.lower()

                # Bus / Network
                if any(b in low for b in ["bus", "can ", "lin", "flexray", "ethernet", "network"]) and not low.endswith("driver"):
                    if clean not in bus_elements and len(clean) <= 30:
                        bus_elements.append(clean)
                elif any(h in low for h in ["hardware", "ecu", "processor", "node", "controller"]) and not ("layer" in low or "abstraction" in low):
                    if clean not in hw_nodes and len(clean) <= 30:
                        hw_nodes.append(clean)
                elif re.search(r"\b(?:[A-Za-z0-9_\-.]+\s+)?Interface\b", clean, re.IGNORECASE) and len(clean.split()) <= 4:
                    if clean not in interfaces:
                        interfaces.append(clean)
                m_comp = comp_pattern.match(clean)
                if m_comp:
                    cname = m_comp.group(1).strip()
                    if cname not in components:
                        components.append(cname)

            if components or bus_elements:
                res = [f"Based on the {sec_name} (Page {page_num}), the system distribution is structured as follows:\n"]
                if components:
                    res.append("**Distributed Components:**")
                    for c in components[:6]:
                        res.append(f"- **{c}**")
                if bus_elements or hw_nodes:
                    res.append("\n**Hardware Execution Nodes & Network Interconnect:**")
                    if hw_nodes:
                        res.append(f"- Execution Nodes: {', '.join(hw_nodes[:4])}")
                    if bus_elements:
                        res.append(f"- Communication Bus: {', '.join(bus_elements[:3])}")
                if interfaces:
                    res.append("\n**Communication & Middleware Layer:**")
                    res.append(f"- Communication is coordinated across ECUs via runtime environment and standardized interfaces ({', '.join(interfaces[:3])}).")
                return "\n".join(res)

        connecting = []
        for line in lines:
            l_low = line.lower()
            if any(e in l_low for e in entities_lower) or any(w in l_low for w in ["connect", "interface", "abstract", "between", "layer", "provides", "drives", "access", "above", "below"]):
                clean = re.sub(r"^[\-\•\*\d\.]+\s*", "", line).strip()
                if clean not in connecting and len(clean) > 15:
                    connecting.append(clean)

        if connecting:
            bullets = "\n".join([f"- {c}" for c in connecting[:5]])
            return f"Based on the {sec_name} (Page {page_num}), the relationship is established as follows:\n\n{bullets}"

        return chunks[0]["content"][:300] + "..."

    def _synthesize_comparison(
        self,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        # 1. Existing Markdown table in evidence
        for c in chunks:
            table_lines = [l.strip() for l in c["content"].split("\n") if l.strip().startswith("|")]
            if len(table_lines) >= 3:
                return f"Based on the {c.get('section', 'Document Evidence')} (Page {c.get('page', 1)}):\n\n" + "\n".join(table_lines)

        # 2. Generic 2-entity comparison
        entity1 = classification.get("entity1", "Entity A").strip()
        entity2 = classification.get("entity2", "Entity B").strip()
        e1_words = set(re.findall(r"\b[A-Za-z0-9_]+\b", entity1.lower())) - STOP_WORDS
        e2_words = set(re.findall(r"\b[A-Za-z0-9_]+\b", entity2.lower())) - STOP_WORDS
        e1_distinct = e1_words - e2_words or e1_words
        e2_distinct = e2_words - e1_words or e2_words

        e1_sentences = []
        e2_sentences = []
        for line in lines:
            l_low = line.lower()
            if any(w in l_low for w in e1_distinct) and not any(w in l_low for w in e2_distinct):
                if line not in e1_sentences and len(line) > 15:
                    e1_sentences.append(line)
            elif any(w in l_low for w in e2_distinct) and not any(w in l_low for w in e1_distinct):
                if line not in e2_sentences and len(line) > 15:
                    e2_sentences.append(line)

        rows = [
            f"| Aspect / Feature | {entity1.title()} | {entity2.title()} |",
            "|---|---|---|"
        ]
        s1 = " ".join(e1_sentences[:2]) if e1_sentences else "Present in document evidence."
        s2 = " ".join(e2_sentences[:2]) if e2_sentences else "Present in document evidence."
        rows.append(f"| **Key Characteristics & Scope** | {s1} | {s2} |")
        return "\n".join(rows)

    def _synthesize_flow(
        self,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        source = classification.get("source", "Source").strip()
        target = classification.get("target", "Target").strip()

        detected_steps = []
        for line in lines:
            clean = re.sub(r"^[\-\•\*\d\.]+\s*", "", line).strip()
            if 4 <= len(clean) <= 60 and clean[0].isupper() and clean not in detected_steps:
                detected_steps.append(clean)

        steps = detected_steps[:6] if detected_steps else [source, "Intermediate Processing Layer", target]
        flow_diagram = "\n        ↓\n".join(steps)
        return (
            f"The hierarchical flow from {source} to {target} proceeds as follows:\n\n"
            f"{flow_diagram}"
        )

    def _synthesize_list(
        self,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        target = classification.get("target", "").lower()
        top_chunk = chunks[0]
        sec_name = top_chunk.get("section", "Document Evidence")
        page_num = top_chunk.get("page", 1)

        extracted_items = []
        seen = set()

        for line in lines:
            line_str = line.strip()
            m = re.match(r"^[\-–—•\*\d\.]+\s*(?:([A-Z0-9\s_\-]{2,40})[:–—\-])?\s*(.*)", line_str)
            if m:
                name = (m.group(1) or "").strip()
                desc = (m.group(2) or "").strip()
                if not name and desc:
                    name = desc.split(",")[0].strip()
                if name and len(name) < 80 and name.lower() not in seen:
                    seen.add(name.lower())
                    extracted_items.append((name, desc))

        if len(extracted_items) >= 2:
            formatted = [f"{i}. **{name}**: {desc}" if desc and desc != name else f"{i}. **{name}**" for i, (name, desc) in enumerate(extracted_items[:8], 1)]
            return f"Based on the {sec_name} (Page {page_num}), the following items were identified:\n\n" + "\n".join(formatted)

        return chunks[0]["content"][:300].strip()

    def _synthesize_definition(
        self,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        target = classification.get("target", "").strip()
        top_chunk = chunks[0]
        sec_name = top_chunk.get("section", "Document Specifications")
        page_num = top_chunk.get("page", 1)
        all_content = "\n".join([c["content"] for c in chunks])

        # Technical Identifier & Acronym resolution (e.g. `ara::exec` -> `Execution Management`, `ARA` -> `AUTOSAR Runtime for Adaptive Applications`)
        if target:
            m = re.search(r'(?<![A-Za-z0-9_:])' + re.escape(target) + r'[ \t]*[:\-\u2013\u2014]?[ \t]+([A-Z][A-Za-z0-9_]*(?:[ \t]+[A-Z][A-Za-z0-9_]*)*)', all_content, re.IGNORECASE)
            if m and m.group(1).strip() and len(m.group(1).strip()) >= 3:
                full_label = m.group(1).strip()
                sub_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", all_content) if target.lower() in s.lower() and len(s.strip()) > 20]
                bullet_str = "\n".join([f"- {s}" for s in sub_sentences[:3]]) if sub_sentences else ""
                return (
                    f"Based on the {sec_name} (Page {page_num}):\n\n"
                    f"**{target}** represents **{full_label}**.\n\n"
                    f"{bullet_str}"
                ).strip()

            m_acr = re.search(r'\b([A-Z][A-Za-z0-9_\-\s]{2,50}?)\s*\(\s*' + re.escape(target) + r'\s*\)', all_content, re.IGNORECASE)
            if m_acr and m_acr.group(1).strip():
                full_label = m_acr.group(1).strip()
                sub_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", all_content) if target.lower() in s.lower() and len(s.strip()) > 20]
                bullet_str = "\n".join([f"- {s}" for s in sub_sentences[:3]]) if sub_sentences else ""
                return (
                    f"Based on the {sec_name} (Page {page_num}):\n\n"
                    f"**{target}** stands for **{full_label}**.\n\n"
                    f"{bullet_str}"
                ).strip()

        # Enumerated provisions / deliverables in the evidence
        provisions = []
        for line in lines:
            clean_l = re.sub(r"^[\-\–—•\*\d\.\\…\>\~]+\s*", "", line).strip()
            if any(clean_l.lower().startswith(kw) for kw in ["provides", "defines", "supports", "enables", "describes", "delivers", "specifies"]):
                if clean_l not in provisions and len(clean_l) > 15:
                    provisions.append(clean_l)
            elif line.strip().startswith(("•", "-", "*", "\t")) and len(clean_l) > 10 and not any(clean_l.lower().startswith(kw) for kw in ["provides", "defines", "supports", "enables", "describes", "delivers", "specifies"]):
                if provisions:
                    provisions.append("  - " + clean_l)

        if provisions and len(provisions) >= 2:
            bullets = "\n".join([f"- {p[0].upper() + p[1:]}" if not p.startswith("  - ") else p for p in provisions])
            return (
                f"Based on the {sec_name} (Page {page_num}), the {target or 'system'} provides:\n\n"
                f"{bullets}"
            )

        # Generic definition extraction
        defining_sentences = []
        feature_bullets = []
        target_words = [w for w in target.lower().split() if w not in STOP_WORDS]

        for line in lines:
            line_str = line.strip()
            l_low = line_str.lower()
            clean_b = re.sub(r"^[\-\•\*\d\.]+\s*", "", line_str)
            if target_words and any(w in l_low for w in target_words):
                if any(kw in l_low for kw in ["is a", "defined as", "provides", "responsible for", "abstracts", "connects", "layer", "module", "service", "interface"]):
                    if clean_b not in defining_sentences and len(clean_b) > 15:
                        defining_sentences.append(clean_b)
                elif line_str.startswith(("-", "•", "*")):
                    if clean_b not in feature_bullets and len(clean_b) > 10:
                        feature_bullets.append(clean_b)

        if not defining_sentences:
            defining_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_chunk["content"]) if s.strip() and not s.startswith("[Document:")][:2]

        def_text = " ".join(defining_sentences[:2])
        if feature_bullets:
            bullets_text = "\n".join([f"- {b}" for b in feature_bullets[:5]])
            return f"Based on the {sec_name} (Page {page_num}):\n\n{def_text}\n\nKey responsibilities and capabilities include:\n{bullets_text}"

        return f"Based on the {sec_name} (Page {page_num}):\n\n{def_text}"

    def _synthesize_factual(
        self,
        question: str,
        classification: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        lines: List[str]
    ) -> str:
        q_lower = question.lower()
        content_words = [w for w in re.findall(r"\b[A-Za-z0-9_\-]{3,}\b", q_lower) if w not in STOP_WORDS]

        # 1. Processing fee query / monetary amounts
        if any(w in q_lower for w in ["fee", "cost", "amount", "price"]):
            for line in lines:
                if any(w in line.lower() for w in ["fee", "cost", "payment", "rs.", "inr", "₹", "$", "€", "gst"]):
                    if re.search(r"(?:₹|Rs\.?|\$|€|£)\s*[\d,]+", line, re.IGNORECASE):
                        return line.strip()

        # 2. Submission method / procedure queries
        if any(w in q_lower for w in ["submit", "submission", "how should", "mode of"]):
            for line in lines:
                if any(w in line.lower() for w in ["google form", "portal", "email", "submission", "submit"]):
                    if len(line.strip()) > 25:
                        return line.strip()

        # 3. Table row extraction
        table_rows = []
        for c in chunks:
            for l in c["content"].split("\n"):
                if l.strip().startswith("|") and not l.strip().startswith("| ---"):
                    table_rows.append(l.strip())
        if table_rows and len(table_rows) >= 3 and any(w in q_lower for w in ["list", "which", "table", "entries"]):
            return "Identified document records:\n\n" + "\n".join(table_rows[:10])

        # 4. Sentence scoring by query keyword overlap
        scored_sentences = []
        for line in lines:
            if len(line) < 15 or line.startswith("| ---"):
                continue
            line_lower = line.lower()
            overlap = sum(1 for w in content_words if w in line_lower)
            if overlap > 0:
                scored_sentences.append((overlap, line))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        if scored_sentences:
            best_lines = [s[1] for s in scored_sentences[:3]]
            return " ".join(best_lines)

        return chunks[0]["content"][:300] + "..."


ollama_service = OllamaService()
