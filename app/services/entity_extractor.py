import re
import json
from typing import List, Dict, Any, Tuple
from app.core.logging import logger


class EntityExtractor:
    """
    Extracts AUTOSAR architecture entities (Components, Interfaces, Ports, Signals, Dependencies)
    from parsed HLD document pages and tables with strict source grounding.
    """

    # AUTOSAR naming patterns
    COMPONENT_PATTERN = re.compile(
        r"\b(?:SWC_|Swc|Comp_|Component_)?([A-Z][A-Za-z0-9_]*(?:Control|Manager|Actuator|Sensor|Handler|Diagnostic|Monitor|Gateway|Coordinator|Controller|SWC|Swc))\b"
    )
    INTERFACE_PATTERN = re.compile(
        r"\b(?:If_|IF_|Interface_)?([A-Z][A-Za-z0-9_]*(?:If|Interface|DataIf|CmdIf|StatusIf|ServiceIf|ReqIf))\b"
    )
    PORT_PATTERN = re.compile(
        r"\b((?:[PpRr]|Pp|Pr)Port_[A-Za-z0-9_]+|(?:PPort|RPort|PRPort)_[A-Za-z0-9_]+|[A-Z][A-Za-z0-9_]*_Port)\b"
    )
    SIGNAL_PATTERN = re.compile(
        r"\b(?:Sig_|sig_|S_|s_)?([A-Z][A-Za-z0-9_]*(?:Speed|Torque|Angle|Temp|Pressure|Voltage|Current|Status|Cmd|Request|State|Flag|Counter|Checksum))\b"
    )

    def extract_from_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract all entities from pages and tables, returning unified architecture entity dicts.
        """
        entities = []
        seen_keys = set()

        for page in pages:
            page_num = page["page"]
            section = page.get("current_section", "General Architecture")

            # 1. Extract from tables first (highest fidelity structured data)
            for tbl in page.get("tables", []):
                tbl_entities = self._extract_from_table(tbl["content"], page_num, section)
                for ent in tbl_entities:
                    key = (ent["entity_type"], ent["name"], ent["page"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        entities.append(ent)

            # 2. Extract from text blocks
            for blk in page.get("blocks", []):
                blk_sec = blk.get("section", section)
                blk_text = blk.get("text", "")
                blk_entities = self._extract_from_text(blk_text, page_num, blk_sec)
                for ent in blk_entities:
                    key = (ent["entity_type"], ent["name"], ent["page"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        entities.append(ent)

        logger.info(f"Extracted {len(entities)} architecture entities across {len(pages)} pages")
        return entities

    def _extract_from_table(self, table_md: str, page_num: int, section: str) -> List[Dict[str, Any]]:
        """Extract structured entities from markdown tables."""
        entities = []
        lines = [l.strip() for l in table_md.split("\n") if l.strip().startswith("|")]
        if len(lines) < 3:
            return entities

        header_cells = [c.strip().lower() for c in lines[0].split("|")[1:-1]]
        data_rows = lines[2:]

        # Detect table domain
        is_comp_table = any("component" in h or "swc" in h for h in header_cells)
        is_if_table = any("interface" in h for h in header_cells)
        is_port_table = any("port" in h for h in header_cells)
        is_sig_table = any("signal" in h or "data element" in h for h in header_cells)
        is_dep_table = any("dependency" in h or "target" in h or "source" in h or "connection" in h for h in header_cells)

        def _clean_id(val: str) -> str:
            s = str(val or "").strip()
            s = re.sub(r"(\w+)\s*_\s*(\w+)", r"\1_\2", s)
            s = re.sub(r"(\w+)\s+(\w+)\s+_", r"\1_\2", s)
            if any(prefix in s for prefix in ["PPort", "RPort", "Sig", "SWC", "If"]):
                s = s.replace(" ", "_")
                s = re.sub(r"_+", "_", s).strip("_")
            return s.strip()

        for row_str in data_rows:
            cells = [_clean_id(c.strip()) for c in row_str.split("|")[1:-1]]
            if len(cells) < len(header_cells):
                continue
            row_dict = {header_cells[i]: cells[i] for i in range(len(header_cells))}

            # Table row entity mapping
            if is_dep_table:
                source = _clean_id(row_dict.get("source") or row_dict.get("source component") or (cells[0] if len(cells) > 0 else ""))
                target = _clean_id(row_dict.get("target") or row_dict.get("target component") or (cells[1] if len(cells) > 1 else ""))
                iface = _clean_id(row_dict.get("interface") or row_dict.get("connection") or "Explicit Link")
                if source and target and source != target:
                    entities.append({
                        "entity_type": "DEPENDENCY",
                        "name": f"{source} -> {target}",
                        "details": {
                            "source": source,
                            "target": target,
                            "dependency_type": iface
                        },
                        "page": page_num,
                        "section": section,
                        "evidence": f"Table mapping: {source} communicates with {target} via {iface}"
                    })

            if is_port_table:
                port_name = _clean_id(row_dict.get("port") or row_dict.get("port name") or cells[0])
                comp = _clean_id(row_dict.get("component") or row_dict.get("swc") or "AppSWC")
                p_type = row_dict.get("direction") or row_dict.get("port type") or ("P-Port" if "P" in port_name else "R-Port")
                iface = _clean_id(row_dict.get("interface") or "Interface")
                if port_name and len(port_name) > 2:
                    entities.append({
                        "entity_type": "PORT",
                        "name": port_name,
                        "details": {
                            "port_type": p_type,
                            "component": comp,
                            "interface": iface
                        },
                        "page": page_num,
                        "section": section,
                        "evidence": f"Port table entry: {port_name} ({p_type}) mapped to {comp} using {iface}"
                    })

            if is_sig_table:
                sig_name = _clean_id(row_dict.get("signal") or row_dict.get("signal name") or row_dict.get("data element") or cells[0])
                dtype = _clean_id(row_dict.get("type") or row_dict.get("data type") or "uint16")
                iface = _clean_id(row_dict.get("interface") or "Sender-Receiver")
                if sig_name and len(sig_name) > 2:
                    entities.append({
                        "entity_type": "SIGNAL",
                        "name": sig_name,
                        "details": {
                            "data_type": dtype,
                            "interface": iface
                        },
                        "page": page_num,
                        "section": section,
                        "evidence": f"Signal definition: {sig_name} [{dtype}] in {iface}"
                    })

            if is_if_table and not is_port_table and not is_dep_table:
                if_name = _clean_id(row_dict.get("interface") or row_dict.get("interface name") or cells[0])
                if_type = row_dict.get("type") or ("Client-Server" if "CS" in if_name or "Server" in if_name else "Sender-Receiver")
                if if_name and len(if_name) > 2:
                    entities.append({
                        "entity_type": "INTERFACE",
                        "name": if_name,
                        "details": {
                            "interface_type": if_type
                        },
                        "page": page_num,
                        "section": section,
                        "evidence": f"Interface catalog: {if_name} ({if_type})"
                    })

            if is_comp_table and not is_dep_table:
                comp_name = _clean_id(row_dict.get("component") or row_dict.get("swc name") or cells[0])
                comp_type = row_dict.get("type") or "Application SWC"
                if comp_name and len(comp_name) > 2:
                    entities.append({
                        "entity_type": "COMPONENT",
                        "name": comp_name,
                        "details": {
                            "component_type": comp_type
                        },
                        "page": page_num,
                        "section": section,
                        "evidence": f"Component inventory: {comp_name} ({comp_type})"
                    })

        return entities

    def _extract_from_text(self, text: str, page_num: int, section: str) -> List[Dict[str, Any]]:
        """Extract explicit entities from text paragraphs using automotive patterns."""
        entities = []

        # 1. Explicit Component mentions
        for m in self.COMPONENT_PATTERN.finditer(text):
            comp_name = m.group(1)
            # Filter common English words
            if comp_name not in {"Vehicle", "System", "Control", "Manager", "Interface", "Software"}:
                entities.append({
                    "entity_type": "COMPONENT",
                    "name": comp_name,
                    "details": {"component_type": "AUTOSAR SWC"},
                    "page": page_num,
                    "section": section,
                    "evidence": text[:200]
                })

        # 2. Explicit Interfaces
        for m in self.INTERFACE_PATTERN.finditer(text):
            if_name = m.group(1)
            if_type = "Client-Server" if "Client" in text or "Server" in text or "CS" in if_name else "Sender-Receiver"
            entities.append({
                "entity_type": "INTERFACE",
                "name": if_name,
                "details": {"interface_type": if_type},
                "page": page_num,
                "section": section,
                "evidence": text[:200]
            })

        # 3. Explicit Ports
        for m in self.PORT_PATTERN.finditer(text):
            port_name = m.group(1)
            p_type = "P-Port" if port_name.startswith(("PPort", "P_", "p_")) else "R-Port"
            entities.append({
                "entity_type": "PORT",
                "name": port_name,
                "details": {
                    "port_type": p_type,
                    "component": "Referenced Component",
                    "interface": "Mapped Interface"
                },
                "page": page_num,
                "section": section,
                "evidence": text[:200]
            })

        # 4. Explicit Signals
        for m in self.SIGNAL_PATTERN.finditer(text):
            sig_name = m.group(1)
            if len(sig_name) > 3 and sig_name not in {"System", "Status", "Control"}:
                entities.append({
                    "entity_type": "SIGNAL",
                    "name": sig_name,
                    "details": {"data_type": "Standard AUTOSAR Type", "interface": "Communication Interface"},
                    "page": page_num,
                    "section": section,
                    "evidence": text[:200]
                })

        # 5. Explicit Dependencies: "Component A communicates with / connects to Component B" or "A -> B"
        dep_matches = re.finditer(
            r"([A-Z][A-Za-z0-9_]+)\s+(?:transmits to|sends data to|communicates with|connects to|interacts with|invokes|calls)\s+([A-Z][A-Za-z0-9_]+)",
            text
        )
        for d in dep_matches:
            src = d.group(1)
            tgt = d.group(2)
            if src != tgt and len(src) > 3 and len(tgt) > 3:
                entities.append({
                    "entity_type": "DEPENDENCY",
                    "name": f"{src} -> {tgt}",
                    "details": {
                        "source": src,
                        "target": tgt,
                        "dependency_type": "Functional Flow"
                    },
                    "page": page_num,
                    "section": section,
                    "evidence": d.group(0)
                })

        arrow_deps = re.finditer(r"([A-Z][A-Za-z0-9_]+)\s*(?:->|→)\s*([A-Z][A-Za-z0-9_]+)", text)
        for d in arrow_deps:
            src = d.group(1)
            tgt = d.group(2)
            if src != tgt:
                entities.append({
                    "entity_type": "DEPENDENCY",
                    "name": f"{src} -> {tgt}",
                    "details": {
                        "source": src,
                        "target": tgt,
                        "dependency_type": "Direct Connection"
                    },
                    "page": page_num,
                    "section": section,
                    "evidence": f"Explicit mapping: {src} -> {tgt}"
                })

        return entities


entity_extractor = EntityExtractor()
