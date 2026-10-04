from typing import List, Dict, Any
from app.core.logging import logger


class InconsistencyDetector:
    """
    Detects potential architectural inconsistencies across different sections and pages of the HLD.
    Outputs findings as 'Potential Inconsistency' with dual-source citations.
    """

    def analyze(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        inconsistencies = []

        # 1. Group ports by name to check for conflicting directions or interface assignments
        ports_by_name: Dict[str, List[Dict[str, Any]]] = {}
        for ent in entities:
            if ent["entity_type"] == "PORT":
                name = ent["name"]
                if name not in ports_by_name:
                    ports_by_name[name] = []
                ports_by_name[name].append(ent)

        for port_name, occurrences in ports_by_name.items():
            if len(occurrences) > 1:
                # Check for conflicting directions
                types = {occ["details"].get("port_type") for occ in occurrences if occ["details"].get("port_type")}
                if len(types) > 1:
                    first = occurrences[0]
                    second = occurrences[1]
                    inconsistencies.append({
                        "entity_name": port_name,
                        "issue_type": "PORT_DIRECTION_CONFLICT",
                        "evidence_1": f"Page {first['page']} | Section: {first['section']} (Defined as {first['details'].get('port_type')})",
                        "evidence_2": f"Page {second['page']} | Section: {second['section']} (Defined as {second['details'].get('port_type')})",
                        "reason": f"Port '{port_name}' is declared with conflicting port types ({', '.join(types)}) across sections.",
                        "status": "Requires Engineer Review"
                    })

                # Check for conflicting interface assignments
                ifaces = {occ["details"].get("interface") for occ in occurrences if occ["details"].get("interface") and occ["details"].get("interface") != "Mapped Interface"}
                if len(ifaces) > 1:
                    first = occurrences[0]
                    second = occurrences[1]
                    inconsistencies.append({
                        "entity_name": port_name,
                        "issue_type": "INTERFACE_MISMATCH",
                        "evidence_1": f"Page {first['page']} | Section: {first['section']} (Bound to {first['details'].get('interface')})",
                        "evidence_2": f"Page {second['page']} | Section: {second['section']} (Bound to {second['details'].get('interface')})",
                        "reason": f"Port '{port_name}' is associated with different interfaces ({' vs '.join(ifaces)}) across sections.",
                        "status": "Requires Engineer Review"
                    })

        # 2. Check for components associated with conflicting interfaces across sections
        # Group dependencies by source component
        component_ifaces: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for ent in entities:
            if ent["entity_type"] == "DEPENDENCY":
                src = ent["details"].get("source")
                dep_type = ent["details"].get("dependency_type")
                if src and dep_type and dep_type not in {"Functional Flow", "Direct Connection"}:
                    if src not in component_ifaces:
                        component_ifaces[src] = {}
                    component_ifaces[src][dep_type] = ent

        # 3. Check for dependencies referencing undeclared target components
        known_components = {e["name"] for e in entities if e["entity_type"] == "COMPONENT"}
        for ent in entities:
            if ent["entity_type"] == "DEPENDENCY":
                tgt = ent["details"].get("target")
                if tgt and known_components and tgt not in known_components:
                    # Target component not listed in known component definitions
                    inconsistencies.append({
                        "entity_name": tgt,
                        "issue_type": "UNDEFINED_TARGET_COMPONENT",
                        "evidence_1": f"Page {ent['page']} | Section: {ent['section']} (Target of {ent['details'].get('source')} dependency)",
                        "evidence_2": "Architecture Inventory (Missing in Component definitions)",
                        "reason": f"Target component '{tgt}' is referenced as a dependency target but has no formal component specification in the document.",
                        "status": "Requires Engineer Review"
                    })

        # 4. Check for conflicting signal data types
        signals_by_name: Dict[str, List[Dict[str, Any]]] = {}
        for ent in entities:
            if ent["entity_type"] == "SIGNAL":
                name = ent["name"]
                if name not in signals_by_name:
                    signals_by_name[name] = []
                signals_by_name[name].append(ent)

        for sig_name, occurrences in signals_by_name.items():
            if len(occurrences) > 1:
                dtypes = {occ["details"].get("data_type") for occ in occurrences if occ["details"].get("data_type") and occ["details"].get("data_type") != "Standard AUTOSAR Type"}
                if len(dtypes) > 1:
                    first = occurrences[0]
                    second = occurrences[1]
                    inconsistencies.append({
                        "entity_name": sig_name,
                        "issue_type": "SIGNAL_DATATYPE_MISMATCH",
                        "evidence_1": f"Page {first['page']} | Section: {first['section']} (Type: {first['details'].get('data_type')})",
                        "evidence_2": f"Page {second['page']} | Section: {second['section']} (Type: {second['details'].get('data_type')})",
                        "reason": f"Signal '{sig_name}' is declared with disparate data types ({' vs '.join(dtypes)}) in separate sections.",
                        "status": "Requires Engineer Review"
                    })

        logger.info(f"Detected {len(inconsistencies)} potential architecture inconsistencies")
        return inconsistencies


inconsistency_detector = InconsistencyDetector()
