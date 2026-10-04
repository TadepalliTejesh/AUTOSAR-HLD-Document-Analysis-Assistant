import io
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session
from app.db.repositories import EntityRepository, DocumentRepository, AuditRepository, InconsistencyRepository


class ExportService:
    @staticmethod
    def export_inventory_csv(db: Session, document_id: int) -> str:
        """Export all architecture inventory entities into a structured CSV string."""
        entities = EntityRepository.get_by_document(db, document_id)
        rows = []
        for e in entities:
            details_dict = {}
            try:
                details_dict = json.loads(e.details or "{}")
            except Exception:
                pass

            rows.append({
                "Entity Type": e.entity_type,
                "Name": e.name,
                "Page": e.page,
                "Section": e.section,
                "Details": json.dumps(details_dict),
                "Evidence": e.evidence
            })

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    @staticmethod
    def export_inventory_xlsx(db: Session, document_id: int) -> bytes:
        """Export multi-sheet Excel file with separate sheets for each entity category."""
        doc = DocumentRepository.get_by_id(db, document_id)
        doc_name = doc.filename if doc else f"Document_{document_id}"
        entities = EntityRepository.get_by_document(db, document_id)
        inconsistencies = InconsistencyRepository.get_by_document(db, document_id)
        audit_events = AuditRepository.list_events(db, document_id)

        # Separate entity lists
        components, interfaces, ports, signals, dependencies = [], [], [], [], []

        for e in entities:
            details = {}
            try:
                details = json.loads(e.details or "{}")
            except Exception:
                pass

            if e.entity_type == "COMPONENT":
                components.append({
                    "Component": e.name,
                    "Type": details.get("component_type", "SWC"),
                    "Page": e.page,
                    "Section": e.section,
                    "Evidence": e.evidence
                })
            elif e.entity_type == "INTERFACE":
                interfaces.append({
                    "Interface": e.name,
                    "Interface Type": details.get("interface_type", "Sender-Receiver"),
                    "Page": e.page,
                    "Section": e.section,
                    "Evidence": e.evidence
                })
            elif e.entity_type == "PORT":
                ports.append({
                    "Port": e.name,
                    "Port Type": details.get("port_type", "P-Port"),
                    "Component": details.get("component", ""),
                    "Interface": details.get("interface", ""),
                    "Page": e.page,
                    "Section": e.section,
                    "Evidence": e.evidence
                })
            elif e.entity_type == "SIGNAL":
                signals.append({
                    "Signal": e.name,
                    "Data Type": details.get("data_type", "uint16"),
                    "Interface": details.get("interface", ""),
                    "Page": e.page,
                    "Section": e.section,
                    "Evidence": e.evidence
                })
            elif e.entity_type == "DEPENDENCY":
                dependencies.append({
                    "Source Component": details.get("source", ""),
                    "Target Component": details.get("target", ""),
                    "Dependency Type": details.get("dependency_type", ""),
                    "Page": e.page,
                    "Section": e.section,
                    "Evidence": e.evidence
                })

        # Potential inconsistencies
        inc_rows = []
        for inc in inconsistencies:
            inc_rows.append({
                "Entity": inc.entity_name,
                "Issue Type": inc.issue_type,
                "Evidence 1": inc.evidence_1,
                "Evidence 2": inc.evidence_2,
                "Reason": inc.reason,
                "Status": inc.status
            })

        # Audit events
        audit_rows = []
        for aud in audit_events:
            audit_rows.append({
                "Event": aud.event_type,
                "Timestamp": aud.timestamp.isoformat() if aud.timestamp else "",
                "Details": aud.details
            })

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(components if components else [{"Note": "No components detected"}]).to_excel(writer, sheet_name="Components", index=False)
            pd.DataFrame(interfaces if interfaces else [{"Note": "No interfaces detected"}]).to_excel(writer, sheet_name="Interfaces", index=False)
            pd.DataFrame(ports if ports else [{"Note": "No ports detected"}]).to_excel(writer, sheet_name="Ports", index=False)
            pd.DataFrame(signals if signals else [{"Note": "No signals detected"}]).to_excel(writer, sheet_name="Signals", index=False)
            pd.DataFrame(dependencies if dependencies else [{"Note": "No dependencies detected"}]).to_excel(writer, sheet_name="Dependencies", index=False)
            pd.DataFrame(inc_rows if inc_rows else [{"Note": "No potential inconsistencies detected"}]).to_excel(writer, sheet_name="Inconsistencies", index=False)
            pd.DataFrame(audit_rows if audit_rows else [{"Note": "No audit records"}]).to_excel(writer, sheet_name="Audit Log", index=False)

        return output.getvalue()


export_service = ExportService()
