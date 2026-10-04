import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import (
    DocumentRepository,
    EntityRepository,
    InconsistencyRepository,
    AuditRepository
)
from app.schemas.inventory import (
    ArchitectureInventoryResponse,
    ComponentItem,
    InterfaceItem,
    PortItem,
    SignalItem,
    DependencyItem,
    PotentialInconsistencyItem
)
from app.services.exporter import export_service

router = APIRouter(tags=["Inventory & Analysis"])


@router.get("/documents/{document_id}/inventory", response_model=ArchitectureInventoryResponse)
def get_document_inventory(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Retrieve full structured architecture inventory for an indexed HLD document."""
    doc = DocumentRepository.get_by_id(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found."
        )

    entities = EntityRepository.get_by_document(db, document_id)
    inconsistencies = InconsistencyRepository.get_by_document(db, document_id)

    components: List[ComponentItem] = []
    interfaces: List[InterfaceItem] = []
    ports: List[PortItem] = []
    signals: List[SignalItem] = []
    dependencies: List[DependencyItem] = []

    for ent in entities:
        details = {}
        try:
            details = json.loads(ent.details or "{}")
        except Exception:
            pass

        if ent.entity_type == "COMPONENT":
            components.append(ComponentItem(
                name=ent.name,
                component_type=details.get("component_type", "Application SWC"),
                page=ent.page,
                section=ent.section,
                evidence=ent.evidence
            ))
        elif ent.entity_type == "INTERFACE":
            interfaces.append(InterfaceItem(
                name=ent.name,
                interface_type=details.get("interface_type", "Sender-Receiver"),
                page=ent.page,
                section=ent.section,
                evidence=ent.evidence
            ))
        elif ent.entity_type == "PORT":
            ports.append(PortItem(
                name=ent.name,
                port_type=details.get("port_type", "P-Port"),
                component=details.get("component", "Application SWC"),
                interface=details.get("interface", "Interface"),
                page=ent.page,
                section=ent.section,
                evidence=ent.evidence
            ))
        elif ent.entity_type == "SIGNAL":
            signals.append(SignalItem(
                name=ent.name,
                data_type=details.get("data_type", "uint16"),
                interface=details.get("interface", "Sender-Receiver"),
                page=ent.page,
                section=ent.section,
                evidence=ent.evidence
            ))
        elif ent.entity_type == "DEPENDENCY":
            dependencies.append(DependencyItem(
                source=details.get("source", "SourceSWC"),
                target=details.get("target", "TargetSWC"),
                dependency_type=details.get("dependency_type", "Direct Connection"),
                page=ent.page,
                section=ent.section,
                evidence=ent.evidence
            ))

    inc_items = [
        PotentialInconsistencyItem(
            entity_name=inc.entity_name,
            issue_type=inc.issue_type,
            evidence_1=inc.evidence_1,
            evidence_2=inc.evidence_2,
            reason=inc.reason,
            status=inc.status
        )
        for inc in inconsistencies
    ]

    summary = {
        "components": len(components),
        "interfaces": len(interfaces),
        "ports": len(ports),
        "signals": len(signals),
        "dependencies": len(dependencies),
        "potential_inconsistencies": len(inc_items)
    }

    return ArchitectureInventoryResponse(
        document_id=doc.id,
        document_name=doc.filename,
        summary=summary,
        components=components,
        interfaces=interfaces,
        ports=ports,
        signals=signals,
        dependencies=dependencies,
        inconsistencies=inc_items
    )


@router.get("/documents/{document_id}/export/csv")
def export_csv(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Export architecture inventory to CSV format."""
    doc = DocumentRepository.get_by_id(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    csv_data = export_service.export_inventory_csv(db, document_id)
    filename = f"{doc.filename.replace('.pdf', '')}_inventory.csv"

    AuditRepository.log(
        db=db,
        event_type="INVENTORY_EXPORTED",
        document_id=document_id,
        details={"format": "csv", "filename": filename}
    )

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/documents/{document_id}/export/xlsx")
def export_xlsx(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Export architecture inventory to multi-sheet Excel format."""
    doc = DocumentRepository.get_by_id(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    xlsx_bytes = export_service.export_inventory_xlsx(db, document_id)
    filename = f"{doc.filename.replace('.pdf', '')}_architecture_inventory.xlsx"

    AuditRepository.log(
        db=db,
        event_type="INVENTORY_EXPORTED",
        document_id=document_id,
        details={"format": "xlsx", "filename": filename}
    )

    return Response(
        content=xlsx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/documents/{document_id}/inconsistencies", response_model=List[PotentialInconsistencyItem])
def get_document_inconsistencies(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Retrieve potential architectural inconsistencies detected for the document."""
    inconsistencies = InconsistencyRepository.get_by_document(db, document_id)
    return [
        PotentialInconsistencyItem(
            entity_name=inc.entity_name,
            issue_type=inc.issue_type,
            evidence_1=inc.evidence_1,
            evidence_2=inc.evidence_2,
            reason=inc.reason,
            status=inc.status
        )
        for inc in inconsistencies
    ]
