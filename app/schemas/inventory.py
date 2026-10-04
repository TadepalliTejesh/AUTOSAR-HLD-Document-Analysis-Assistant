from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ComponentItem(BaseModel):
    name: str
    component_type: Optional[str] = "Application SWC"
    page: int
    section: str
    evidence: str


class InterfaceItem(BaseModel):
    name: str
    interface_type: str  # Sender-Receiver, Client-Server, Service
    page: int
    section: str
    evidence: str


class PortItem(BaseModel):
    name: str
    port_type: str  # P-Port, R-Port, PR-Port
    component: str
    interface: str
    page: int
    section: str
    evidence: str


class SignalItem(BaseModel):
    name: str
    data_type: str
    interface: str
    page: int
    section: str
    evidence: str


class DependencyItem(BaseModel):
    source: str
    target: str
    dependency_type: str
    page: int
    section: str
    evidence: str


class PotentialInconsistencyItem(BaseModel):
    entity_name: str
    issue_type: str
    evidence_1: str
    evidence_2: str
    reason: str
    status: str = "Requires Engineer Review"


class ArchitectureInventoryResponse(BaseModel):
    document_id: int
    document_name: str
    summary: Dict[str, int]
    components: List[ComponentItem]
    interfaces: List[InterfaceItem]
    ports: List[PortItem]
    signals: List[SignalItem]
    dependencies: List[DependencyItem]
    inconsistencies: List[PotentialInconsistencyItem]
