import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import init_db
from sample_data.generate_synthetic_hld import generate_synthetic_autosar_hld


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_db()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def uploaded_doc_id(client, tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("api_test")
    pdf_path = tmp_dir / "api_test_sample.pdf"
    generate_synthetic_autosar_hld(str(pdf_path))

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/upload-hld",
        files={"file": ("api_test_sample.pdf", file_bytes, "application/pdf")},
        data={"project_name": "API Test Project"}
    )
    assert response.status_code == 200
    data = response.json()
    return data["document_id"]


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "chromadb" in data


def test_upload(client, uploaded_doc_id):
    assert uploaded_doc_id > 0
    res = client.get(f"/documents/{uploaded_doc_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == uploaded_doc_id
    assert data["status"] == "INDEXED"
    assert data["page_count"] >= 3


def test_query(client, uploaded_doc_id):
    payload = {
        "document_id": uploaded_doc_id,
        "question": "Which software components communicate with the CAN interface?"
    }
    res = client.post("/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["document_id"] == uploaded_doc_id
    assert len(data["answer"]) > 10
    assert data["confidence"] in ["High", "Medium", "Low", "Insufficient"]
    assert "citations" in data
    assert "retrieved_evidence" in data


def test_inventory(client, uploaded_doc_id):
    res = client.get(f"/documents/{uploaded_doc_id}/inventory")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert data["summary"]["components"] > 0
    assert len(data["components"]) > 0
    assert len(data["interfaces"]) > 0
    assert len(data["ports"]) > 0
    assert len(data["signals"]) > 0
    assert len(data["dependencies"]) > 0

    # Test CSV export
    res_csv = client.get(f"/documents/{uploaded_doc_id}/export/csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert "Entity Type" in res_csv.text

    # Test Excel export
    res_xlsx = client.get(f"/documents/{uploaded_doc_id}/export/xlsx")
    assert res_xlsx.status_code == 200
    assert len(res_xlsx.content) > 100
