import os
import pytest
from pathlib import Path
from sample_data.generate_synthetic_hld import generate_synthetic_autosar_hld
from app.services.pdf_parser import PDFParser
from app.services.chunker import SectionAwareChunker


@pytest.fixture(scope="module")
def sample_pdf_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("pdf_test")
    pdf_file = tmp_dir / "test_autosar_sample.pdf"
    generate_synthetic_autosar_hld(str(pdf_file))
    return str(pdf_file)


def test_pdf_extraction(sample_pdf_path):
    """Test that PDF text, pages, and tables are accurately extracted."""
    parser = PDFParser(sample_pdf_path)
    result = parser.parse()

    assert result["page_count"] >= 3
    assert len(result["pages"]) == result["page_count"]
    
    # Check that text content is present
    first_page = result["pages"][0]
    assert "Vehicle Domain" in first_page["text"] or "SYNTHETIC AUTOSAR" in first_page["text"]

    # Check table extraction
    has_table = any(len(p.get("tables", [])) > 0 for p in result["pages"])
    assert has_table, "At least one table must be extracted from the document"


def test_section_detection(sample_pdf_path):
    """Test that section headings are detected accurately."""
    parser = PDFParser(sample_pdf_path)
    result = parser.parse()

    detected_sections = result["detected_sections"]
    assert len(detected_sections) > 0

    # Ensure key AUTOSAR sections are recognized
    joined = " ".join(detected_sections)
    assert "Software Component Architecture" in joined or "Interface Specifications" in joined or "System Overview" in joined


def test_chunking(sample_pdf_path):
    """Test that section-aware chunking preserves metadata and hierarchical context."""
    parser = PDFParser(sample_pdf_path)
    result = parser.parse()

    chunker = SectionAwareChunker(target_chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_document(
        document_id=1,
        document_name="test_autosar_sample.pdf",
        parsed_pages=result["pages"]
    )

    assert len(chunks) > 0

    # Verify metadata on every chunk
    for chunk in chunks:
        assert "document_id" in chunk["metadata"]
        assert chunk["metadata"]["document_id"] == "1"
        assert "document_name" in chunk["metadata"]
        assert "page" in chunk["metadata"]
        assert "section" in chunk["metadata"]
        assert "source_type" in chunk["metadata"]
        assert chunk["metadata"]["source_type"] in ["text", "table"]
        assert chunk["text"].startswith("[Document:")
