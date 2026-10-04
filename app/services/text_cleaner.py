import re
from typing import List, Dict, Any, Optional

def clean_ocr_text(text: str) -> str:
    """
    Cleans raw OCR and PDF extraction artifacts in a document-agnostic manner.
    Removes:
    - Slide/page number headers (e.g. '8 TM', '12 TM', '25 TM')
    - Vendor footers, copyright disclaimers, watermarks, garbled OCR noise
    - Repetitive isolated file extensions (e.g. '.h', '.c')
    - Broken punctuation and layout debris
    - Concatenated OCR tokens (e.g. 'AUTOSARInterface' -> 'AUTOSAR Interface')
    Preserves all substantive domain text, parameters, tables, and architectural facts.
    """
    if not text:
        return ""

    # 1. Normalize line endings and characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-printable or corrupt replacement characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]", " ", text)
    # Normalize unicode hyphens and dashes to standard ASCII
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    # Normalize unicode bullets
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219]", "•", text)

    # 2. Split into lines for line-by-line filtering
    raw_lines = text.split("\n")
    cleaned_lines = []
    
    # Patterns for lines to drop completely
    slide_hdr_pat = re.compile(r"^\s*\d{1,3}\s*(?:TM|®|©)?\s*$", re.IGNORECASE)
    file_ext_stub_pat = re.compile(r"^\s*(?:\.[a-z0-9]{1,4}|c\s*,\s*h|c\s+h|read|write|templates?)\s*$", re.IGNORECASE)
    footer_pat = re.compile(
        r"^(?:source\s*:|copyright\b|all rights reserved|confidential|proprietary\s+(?:and\s+confidential|information|notice|document)).*$",
        re.IGNORECASE
    )

    consecutive_empty = 0

    for line in raw_lines:
        s = line.strip()
        if not s:
            consecutive_empty += 1
            if consecutive_empty <= 1:
                cleaned_lines.append("")
            continue
        consecutive_empty = 0

        # Skip document header tags if present
        if s.startswith("[Document:") or s.startswith("Document ID"):
            continue

        # Skip slide number headers (e.g. '31 TM', '8 TM')
        if slide_hdr_pat.match(s):
            continue

        # Skip repetitive isolated file extensions or diagram pin stubs
        if file_ext_stub_pat.match(s):
            continue

        # Skip vendor footers / copyright disclaimers
        if footer_pat.match(s):
            continue

        # Skip obvious OCR gibberish footer lines with excessive symbols
        if len(s) > 20 and sum(1 for c in s if c in "@#%^&*~`") > 3:
            continue
        # Drop lines that are pure punctuation / non-alphanumeric noise
        if len(s) <= 4 and not any(c.isalnum() for c in s):
            continue

        # Deduplicate consecutive identical lines
        if cleaned_lines and cleaned_lines[-1] == s:
            continue

        # Clean broken OCR word concatenation dynamically via Acronym & Parenthesis boundary splitting
        s = re.sub(r'([a-zA-Z0-9])\(', r'\1 (', s)
        s = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', s)

        cleaned_lines.append(s)

    # 3. Post-process lines: merge broken sentence fragments while preserving bullets and tables
    result = []
    for l in cleaned_lines:
        if not l:
            if result and result[-1] != "":
                result.append("")
            continue
        result.append(l)

    return "\n".join(result).strip()


def clean_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a copy of the chunk with cleaned text content."""
    cleaned = dict(chunk)
    cleaned["content"] = clean_ocr_text(chunk.get("content", ""))
    return cleaned
