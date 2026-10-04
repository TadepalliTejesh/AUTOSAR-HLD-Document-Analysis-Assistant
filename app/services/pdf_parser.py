import re
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from app.core.logging import logger


class PDFParserError(Exception):
    pass


try:
    from rapidocr_onnxruntime import RapidOCR
    _ocr_engine = RapidOCR()
except Exception:
    _ocr_engine = None


class PDFParser:
    """
    Extracts text, headings, structural context, and tables from PDFs.
    Preserves exact section provenance and tabular data.
    """

    HEADING_REGEX = re.compile(
        r"^(?:\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9_\-\s]{2,80}:?|(?:SECTION|CHAPTER|PART|APPENDIX)\s+\d+[:\s].*|[A-Z][A-Z0-9_\s]{3,50}:?)$"
    )

    def __init__(self, file_path: str):
        self.file_path = file_path

    def _extract_slide_title(self, page: fitz.Page, total_pages: int) -> Optional[str]:
        """
        Extracts slide title by inspecting text blocks near the top banner,
        falling back to header OCR if text is not present.
        """
        td = page.get_text("dict")
        for b in td.get("blocks", []):
            if b.get("bbox", [0, 0, 0, 0])[1] < 65 and "lines" in b:
                lines_text = []
                max_s = 0
                for line in b.get("lines", []):
                    line_str = " ".join([s.get("text", "") for s in line.get("spans", [])]).strip()
                    if line_str:
                        lines_text.append(line_str)
                    for s in line.get("spans", []):
                        max_s = max(max_s, s.get("size", 0))
                if max_s >= 13.0 and lines_text:
                    full_banner = " ".join(lines_text)
                    title = lines_text[-1] if len(lines_text) > 1 else full_banner
                    title = re.sub(r"^(?:Architecture|Overview|Presentation|Section|Topic)\s*[–\-]\s*.*?\s*", "", title).strip()
                    title = re.sub(r"^Scope:\s*", "", title).strip()
                    title = re.sub(r"\{DRAFT\}", "", title).strip()
                    if len(title) >= 3 and not re.match(r"^(TM|\d+|NXP|freescale|freescale['\"]?)$", title, re.I):
                        return title

        if _ocr_engine is not None:
            try:
                rect = fitz.Rect(0, 0, page.rect.width, 72)
                pix = page.get_pixmap(clip=rect, dpi=120)
                res, _ = _ocr_engine(pix.tobytes("png"))
                if res:
                    words = []
                    for r in res:
                        t = r[1].strip()
                        if t and not re.match(r"^(NXP|MXP|MP|TM|WL|NP|freescale|freescale['\"]?)$", t, re.I):
                            words.append(t)
                    if words:
                        candidate = " ".join(words)
                        candidate = re.sub(r"^[\.\?\-\s]+", "", candidate).strip()
                        if len(candidate) >= 3:
                            return candidate
            except Exception as e:
                logger.debug(f"Header OCR failed for page: {e}")

        return None

    def _extract_page_ocr(self, page: fitz.Page) -> str:
        """Extracts diagram and annotation text via OCR for visual-heavy slides."""
        if _ocr_engine is None:
            return ""
        try:
            pix = page.get_pixmap(dpi=140)
            res, _ = _ocr_engine(pix.tobytes("png"))
            if not res:
                return ""
            lines = []
            for r in res:
                t = r[1].strip()
                if t and not re.match(r"^(NXP|MXP|MP|TM|WL|NP|freescale|freescale['\"]?)$", t, re.I):
                    lines.append(t)
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Full page OCR failed: {e}")
            return ""

    def parse(self) -> Dict[str, Any]:
        """
        Parse the PDF file and return page-by-page structured contents.
        """
        try:
            doc = fitz.open(self.file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {self.file_path}: {str(e)}")
            raise PDFParserError(f"Corrupt or unreadable PDF: {str(e)}")

        if doc.page_count == 0:
            doc.close()
            raise PDFParserError("The uploaded PDF contains 0 pages (empty file).")

        total_pages = doc.page_count
        pages_data = []
        all_detected_sections = set()
        total_text_length = 0
        current_section = "Document Overview"

        # Generic detection of slide presentation decks based on page aspect ratio (landscape)
        first_page = doc[0]
        is_slide_deck = (first_page.rect.width > first_page.rect.height * 1.15)

        for page_num in range(1, total_pages + 1):
            page = doc.load_page(page_num - 1)
            page_text = page.get_text("text").strip()

            effective_page = page_num
            slide_title = None
            ocr_text = ""

            if is_slide_deck:
                text_dict = page.get_text("dict")
                # Detect slide number
                for b in text_dict.get("blocks", []):
                    for line in b.get("lines", []):
                        span_text = " ".join([s.get("text", "") for s in line.get("spans", [])]).strip()
                        m = re.search(r"(?:Document ID\s*\d+\s*:\s*\S+\s+|Slide\s+|Page\s+)?(\d{1,4})$", span_text)
                        if m and (b.get("bbox", [0, 0, 0, 0])[1] > 480 or b.get("bbox", [0, 0, 0, 0])[1] < 60):
                            try:
                                num_val = int(m.group(1))
                                if 1 <= num_val <= total_pages * 2:
                                    effective_page = num_val
                                    break
                            except (ValueError, IndexError):
                                pass
                    if effective_page != page_num:
                        break

                slide_title = self._extract_slide_title(page, total_pages)
                if slide_title:
                    current_section = slide_title

                # For visual/diagram slides with sparse text or multiple images, extract OCR
                if len(page_text) < 150 or (is_slide_deck and len(page.get_images()) >= 3):
                    ocr_text = self._extract_page_ocr(page)
                    if ocr_text:
                        page_text = (page_text + "\n" + ocr_text).strip()

            total_text_length += len(page_text)
            incoming_section = current_section

            # 1. Extract text blocks and identify headings
            blocks_data, new_sections, heading_positions = self._extract_text_blocks(
                page, effective_page, incoming_section, is_slide_deck=is_slide_deck, slide_title=slide_title, ocr_text=ocr_text
            )
            all_detected_sections.update(new_sections)

            # 2. Extract tables with exact section context alignment
            tables_data = self._extract_tables(page, effective_page, incoming_section, heading_positions)

            if new_sections:
                current_section = list(new_sections)[-1]

            pages_data.append({
                "page": effective_page,
                "text": page_text,
                "blocks": blocks_data,
                "tables": tables_data,
                "current_section": current_section
            })

        doc.close()

        if total_text_length < 20:
            raise PDFParserError(
                "PDF contains no extractable text. It may be a scanned image or empty. OCR is required for scanned documents."
            )

        return {
            "page_count": total_pages,
            "sections_count": len(all_detected_sections) if all_detected_sections else 1,
            "detected_sections": sorted(list(all_detected_sections)),
            "pages": pages_data
        }

    def _is_heading(self, text: str, is_bold: bool, max_font_size: float) -> bool:
        """Determines if a text block represents a section heading."""
        t = text.strip()
        if len(t) < 3 or len(t) > 200:
            return False
        # Filter false positives and table header text
        if t.startswith("(") and t.endswith(")"):
            return False
        if re.match(r"^(Page\s+\d+|Sl\.\s*No|UID|Table\s+\d+|Figure\s+\d+|Total\s+Pending|\(?Rs\.?\)?)", t, re.IGNORECASE):
            return False
        if re.search(r"Name of the (College|Institution)|Sanctioned amount|Receipt of Utilisation", t, re.IGNORECASE):
            return False
        if re.match(r"^(Receipt of|SE from|Sanctioned|amount|from the Institution|the institutions|Institution)\b", t, re.IGNORECASE):
            return False
        if t.startswith(("o ", "• ", "- ", "* ", "", "", "\uf0d8", "\uf06e")):
            return False
        if re.match(r"^(#define|#include|typedef|const\s+|extern\s+|void\s+|static\s+|struct\s+|union\s+|enum\s+|return\b)", t):
            return False
        if re.match(r"^(Task|Properties|Example|Note|Problem|Solution|API|Inputs|Conclusions?|Restrictions?|Use cases?)\b:?", t, re.IGNORECASE):
            return False
        if "page id:" in t.lower() or "document id" in t.lower():
            return False

        # Specific series headings e.g. "49th Series SPP (2025-26): List of Institutions..."
        if re.search(r"\b\d+(?:st|nd|rd|th)\s+Series\b", t, re.IGNORECASE) and len(t) <= 190:
            return True

        # Common procedural & architecture headings
        if re.match(
            r"^(Processing Fee|Submission Process|Proposal Requirements|Budget and Project Information|General Guidelines|Project Proposal Format|Instructions to the|Role of SPP|Who can be SPP|How to apply|Steps to follow|Eligibility Criteria|Important Dates)",
            t,
            re.IGNORECASE
        ):
            return True

        # Numbered headings e.g. "1. System Overview", "2. Software Component Architecture"
        if re.match(r"^\d+(?:\.\d+)*[\.\s]+[A-Z][A-Za-z0-9_\-\s]{2,80}:?$", t):
            return True

        # Standard numbered or uppercase headings
        if self.HEADING_REGEX.match(t):
            return True

        # Bold short titles not ending in full stop
        if (is_bold or max_font_size >= 12.0) and len(t) < 70 and not t.endswith("."):
            return True

        return False

    def _extract_text_blocks(
        self,
        page: fitz.Page,
        page_num: int,
        active_section: str,
        is_slide_deck: bool = False,
        slide_title: Optional[str] = None,
        ocr_text: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """
        Extract text blocks with font analysis to identify headings and structural context.
        Returns (blocks, new_sections, heading_positions).
        """
        blocks = []
        new_sections = []
        heading_positions = []
        current_sec = slide_title if (is_slide_deck and slide_title) else active_section

        if is_slide_deck and slide_title:
            new_sections.append(slide_title)

        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue

                block_text_lines = []
                max_font_size = 0.0
                is_bold = False

                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        line_text += text
                        size = span.get("size", 10.0)
                        flags = span.get("flags", 0)
                        if size > max_font_size:
                            max_font_size = size
                        if flags & 2 or "bold" in span.get("font", "").lower():
                            is_bold = True
                    if line_text.strip():
                        block_text_lines.append(line_text.strip())

                block_full_text = " ".join(block_text_lines).strip()
                if not block_full_text:
                    continue

                if is_slide_deck:
                    if block.get("bbox", [0, 0, 0, 0])[1] < 60:
                        heading_positions.append({
                            "heading": current_sec,
                            "bbox": block.get("bbox", (0, 0, 0, 0))
                        })
                    else:
                        blocks.append({
                            "section": current_sec,
                            "text": block_full_text,
                            "page": page_num,
                            "bbox": block.get("bbox", (0, 0, 0, 0))
                        })
                else:
                    if self._is_heading(block_full_text, is_bold, max_font_size):
                        clean_heading = block_full_text.rstrip(":").strip()
                        clean_heading = re.sub(r"\s*Sl\.\s*No\.?\s*$", "", clean_heading, flags=re.IGNORECASE).strip()
                        # Clean up multiple spaces
                        clean_heading = re.sub(r"\s+", " ", clean_heading)
                        current_sec = clean_heading
                        new_sections.append(current_sec)
                        heading_positions.append({
                            "heading": clean_heading,
                            "bbox": block.get("bbox", (0, 0, 0, 0))
                        })
                    else:
                        blocks.append({
                            "section": current_sec,
                            "text": block_full_text,
                            "page": page_num,
                            "bbox": block.get("bbox", (0, 0, 0, 0))
                        })

            if is_slide_deck and ocr_text:
                existing_texts = set(b["text"].strip().lower() for b in blocks)
                for line in ocr_text.split("\n"):
                    l_str = line.strip()
                    if l_str and l_str.lower() not in existing_texts and len(l_str) > 2:
                        blocks.append({
                            "section": current_sec,
                            "text": l_str,
                            "page": page_num,
                            "bbox": (0, 0, 0, 0)
                        })
                        existing_texts.add(l_str.lower())

        except Exception as e:
            logger.warning(f"Fallback to simple text for page {page_num}: {e}")
            simple_text = page.get_text("text").strip()
            if simple_text:
                blocks.append({
                    "section": current_sec,
                    "text": simple_text,
                    "page": page_num,
                    "bbox": (0, 0, 0, 0)
                })

        return blocks, new_sections, heading_positions

    def _extract_tables(
        self,
        page: fitz.Page,
        page_num: int,
        default_section: str,
        heading_positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract tables using PyMuPDF and bind each table to its nearest heading."""
        tables = []
        try:
            tabs = page.find_tables()
            if tabs and hasattr(tabs, "tables"):
                for idx, tab in enumerate(tabs.tables):
                    header = tab.header.names if tab.header else []
                    rows = tab.extract()
                    md_lines = []

                    def _clean_cell(val):
                        s = str(val or "").strip().replace("\r", " ").replace("\n", " ")
                        s = re.sub(r"(\w+)\s*_\s*(\w+)", r"\1_\2", s)
                        s = re.sub(r"(\w+)\s+(\w+)\s+_", r"\1_\2", s)
                        return re.sub(r"\s+", " ", s).strip()

                    clean_headers = [_clean_cell(h) for h in (header or (rows[0] if rows else []))]
                    if clean_headers:
                        md_lines.append("| " + " | ".join(clean_headers) + " |")
                        md_lines.append("| " + " | ".join(["---"] * len(clean_headers)) + " |")

                    data_rows = rows[1:] if tab.header else rows
                    for r in data_rows:
                        if r:
                            row_cells = [_clean_cell(c) for c in r]
                            md_lines.append("| " + " | ".join(row_cells) + " |")

                    table_content = "\n".join(md_lines)
                    if table_content.strip():
                        # Determine specific section for this table based on heading positions on this page
                        table_section = default_section
                        tab_y0 = tab.bbox[1] if hasattr(tab, "bbox") else 0
                        # Find closest heading on this page whose y is above or at top of table
                        closest_h = None
                        min_dist = float("inf")
                        for hp in heading_positions:
                            h_y0 = hp["bbox"][1]
                            if h_y0 <= tab_y0 + 10:  # Heading starts above or at table top
                                dist = tab_y0 - h_y0
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_h = hp["heading"]

                        if closest_h:
                            table_section = closest_h

                        tables.append({
                            "table_index": idx + 1,
                            "content": table_content,
                            "page": page_num,
                            "section": table_section
                        })
        except Exception as e:
            logger.debug(f"Table extraction notice on page {page_num}: {str(e)}")

        return tables
