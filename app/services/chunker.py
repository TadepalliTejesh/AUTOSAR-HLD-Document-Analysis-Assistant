import uuid
import re
from typing import List, Dict, Any


class SectionAwareChunker:
    """
    Splits parsed PDF data into section-aware chunks maintaining:
    Document -> Page -> Section -> Chunk hierarchy.
    """

    def __init__(self, target_chunk_size: int = 800, chunk_overlap: int = 150):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        document_id: int,
        document_name: str,
        parsed_pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        chunks = []
        chunk_counter = 1

        for page_data in parsed_pages:
            page_num = page_data["page"]

            # 1. Process Tables first as standalone high-context chunks
            for tbl in page_data.get("tables", []):
                chunk_id = f"doc_{document_id}_p{page_num}_tbl_{tbl.get('table_index', 1)}"
                section = tbl.get("section") or page_data.get("current_section", "System Architecture")
                content = tbl["content"]
                
                # Contextualized text for embedding
                context_header = f"[Document: {document_name} | Page {page_num} | Section: {section} | Type: Table]\n"
                enriched_text = context_header + content

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": enriched_text,
                    "raw_content": content,
                    "metadata": {
                        "document_id": str(document_id),
                        "document_name": document_name,
                        "page": int(page_num),
                        "section": str(section),
                        "chunk_id": chunk_id,
                        "source_type": "table"
                    }
                })
                chunk_counter += 1

            # 2. Group text blocks by section on the page
            blocks = page_data.get("blocks", [])
            section_groups: Dict[str, List[str]] = {}
            for blk in blocks:
                sec = blk.get("section", page_data.get("current_section", "General"))
                txt = blk.get("text", "").strip()
                if txt:
                    if sec not in section_groups:
                        section_groups[sec] = []
                    section_groups[sec].append(txt)

            # If no blocks extracted from dict, fallback to page raw text
            if not section_groups and page_data.get("text"):
                sec = page_data.get("current_section", "General")
                section_groups[sec] = [page_data["text"]]

            # 3. Create section-aware chunks
            for sec_name, paragraphs in section_groups.items():
                sec_text = "\n\n".join(paragraphs).strip()
                if not sec_text:
                    continue

                if len(sec_text) <= self.target_chunk_size:
                    # Fits comfortably in one chunk
                    chunk_id = f"doc_{document_id}_p{page_num}_c{chunk_counter}"
                    context_header = f"[Document: {document_name} | Page {page_num} | Section: {sec_name}]\n"
                    enriched_text = context_header + sec_text

                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": enriched_text,
                        "raw_content": sec_text,
                        "metadata": {
                            "document_id": str(document_id),
                            "document_name": document_name,
                            "page": int(page_num),
                            "section": str(sec_name),
                            "chunk_id": chunk_id,
                            "source_type": "text"
                        }
                    })
                    chunk_counter += 1
                else:
                    # Sub-chunk longer section content along sentence/paragraph boundaries
                    sub_chunks = self._split_text_with_overlap(sec_text, self.target_chunk_size, self.chunk_overlap)
                    for idx, sub_txt in enumerate(sub_chunks):
                        chunk_id = f"doc_{document_id}_p{page_num}_c{chunk_counter}_{idx + 1}"
                        context_header = f"[Document: {document_name} | Page {page_num} | Section: {sec_name} (Part {idx + 1})]\n"
                        enriched_text = context_header + sub_txt

                        chunks.append({
                            "chunk_id": chunk_id,
                            "text": enriched_text,
                            "raw_content": sub_txt,
                            "metadata": {
                                "document_id": str(document_id),
                                "document_name": document_name,
                                "page": int(page_num),
                                "section": str(sec_name),
                                "chunk_id": chunk_id,
                                "source_type": "text"
                            }
                        })
                        chunk_counter += 1

        return chunks

    def _split_text_with_overlap(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Splits long text while respecting sentence boundaries and maintaining overlap."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result_chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            s_len = len(sentence)
            if current_len + s_len > max_size and current_chunk:
                result_chunks.append(" ".join(current_chunk))
                # retain tail for overlap
                overlap_tokens = []
                overlap_len = 0
                for prev in reversed(current_chunk):
                    if overlap_len + len(prev) <= overlap:
                        overlap_tokens.insert(0, prev)
                        overlap_len += len(prev)
                    else:
                        break
                current_chunk = overlap_tokens
                current_len = overlap_len

            current_chunk.append(sentence)
            current_len += s_len

        if current_chunk:
            result_chunks.append(" ".join(current_chunk))

        return result_chunks
