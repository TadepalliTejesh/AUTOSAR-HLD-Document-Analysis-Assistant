from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.core.logging import logger


class VectorStoreService:
    _instance: Optional["VectorStoreService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
            cls._instance._init_chroma()
        return cls._instance

    def _init_chroma(self):
        chroma_path = Path(settings.CHROMA_DIR).resolve()
        chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing Persistent ChromaDB client at: {chroma_path}")
        
        self.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="autosar_hld_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Store chunk texts, metadatas, and embeddings in ChromaDB.
        """
        if not chunks:
            return 0

        # Remove existing chunks for this document first if re-indexing
        self.delete_by_document(document_id)

        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # ChromaDB batch size limits (max ~5000 per call)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embs,
                metadatas=batch_metas
            )

        logger.info(f"Successfully stored {len(ids)} chunks for document {document_id} in ChromaDB")
        return len(ids)

    def search(
        self,
        document_id: int,
        query_embedding: List[float],
        top_k: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Search for top-k relevant chunks strictly isolated by document_id.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"document_id": str(document_id)},
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB search error for document {document_id}: {e}")
            return []

        retrieved_chunks = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc_text, meta, distance in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance: distance = 1 - cosine_similarity
            similarity = round(max(0.0, 1.0 - float(distance)), 4)
            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "content": doc_text,
                "document_id": int(meta.get("document_id", document_id)),
                "document_name": meta.get("document_name", ""),
                "page": int(meta.get("page", 1)),
                "section": meta.get("section", "General"),
                "source_type": meta.get("source_type", "text"),
                "similarity": similarity
            })

        # Sort descending by similarity
        retrieved_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        return retrieved_chunks

    def delete_by_document(self, document_id: int) -> bool:
        """Purge all chunks associated with a document."""
        try:
            self.collection.delete(where={"document_id": str(document_id)})
            return True
        except Exception as e:
            logger.warning(f"ChromaDB delete error for doc {document_id}: {e}")
            return False

    def count_chunks(self, document_id: Optional[int] = None) -> int:
        """Count indexed chunks, optionally filtered by document_id."""
        if document_id is not None:
            results = self.collection.get(where={"document_id": str(document_id)})
            return len(results.get("ids", []))
        return self.collection.count()

    def get_all_chunks(self, document_id: int) -> List[Dict[str, Any]]:
        """
        Fetch all chunks belonging to a document for hybrid retrieval.
        Strictly isolated by document_id.
        """
        try:
            results = self.collection.get(
                where={"document_id": str(document_id)},
                include=["documents", "metadatas"]
            )
            if not results or not results.get("ids"):
                return []

            chunks = []
            for chunk_id, doc_text, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": doc_text,
                    "document_id": int(meta.get("document_id", document_id)),
                    "document_name": meta.get("document_name", ""),
                    "page": int(meta.get("page", 1)),
                    "section": meta.get("section", "General"),
                    "source_type": meta.get("source_type", "text"),
                    "similarity": 0.0
                })
            return chunks
        except Exception as e:
            logger.error(f"Error fetching all chunks for document {document_id}: {e}")
            return []



vector_store_service = VectorStoreService()
