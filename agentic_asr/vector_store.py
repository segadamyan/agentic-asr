"""Simple local FAISS vector store for transcription RAG."""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import Config

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """Simple FAISS-based vector store for transcriptions."""
    
    def __init__(
        self,
        store_path: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.store_path = Path(store_path) if store_path else Config.FAISS_INDICES_DIR
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name or Config.VECTOR_STORE_MODEL
        self.model = None
        self.index = None
        self.metadata = {}
        
        self.index_file = self.store_path / "faiss_index.bin"
        self.metadata_file = self.store_path / "metadata.json"
        
    def initialize(self):
        """Initialize the vector store."""
        logger.info(f"Initializing vector store with model: {self.model_name}")
        
        self.model = SentenceTransformer(self.model_name)
        
        self._load_index()
        
        logger.info(f"Vector store initialized. Index size: {self.index.ntotal if self.index else 0}")
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_file))
                
                # Load metadata
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Get embedding dimension from model
        sample_embedding = self.model.encode(["test"])
        dimension = sample_embedding.shape[1]
        
        # Create flat index (exact search)
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = {
            "chunks": {},
            "documents": {},
            "next_id": 0
        }
        
        logger.info(f"Created new FAISS index with dimension {dimension}")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            logger.debug("Saved index and metadata")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple text chunking."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def add_document(
        self,
        filename: str,
        content: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """Add a document to the vector store."""
        logger.info(f"Adding document: {filename}")
        
        # Use config defaults if not provided
        chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
        overlap = overlap or Config.DEFAULT_CHUNK_OVERLAP
        
        # Generate document ID
        doc_id = hashlib.sha256(f"{filename}_{content[:100]}".encode()).hexdigest()[:16]
        
        # Remove existing document if it exists
        if doc_id in self.metadata["documents"]:
            self.remove_document(doc_id)
        
        # Chunk the text
        chunks = self._chunk_text(content, chunk_size, overlap)
        
        if not chunks:
            return {
                "success": False,
                "error": "No chunks generated from document"
            }
        
        # Generate embeddings
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        
        # Add to FAISS index
        start_id = self.metadata["next_id"]
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = start_id + i
            chunk_ids.append(chunk_id)
            
            self.metadata["chunks"][str(chunk_id)] = {
                "document_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "content": chunk,
                "char_count": len(chunk)
            }
        
        self.metadata["documents"][doc_id] = {
            "filename": filename,
            "chunk_ids": chunk_ids,
            "total_chunks": len(chunks),
            "char_count": len(content)
        }
        
        self.metadata["next_id"] = start_id + len(chunks)
        
        # Save to disk
        self._save_index()
        
        logger.info(f"Added document {filename} with {len(chunks)} chunks")
        
        return {
            "success": True,
            "document_id": doc_id,
            "filename": filename,
            "total_chunks": len(chunks),
            "chunks_processed": len(chunks)
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search
        similarities, indices = self.index.search(
            query_embedding.astype(np.float32), 
            min(top_k, self.index.ntotal)
        )
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1 or sim < similarity_threshold:
                continue
            
            chunk_data = self.metadata["chunks"].get(str(idx))
            if chunk_data:
                results.append({
                    "chunk_id": str(idx),
                    "document_id": chunk_data["document_id"],
                    "filename": chunk_data["filename"],
                    "chunk_index": chunk_data["chunk_index"],
                    "content": chunk_data["content"],
                    "similarity_score": float(sim),
                    "char_count": chunk_data["char_count"]
                })
        
        return results
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the store."""
        if document_id not in self.metadata["documents"]:
            return False
        
        # Note: FAISS doesn't support efficient deletion
        # For a simple implementation, we'll just remove from metadata
        # In practice, you'd need to rebuild the index periodically
        
        doc_data = self.metadata["documents"][document_id]
        chunk_ids = doc_data["chunk_ids"]
        
        # Remove chunk metadata
        for chunk_id in chunk_ids:
            self.metadata["chunks"].pop(str(chunk_id), None)
        
        # Remove document metadata
        self.metadata["documents"].pop(document_id, None)
        
        # Save metadata
        self._save_index()
        
        logger.info(f"Removed document {document_id} from metadata")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.metadata.get("documents", {})),
            "total_chunks": len(self.metadata.get("chunks", {})),
            "model_name": self.model_name,
            "store_path": str(self.store_path),
            "index_file_exists": self.index_file.exists(),
            "index_file_size": self.index_file.stat().st_size if self.index_file.exists() else 0
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the store."""
        return [
            {
                "document_id": doc_id,
                "filename": doc_data["filename"],
                "total_chunks": doc_data["total_chunks"],
                "char_count": doc_data["char_count"]
            }
            for doc_id, doc_data in self.metadata.get("documents", {}).items()
        ]


# Global instance
_vector_store = None


def get_vector_store() -> SimpleVectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = SimpleVectorStore()
        _vector_store.initialize()
    return _vector_store
