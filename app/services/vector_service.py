import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from app.services.document_parser import DocumentChunk

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector database operations with ChromaDB"""
    
    def __init__(self, persist_directory: str = "./data/chroma"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info(f"VectorService initialized with persist directory: {self.persist_directory}")
    
    def _get_collection_name(self, user_id: int) -> str:
        """Generate collection name for user"""
        return f"user_{user_id}_documents"
    
    def _get_or_create_collection(self, user_id: int):
        """Get or create ChromaDB collection for user"""
        collection_name = self._get_collection_name(user_id)
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(collection_name)
            logger.info(f"Retrieved existing collection: {collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"user_id": user_id}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        return collection
    
    def store_document_chunks(
        self, 
        user_id: int, 
        document_id: str, 
        chunks: List[DocumentChunk]
    ) -> List[str]:
        """Store document chunks in vector database"""
        if not chunks:
            return []
        
        collection = self._get_or_create_collection(user_id)
        
        # Prepare data for ChromaDB
        chunk_ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Generate unique chunk ID
            chunk_id = f"{document_id}_chunk_{i}_{uuid.uuid4().hex[:8]}"
            chunk_ids.append(chunk_id)
            
            # Document content
            documents.append(chunk.content)
            
            # Generate embedding
            embedding = self.embedding_model.encode(chunk.content).tolist()
            embeddings.append(embedding)
            
            # Metadata
            metadata = {
                "document_id": document_id,
                "chunk_index": i,
                "user_id": user_id,
                **chunk.metadata
            }
            metadatas.append(metadata)
        
        try:
            # Store in ChromaDB
            collection.add(
                ids=chunk_ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(chunk_ids)} chunks for document {document_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error storing chunks in vector database: {e}")
            raise
    
    def similarity_search(
        self,
        user_id: int,
        query: str,
        document_filters: Optional[List[str]] = None,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in vector database"""
        collection = self._get_or_create_collection(user_id)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Prepare where clause for filtering (ChromaDB limitation: single operator only)
        if document_filters:
            # Filter by document_id only (since user collections are already isolated)
            where_clause = {"document_id": {"$in": document_filters}}
        else:
            # Filter by user_id only
            where_clause = {"user_id": user_id}
        
        try:
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # Calculate similarity score (ChromaDB returns distances, we want similarity)
                    distance = results['distances'][0][i]
                    # Use cosine similarity conversion: similarity = 1 - (distance^2 / 2)
                    # For better scoring, use exponential decay: similarity = exp(-distance)
                    import math
                    similarity_score = math.exp(-distance)
                    
                    # Filter by threshold
                    if similarity_score >= threshold:
                        search_results.append({
                            'chunk_id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'relevance_score': similarity_score
                        })
            
            logger.info(f"Found {len(search_results)} relevant chunks for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def delete_document_chunks(self, user_id: int, document_id: str) -> bool:
        """Delete all chunks for a specific document"""
        collection = self._get_or_create_collection(user_id)
        
        try:
            # Get all chunk IDs for this document (user already isolated by collection)
            results = collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results['ids']:
                # Delete chunks
                collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                logger.warning(f"No chunks found for document {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            raise
    
    def get_document_chunk_count(self, user_id: int, document_id: str) -> int:
        """Get number of chunks for a specific document"""
        collection = self._get_or_create_collection(user_id)
        
        try:
            # Since each user has their own collection, only filter by document_id
            results = collection.get(
                where={"document_id": document_id},
                include=[]
            )
            return len(results['ids']) if results['ids'] else 0
            
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0
    
    def get_user_document_stats(self, user_id: int) -> Dict[str, Any]:
        """Get statistics about user's documents in vector database"""
        collection = self._get_or_create_collection(user_id)
        
        try:
            # Get all user's chunks
            results = collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            if not results['ids']:
                return {
                    "total_chunks": 0,
                    "total_documents": 0,
                    "documents": {}
                }
            
            # Analyze by document
            document_stats = {}
            total_chunks = len(results['ids'])
            
            for metadata in results['metadatas']:
                doc_id = metadata.get('document_id')
                if doc_id:
                    if doc_id not in document_stats:
                        document_stats[doc_id] = 0
                    document_stats[doc_id] += 1
            
            return {
                "total_chunks": total_chunks,
                "total_documents": len(document_stats),
                "documents": document_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting user document stats: {e}")
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "documents": {}
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check vector service health"""
        try:
            # Test basic operations
            collections = self.client.list_collections()
            
            return {
                "status": "healthy",
                "total_collections": len(collections),
                "embedding_model": "all-MiniLM-L6-v2",
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Vector service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global instance
vector_service = VectorService()