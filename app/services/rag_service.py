import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session

from app.models import Document, DocumentCollection, DocumentSearchResult, ContextSource
from app.services.vector_service import vector_service

logger = logging.getLogger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def retrieve_context(
        self,
        query: str,
        user_id: int,
        document_ids: Optional[List[str]] = None,
        collection_id: Optional[str] = None,
        max_chunks: int = 10,
        relevance_threshold: float = 0.7
    ) -> List[ContextSource]:
        """Retrieve relevant document chunks for RAG"""
        try:
            # Resolve target documents
            target_document_ids = self._resolve_target_documents(
                user_id, document_ids, collection_id
            )
            
            if not target_document_ids:
                logger.warning("No documents found for context retrieval")
                return []
            
            # Perform vector search
            search_results = vector_service.similarity_search(
                user_id=user_id,
                query=query,
                document_filters=target_document_ids,
                limit=max_chunks * 2,  # Get more to allow for diversification
                threshold=relevance_threshold
            )
            
            # Diversify results by document
            diversified_results = self._diversify_chunks_by_document(search_results, max_chunks)
            
            # Convert to ContextSource objects
            context_sources = []
            document_cache = {}  # Cache document info to avoid repeated queries
            
            for result in diversified_results:
                document_id = result['metadata']['document_id']
                
                # Get document info (use cache to avoid repeated DB queries)
                if document_id not in document_cache:
                    document = self.db.query(Document).filter(
                        Document.id == document_id,
                        Document.user_id == user_id
                    ).first()
                    
                    if document:
                        document_cache[document_id] = {
                            'name': document.original_filename,
                            'file_type': document.file_type
                        }
                    else:
                        continue
                
                doc_info = document_cache[document_id]
                
                # Extract page number from metadata
                page_number = result['metadata'].get('page_number') or result['metadata'].get('slide_number')
                
                context_sources.append(ContextSource(
                    document_id=document_id,
                    document_name=doc_info['name'],
                    chunk_text=result['content'],
                    page_number=page_number,
                    relevance_score=result['relevance_score']
                ))
            
            logger.info(f"Retrieved {len(context_sources)} context sources for query")
            return context_sources
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise
    
    def search_documents(
        self,
        query: str,
        user_id: int,
        document_ids: Optional[List[str]] = None,
        collection_id: Optional[str] = None,
        limit: int = 10,
        relevance_threshold: float = 0.7
    ) -> List[DocumentSearchResult]:
        """Search across user documents and return detailed results"""
        try:
            # Resolve target documents
            target_document_ids = self._resolve_target_documents(
                user_id, document_ids, collection_id
            )
            
            if not target_document_ids:
                return []
            
            # Perform vector search
            search_results = vector_service.similarity_search(
                user_id=user_id,
                query=query,
                document_filters=target_document_ids,
                limit=limit,
                threshold=relevance_threshold
            )
            
            # Convert to DocumentSearchResult objects
            results = []
            document_cache = {}
            
            for result in search_results:
                document_id = result['metadata']['document_id']
                
                # Get document info
                if document_id not in document_cache:
                    document = self.db.query(Document).filter(
                        Document.id == document_id,
                        Document.user_id == user_id
                    ).first()
                    
                    if document:
                        document_cache[document_id] = document.original_filename
                    else:
                        continue
                
                document_name = document_cache[document_id]
                
                # Extract page number from metadata
                page_number = result['metadata'].get('page_number') or result['metadata'].get('slide_number')
                
                results.append(DocumentSearchResult(
                    chunk_id=result.get('chunk_id', ''),
                    document_id=document_id,
                    document_name=document_name,
                    content=result['content'],
                    page_number=page_number,
                    relevance_score=result['relevance_score'],
                    metadata=result['metadata']
                ))
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def build_context_prompt(
        self,
        context_sources: List[ContextSource],
        max_context_length: int = 4000
    ) -> str:
        """Build context prompt from retrieved sources"""
        if not context_sources:
            return ""
        
        # Sort by relevance score
        sorted_sources = sorted(context_sources, key=lambda x: x.relevance_score, reverse=True)
        
        context_parts = []
        current_length = 0
        
        for source in sorted_sources:
            # Format source with metadata
            source_text = f"[Document: {source.document_name}"
            if source.page_number:
                source_text += f", Page {source.page_number}"
            source_text += f"]\n{source.chunk_text}\n\n"
            
            # Check if adding this source would exceed length limit
            if current_length + len(source_text) > max_context_length:
                break
            
            context_parts.append(source_text)
            current_length += len(source_text)
        
        if context_parts:
            context_prompt = "Based on the following documents:\n\n" + "".join(context_parts)
            context_prompt += "Please answer the user's question using the information from these documents. If the documents don't contain relevant information, please say so."
            return context_prompt
        
        return ""
    
    def _resolve_target_documents(
        self,
        user_id: int,
        document_ids: Optional[List[str]] = None,
        collection_id: Optional[str] = None
    ) -> List[str]:
        """Resolve target document IDs from various inputs"""
        if document_ids:
            # Validate that documents belong to user
            valid_docs = self.db.query(Document.id).filter(
                Document.id.in_(document_ids),
                Document.user_id == user_id,
                Document.processing_status == "completed"
            ).all()
            
            return [doc.id for doc in valid_docs]
        
        elif collection_id:
            # Get documents from collection
            collection = self.db.query(DocumentCollection).filter(
                DocumentCollection.id == collection_id,
                DocumentCollection.user_id == user_id
            ).first()
            
            if collection and collection.document_ids:
                doc_ids = json.loads(collection.document_ids)
                
                # Validate documents exist and are completed
                valid_docs = self.db.query(Document.id).filter(
                    Document.id.in_(doc_ids),
                    Document.user_id == user_id,
                    Document.processing_status == "completed"
                ).all()
                
                return [doc.id for doc in valid_docs]
        
        else:
            # No specific documents/collection specified - use all user's completed documents
            docs = self.db.query(Document.id).filter(
                Document.user_id == user_id,
                Document.processing_status == "completed"
            ).all()
            
            return [doc.id for doc in docs]
        
        return []
    
    def _diversify_chunks_by_document(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: int
    ) -> List[Dict[str, Any]]:
        """Ensure diverse chunks from different documents"""
        if not chunks:
            return []
        
        # Group chunks by document
        document_chunks = {}
        for chunk in chunks:
            doc_id = chunk['metadata']['document_id']
            if doc_id not in document_chunks:
                document_chunks[doc_id] = []
            document_chunks[doc_id].append(chunk)
        
        # Sort chunks within each document by relevance
        for doc_id in document_chunks:
            document_chunks[doc_id].sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Round-robin selection from different documents
        result = []
        max_per_doc = max(1, max_chunks // len(document_chunks))
        
        # First, take top chunks from each document
        for doc_id, doc_chunks in document_chunks.items():
            result.extend(doc_chunks[:max_per_doc])
        
        # If we haven't reached max_chunks, add more from highest-scoring chunks
        if len(result) < max_chunks:
            remaining_chunks = []
            for doc_id, doc_chunks in document_chunks.items():
                remaining_chunks.extend(doc_chunks[max_per_doc:])
            
            # Sort remaining by score and add until we reach max_chunks
            remaining_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            result.extend(remaining_chunks[:max_chunks - len(result)])
        
        # Final sort by relevance score
        result.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return result[:max_chunks]
    
    def get_context_statistics(self, user_id: int) -> Dict[str, Any]:
        """Get statistics about user's document context"""
        try:
            # Get document stats from vector service
            vector_stats = vector_service.get_user_document_stats(user_id)
            
            # Get database stats
            total_documents = self.db.query(Document).filter(
                Document.user_id == user_id
            ).count()
            
            completed_documents = self.db.query(Document).filter(
                Document.user_id == user_id,
                Document.processing_status == "completed"
            ).count()
            
            total_collections = self.db.query(DocumentCollection).filter(
                DocumentCollection.user_id == user_id
            ).count()
            
            return {
                "total_documents": total_documents,
                "completed_documents": completed_documents,
                "total_collections": total_collections,
                "total_chunks": vector_stats.get("total_chunks", 0),
                "documents_in_vector_db": vector_stats.get("total_documents", 0),
                "vector_service_status": "healthy" if vector_stats.get("total_chunks", 0) > 0 else "no_data"
            }
            
        except Exception as e:
            logger.error(f"Error getting context statistics: {e}")
            return {
                "total_documents": 0,
                "completed_documents": 0,
                "total_collections": 0,
                "total_chunks": 0,
                "documents_in_vector_db": 0,
                "vector_service_status": "error",
                "error": str(e)
            }