import json
import logging
import uuid
import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from sqlalchemy.orm import Session

from app.models import (
    Document, DocumentChunk, DocumentCollection, DocumentStatus, DocumentType,
    DocumentResponse, DocumentCollectionResponse
)
from app.services.document_parser import DocumentParserService
from app.services.vector_service import vector_service
from app.database import get_db

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing document upload, processing, and storage"""
    
    def __init__(self, db: Session):
        self.db = db
        self.upload_dir = Path("uploads/documents")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def upload_document(
        self, 
        file_content: bytes, 
        filename: str, 
        user_id: int,
        file_type: DocumentType
    ) -> str:
        """Upload and save document file"""
        try:
            # Generate unique document ID and filename
            doc_id = str(uuid.uuid4())
            file_extension = Path(filename).suffix
            unique_filename = f"{doc_id}{file_extension}"
            file_path = self.upload_dir / unique_filename
            
            # Save file to disk
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Create database record
            document = Document(
                id=doc_id,
                user_id=user_id,
                filename=unique_filename,
                original_filename=filename,
                file_type=file_type.value,
                file_size=len(file_content),
                file_path=str(file_path),
                processing_status=DocumentStatus.PENDING.value
            )
            
            self.db.add(document)
            self.db.commit()
            
            logger.info(f"Document uploaded: {doc_id} ({filename})")
            return doc_id
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error uploading document: {e}")
            raise
    
    def process_document(self, document_id: str) -> bool:
        """Process document by parsing and storing chunks in vector database"""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            logger.error(f"Document not found: {document_id}")
            return False
        
        try:
            # Update status to processing
            document.processing_status = DocumentStatus.PROCESSING.value
            self.db.commit()
            
            # Parse document
            file_type = DocumentType(document.file_type)
            chunks = DocumentParserService.parse_document(document.file_path, file_type)
            
            if not chunks:
                logger.warning(f"No content extracted from document: {document_id}")
                document.processing_status = DocumentStatus.FAILED.value
                document.error_message = "No content could be extracted from the document"
                self.db.commit()
                return False
            
            # Store chunks in vector database
            chunk_ids = vector_service.store_document_chunks(
                user_id=document.user_id,
                document_id=document_id,
                chunks=chunks
            )
            
            # Store chunk metadata in database
            for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
                db_chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=i,
                    content=chunk.content,
                    chunk_metadata=json.dumps(chunk.metadata),
                    vector_id=chunk_id
                )
                self.db.add(db_chunk)
            
            # Update document status
            document.processing_status = DocumentStatus.COMPLETED.value
            document.chunk_count = len(chunks)
            document.error_message = None
            
            logger.info(f"Setting document status to: {document.processing_status}")
            self.db.commit()
            
            # Verify status was saved
            self.db.refresh(document)
            logger.info(f"Document status after commit: {document.processing_status}")
            
            logger.info(f"Document processed successfully: {document_id} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error processing document {document_id}: {e}")
            
            # Update status to failed
            try:
                document.processing_status = DocumentStatus.FAILED.value
                document.error_message = str(e)
                self.db.commit()
            except Exception:
                pass
            
            return False
    
    def upload_and_process_document(
        self, 
        file_content: bytes, 
        filename: str, 
        user_id: int,
        file_type: DocumentType
    ) -> Tuple[str, bool]:
        """Upload and immediately process document"""
        try:
            # Upload document
            doc_id = self.upload_document(file_content, filename, user_id, file_type)
            
            # Process document
            success = self.process_document(doc_id)
            
            return doc_id, success
            
        except Exception as e:
            logger.error(f"Error in upload_and_process_document: {e}")
            raise
    
    def _generate_document_url(self, document_id: str) -> str:
        """Generate URL for document download"""
        return f"/documents/{document_id}/download"
    
    def get_user_documents(
        self, 
        user_id: int, 
        status_filter: Optional[DocumentStatus] = None
    ) -> List[DocumentResponse]:
        """Get all documents for a user"""
        query = self.db.query(Document).filter(Document.user_id == user_id)
        
        if status_filter:
            query = query.filter(Document.processing_status == status_filter.value)
        
        documents = query.order_by(Document.created_at.desc()).all()
        
        return [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                original_filename=doc.original_filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                processing_status=doc.processing_status,
                chunk_count=doc.chunk_count,
                error_message=doc.error_message,
                document_url=self._generate_document_url(doc.id),
                created_at=doc.created_at.isoformat(),
                updated_at=doc.updated_at.isoformat()
            )
            for doc in documents
        ]
    
    def get_document(self, document_id: str, user_id: int) -> Optional[DocumentResponse]:
        """Get specific document by ID"""
        document = self.db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
        
        if not document:
            return None
        
        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            original_filename=document.original_filename,
            file_type=document.file_type,
            file_size=document.file_size,
            processing_status=document.processing_status,
            chunk_count=document.chunk_count,
            error_message=document.error_message,
            document_url=self._generate_document_url(document.id),
            created_at=document.created_at.isoformat(),
            updated_at=document.updated_at.isoformat()
        )
    
    def delete_document(self, document_id: str, user_id: int) -> bool:
        """Delete document and all associated data"""
        document = self.db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id
        ).first()
        
        if not document:
            logger.warning(f"Document not found for deletion: {document_id}")
            return False
        
        try:
            # Delete from vector database
            vector_service.delete_document_chunks(user_id, document_id)
            
            # Delete chunks from database
            self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete()
            
            # Delete file from disk
            try:
                if os.path.exists(document.file_path):
                    os.remove(document.file_path)
            except Exception as e:
                logger.warning(f"Could not delete file {document.file_path}: {e}")
            
            # Delete document record
            self.db.delete(document)
            self.db.commit()
            
            logger.info(f"Document deleted: {document_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting document {document_id}: {e}")
            raise
    
    def create_collection(
        self, 
        user_id: int, 
        name: str, 
        description: Optional[str] = None,
        document_ids: List[str] = []
    ) -> str:
        """Create a new document collection"""
        try:
            # Validate that all documents belong to the user
            if document_ids:
                valid_docs = self.db.query(Document.id).filter(
                    Document.id.in_(document_ids),
                    Document.user_id == user_id
                ).all()
                
                valid_doc_ids = [doc.id for doc in valid_docs]
                if len(valid_doc_ids) != len(document_ids):
                    raise ValueError("Some documents do not exist or do not belong to user")
            
            # Create collection
            collection_id = str(uuid.uuid4())
            collection = DocumentCollection(
                id=collection_id,
                user_id=user_id,
                name=name,
                description=description,
                document_ids=json.dumps(document_ids)
            )
            
            self.db.add(collection)
            self.db.commit()
            
            logger.info(f"Document collection created: {collection_id}")
            return collection_id
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating collection: {e}")
            raise
    
    def get_user_collections(self, user_id: int) -> List[DocumentCollectionResponse]:
        """Get all collections for a user"""
        collections = self.db.query(DocumentCollection).filter(
            DocumentCollection.user_id == user_id
        ).order_by(DocumentCollection.created_at.desc()).all()
        
        result = []
        for collection in collections:
            document_ids = json.loads(collection.document_ids) if collection.document_ids else []
            
            result.append(DocumentCollectionResponse(
                id=collection.id,
                name=collection.name,
                description=collection.description,
                document_ids=document_ids,
                document_count=len(document_ids),
                created_at=collection.created_at.isoformat(),
                updated_at=collection.updated_at.isoformat()
            ))
        
        return result
    
    def update_collection(
        self, 
        collection_id: str, 
        user_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> bool:
        """Update document collection"""
        collection = self.db.query(DocumentCollection).filter(
            DocumentCollection.id == collection_id,
            DocumentCollection.user_id == user_id
        ).first()
        
        if not collection:
            return False
        
        try:
            if name is not None:
                collection.name = name
            
            if description is not None:
                collection.description = description
            
            if document_ids is not None:
                # Validate documents
                if document_ids:
                    valid_docs = self.db.query(Document.id).filter(
                        Document.id.in_(document_ids),
                        Document.user_id == user_id
                    ).all()
                    
                    valid_doc_ids = [doc.id for doc in valid_docs]
                    if len(valid_doc_ids) != len(document_ids):
                        raise ValueError("Some documents do not exist or do not belong to user")
                
                collection.document_ids = json.dumps(document_ids)
            
            self.db.commit()
            logger.info(f"Collection updated: {collection_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating collection {collection_id}: {e}")
            raise
    
    def delete_collection(self, collection_id: str, user_id: int) -> bool:
        """Delete document collection"""
        collection = self.db.query(DocumentCollection).filter(
            DocumentCollection.id == collection_id,
            DocumentCollection.user_id == user_id
        ).first()
        
        if not collection:
            return False
        
        try:
            self.db.delete(collection)
            self.db.commit()
            
            logger.info(f"Collection deleted: {collection_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting collection {collection_id}: {e}")
            raise
    
    @staticmethod
    def detect_file_type(filename: str) -> Optional[DocumentType]:
        """Detect document type from filename"""
        suffix = Path(filename).suffix.lower()
        
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.xlsx': DocumentType.XLSX,
            '.pptx': DocumentType.PPTX,
            '.csv': DocumentType.CSV,
        }
        
        return type_mapping.get(suffix)