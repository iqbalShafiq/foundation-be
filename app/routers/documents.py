from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional, List, cast
import logging
import os

from app.models import (
    User, DocumentResponse, DocumentCollectionResponse, DocumentCollectionCreate, DocumentCollectionUpdate,
    DocumentSearchRequest, DocumentSearchResult, DocumentStatus, DocumentType
)
from app.services.document_service import DocumentService
from app.services.rag_service import RAGService
from app.dependencies import require_user_or_admin
from app.database import get_db

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


def process_document_background(doc_id: str):
    """Background task to process document"""
    from app.database import SessionLocal
    
    # Create dedicated DB session for background task
    db = SessionLocal()
    try:
        document_service = DocumentService(db)
        
        logger.info(f"Starting background processing for document: {doc_id}")
        success = document_service.process_document(doc_id)
        
        if success:
            logger.info(f"Document processed successfully: {doc_id}")
        else:
            logger.error(f"Document processing failed: {doc_id}")
            
    except Exception as e:
        logger.error(f"Background processing error for document {doc_id}: {e}")
    finally:
        db.close()


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Upload a document for processing"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Detect file type
    file_type = DocumentService.detect_file_type(file.filename)
    if not file_type:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported types: {DocumentService.detect_file_type.__doc__}"
        )
    
    # Check file size (10MB limit)
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    try:
        document_service = DocumentService(db)
        
        # Upload document immediately (fast)
        doc_id = document_service.upload_document(
            file_content=file_content,
            filename=file.filename,
            user_id=cast(int, current_user.id),
            file_type=file_type
        )
        
        # Process document in background (slow)
        background_tasks.add_task(
            process_document_background,
            doc_id
        )
        
        # Get document info (status will be "pending")
        document = document_service.get_document(doc_id, cast(int, current_user.id))
        
        if not document:
            raise HTTPException(status_code=500, detail="Document upload failed")
        
        return document
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("", response_model=List[DocumentResponse])
async def get_documents(
    status: Optional[DocumentStatus] = None,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Get all documents for the current user"""
    document_service = DocumentService(db)
    return document_service.get_user_documents(
        user_id=cast(int, current_user.id),
        status_filter=status
    )


# Document Collections endpoints

@router.post("/collections", response_model=DocumentCollectionResponse)
async def create_collection(
    collection_data: DocumentCollectionCreate,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Create a new document collection"""
    document_service = DocumentService(db)
    
    try:
        collection_id = document_service.create_collection(
            user_id=cast(int, current_user.id),
            name=collection_data.name,
            description=collection_data.description,
            document_ids=collection_data.document_ids
        )
        
        # Get created collection
        collections = document_service.get_user_collections(cast(int, current_user.id))
        created_collection = next((c for c in collections if c.id == collection_id), None)
        
        if not created_collection:
            raise HTTPException(status_code=500, detail="Failed to retrieve created collection")
        
        return created_collection
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(status_code=500, detail=f"Collection creation failed: {str(e)}")


@router.get("/collections", response_model=List[DocumentCollectionResponse])
async def get_collections(
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Get all document collections for the current user"""
    document_service = DocumentService(db)
    return document_service.get_user_collections(cast(int, current_user.id))


@router.get("/collections/{collection_id}", response_model=DocumentCollectionResponse)
async def get_collection(
    collection_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Get specific collection by ID"""
    document_service = DocumentService(db)
    collections = document_service.get_user_collections(cast(int, current_user.id))
    
    collection = next((c for c in collections if c.id == collection_id), None)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    return collection


@router.put("/collections/{collection_id}", response_model=DocumentCollectionResponse)
async def update_collection(
    collection_id: str,
    update_data: DocumentCollectionUpdate,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Update document collection"""
    document_service = DocumentService(db)
    
    try:
        success = document_service.update_collection(
            collection_id=collection_id,
            user_id=cast(int, current_user.id),
            name=update_data.name,
            description=update_data.description,
            document_ids=update_data.document_ids
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Get updated collection
        collections = document_service.get_user_collections(cast(int, current_user.id))
        updated_collection = next((c for c in collections if c.id == collection_id), None)
        
        if not updated_collection:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated collection")
        
        return updated_collection
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating collection {collection_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Collection update failed: {str(e)}")


@router.delete("/collections/{collection_id}")
async def delete_collection(
    collection_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Delete a document collection"""
    document_service = DocumentService(db)
    success = document_service.delete_collection(collection_id, cast(int, current_user.id))
    
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    return {"message": "Collection deleted successfully"}


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Get specific document by ID with processing status"""
    document_service = DocumentService(db)
    document = document_service.get_document(document_id, cast(int, current_user.id))
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Download/serve the original document file"""
    from app.models import Document
    
    # Check if document exists and belongs to user
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if file exists on disk
    if not os.path.exists(document.file_path):
        raise HTTPException(status_code=404, detail="Document file not found on server")
    
    # Return file response
    return FileResponse(
        path=document.file_path,
        filename=document.original_filename,
        media_type='application/octet-stream'
    )


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Get document processing status"""
    from app.models import Document
    
    # Query document processing status
    raw_document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not raw_document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": document_id,
        "status": raw_document.processing_status,
        "chunk_count": raw_document.chunk_count,
        "error_message": raw_document.error_message,
        "ready_for_search": raw_document.processing_status == "completed"
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Delete a document and all associated data"""
    document_service = DocumentService(db)
    success = document_service.delete_document(document_id, cast(int, current_user.id))
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: str,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Reprocess a document (useful if processing failed)"""
    document_service = DocumentService(db)
    
    # Check if document exists and belongs to user
    document = document_service.get_document(document_id, cast(int, current_user.id))
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        success = document_service.process_document(document_id)
        
        # Get updated document info
        updated_document = document_service.get_document(document_id, cast(int, current_user.id))
        
        if not updated_document:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated document")
        
        return updated_document
        
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Reprocessing failed: {str(e)}")


@router.post("/search", response_model=List[DocumentSearchResult])
async def search_documents(
    search_request: DocumentSearchRequest,
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Search across user documents"""
    rag_service = RAGService(db)
    
    try:
        results = rag_service.search_documents(
            query=search_request.query,
            user_id=cast(int, current_user.id),
            document_ids=search_request.document_ids,
            collection_id=search_request.collection_id,
            limit=search_request.limit,
            relevance_threshold=search_request.relevance_threshold
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stats/context")
async def get_context_stats(
    current_user: User = Depends(require_user_or_admin),
    db: Session = Depends(get_db)
):
    """Get context statistics for the user"""
    rag_service = RAGService(db)
    return rag_service.get_context_statistics(cast(int, current_user.id))