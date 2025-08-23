import logging
from typing import List, Optional, AsyncGenerator
from app.models import ContextSource, ModelType, ImageData
from app.services.chat_service import chat_service
from app.services.react_agent_service import react_agent_service

logger = logging.getLogger(__name__)


class ChatRouterService:
    """Service to route chat requests to appropriate handler based on context"""
    
    @staticmethod
    def needs_react_agent(context_sources: Optional[List[ContextSource]]) -> bool:
        """
        Determine if React Agent is needed based on document types
        
        Args:
            context_sources: List of document context sources
            
        Returns:
            True if React Agent (with pandas tools) is needed, False otherwise
        """
        if not context_sources:
            return False
        
        # Data file extensions that benefit from pandas analysis tools
        data_file_extensions = ["csv", "xlsx", "xls"]
        
        has_data_files = any(
            hasattr(source, 'document_name') and 
            source.document_name and
            '.' in source.document_name and
            source.document_name.split('.')[-1].lower() in data_file_extensions
            for source in context_sources
        )
        
        logger.info(f"Document analysis - Has data files: {has_data_files}")
        if has_data_files:
            data_files = [
                source.document_name for source in context_sources 
                if hasattr(source, 'document_name') and 
                source.document_name and
                '.' in source.document_name and
                source.document_name.split('.')[-1].lower() in data_file_extensions
            ]
            logger.info(f"Data files detected: {data_files}")
        
        return has_data_files
    
    @staticmethod  
    async def route_chat_request(
        message: str, 
        model_type: ModelType = ModelType.STANDARD,
        conversation_id: str | None = None,
        user_id: int | None = None,
        images: Optional[List[ImageData]] = None,
        context_sources: Optional[List[ContextSource]] = None,
        collection_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Route chat request to appropriate service based on context
        
        Args:
            message: User message
            model_type: Model type to use
            conversation_id: Conversation ID
            user_id: User ID  
            images: List of images
            context_sources: Document context sources
            collection_id: Collection ID
            
        Yields:
            Streaming response chunks
        """
        
        # Determine which service to use
        use_react_agent = ChatRouterService.needs_react_agent(context_sources)
        
        logger.info(f"Routing decision - Use React Agent: {use_react_agent}")
        
        if use_react_agent:
            logger.info("Using React Agent for data analysis")
            # Use React Agent for CSV/Excel analysis
            async for chunk in react_agent_service.generate_stream_response(
                message=message,
                model_type=model_type,
                conversation_id=conversation_id,
                user_id=user_id,
                images=images,
                context_sources=context_sources,
                collection_id=collection_id
            ):
                yield chunk
        else:
            logger.info("Using ChatOpenAI for normal conversation")
            # Use simple ChatOpenAI for normal chat or document Q&A
            async for chunk in chat_service.generate_stream_response(
                message=message,
                model_type=model_type, 
                conversation_id=conversation_id,
                user_id=user_id,
                images=images,
                context_sources=context_sources,
                collection_id=collection_id
            ):
                yield chunk


# Create singleton instance
chat_router_service = ChatRouterService()