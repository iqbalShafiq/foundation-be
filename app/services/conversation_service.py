from sqlalchemy.orm import Session
from sqlalchemy import or_, func, and_
from typing import List, Optional
import json
from app.models import Conversation, Message, ConversationResponse, MessageResponse, MessageDocumentContext, DocumentContextInfo


class ConversationService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_conversations(
        self, user_id: int, keyword: Optional[str] = None, page: int = 1, limit: int = 20
    ) -> dict:
        """Get all conversations for a user, optionally with keyword search and related chats"""
        base_query = self.db.query(Conversation).filter(Conversation.user_id == user_id)

        # Calculate pagination
        offset = (page - 1) * limit
        total_count = base_query.count()
        total_pages = (total_count + limit - 1) // limit

        if keyword:
            conversations = self._get_conversations_with_related_chats(base_query, keyword, offset, limit)
        else:
            conversations = self._get_conversations_basic(base_query, offset, limit)

        return {
            "data": conversations,
            "pagination": {
                "page": page,
                "limit": limit,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }

    def _get_conversations_basic(self, base_query, offset: int, limit: int) -> List[ConversationResponse]:
        """Get conversations without related chats"""
        conversations = base_query.order_by(Conversation.updated_at.desc()).offset(offset).limit(limit).all()
        
        return [
            self._conversation_to_response(conv, include_related=False)
            for conv in conversations
        ]

    def _get_conversations_with_related_chats(
        self, base_query, keyword: str, offset: int, limit: int
    ) -> List[ConversationResponse]:
        """Get conversations with related chats based on keyword search"""
        keyword_filter = f"%{keyword}%"
        
        # Find conversations that match the keyword
        conversations = self._search_conversations_by_keyword(base_query, keyword_filter, offset, limit)
        
        
        # Build result with related chats
        result = []
        for conv in conversations:
            related_messages = self._get_related_messages(conv.id, keyword_filter)
            result.append(
                self._conversation_to_response(
                    conv, include_related=True, related_messages=related_messages
                )
            )
        
        return result

    def _search_conversations_by_keyword(self, base_query, keyword_filter: str, offset: int, limit: int):
        """Search conversations by keyword in title or message content"""
        return (
            base_query.filter(
                or_(
                    Conversation.title.ilike(keyword_filter),
                    Conversation.id.in_(
                        self.db.query(Message.conversation_id)
                        .filter(Message.content.ilike(keyword_filter))
                        .distinct()
                    ),
                )
            )
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def _get_related_messages(
        self, conversation_id: str, keyword_filter: str
    ) -> List[Message]:
        """Get messages from this conversation that match the keyword, limited to 5"""
        return (
            self.db.query(Message)
            .filter(
                Message.conversation_id == conversation_id,
                Message.content.ilike(keyword_filter)
            )
            .order_by(Message.created_at.desc())
            .limit(5)
            .all()
        )

    def _conversation_to_response(
        self, 
        conversation: Conversation, 
        include_related: bool = False,
        related_messages: Optional[List[Message]] = None
    ) -> ConversationResponse:
        """Convert Conversation model to ConversationResponse"""
        # Get message count for this conversation (active branch only)
        message_count = (
            self.db.query(func.count(Message.id))
            .filter(
                and_(
                    Message.conversation_id == conversation.id,
                    or_(
                        Message.is_active_branch.is_(True),
                        Message.is_active_branch.is_(None)  # For backward compatibility
                    )
                )
            )
            .scalar()
        )
        
        base_data = {
            "id": conversation.id,
            "title": conversation.title,
            "model_type": conversation.model_type,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "message_count": message_count,
            "parent_conversation_id": getattr(conversation, 'parent_conversation_id', None),
            "edited_message_id": getattr(conversation, 'edited_message_id', None),
            "is_branch": bool(getattr(conversation, 'parent_conversation_id', None)),
        }
        
        if include_related:
            related_chats = []
            if related_messages:
                related_chats = [
                    self._message_to_response(msg)
                    for msg in related_messages
                ]
            base_data["related_chats"] = related_chats
        
        return ConversationResponse.model_validate(base_data)

    def _message_to_response(self, message: Message) -> MessageResponse:
        """Convert Message model to MessageResponse with document context and chart data"""
        # Parse document context if exists and field is available
        document_context = None
        chart_data = None
        if hasattr(message, 'document_context') and message.document_context:
            try:
                context_data = json.loads(message.document_context)
                if context_data:
                    # Handle regular document context
                    documents = [
                        DocumentContextInfo.model_validate(doc)
                        for doc in context_data.get("documents", [])
                    ]
                    if documents:  # Only create document_context if there are documents
                        document_context = MessageDocumentContext(
                            collection_id=context_data.get("collection_id"),
                            documents=documents,
                            context_chunks_count=context_data.get("context_chunks_count", 0)
                        )
                    
                    # Handle chart data
                    if "chart_data" in context_data:
                        chart_data = context_data["chart_data"]
                        
            except (json.JSONDecodeError, ValueError, Exception):
                # If JSON parsing fails or any other error, ignore document context
                document_context = None
                chart_data = None
        
        # Parse image URLs if exists
        image_urls = None
        if hasattr(message, 'image_urls') and message.image_urls:
            try:
                image_urls = json.loads(message.image_urls)
            except (json.JSONDecodeError, ValueError):
                image_urls = None
        
        # Check if this message has branches (alternative versions)
        has_branches = False
        if hasattr(message, 'parent_message_id') and message.parent_message_id:
            # Check if there are other messages with the same parent but different branch_id
            sibling_count = self.db.query(func.count(Message.id)).filter(
                and_(
                    Message.parent_message_id == message.parent_message_id,
                    Message.branch_id != message.branch_id
                )
            ).scalar()
            has_branches = sibling_count > 0
        
        return MessageResponse(
            id=message.id,
            role=message.role,
            content=message.content,
            image_urls=image_urls,
            document_context=document_context,
            chart_data=chart_data,
            # Branching fields
            parent_message_id=getattr(message, 'parent_message_id', None),
            branch_id=getattr(message, 'branch_id', None),
            is_active_branch=getattr(message, 'is_active_branch', True),
            has_branches=has_branches,
            # Token usage fields
            input_tokens=getattr(message, 'input_tokens', None),
            output_tokens=getattr(message, 'output_tokens', None),
            total_tokens=getattr(message, 'total_tokens', None),
            model_cost=getattr(message, 'model_cost', None),
            created_at=message.created_at.isoformat()
        )

    def get_conversation_detail(self, conversation_id: str, user_id: int, active_branch_only: bool = True):
        """Get detailed conversation with full message history"""
        conversation = (
            self.db.query(Conversation)
            .filter(
                Conversation.id == conversation_id, 
                Conversation.user_id == user_id
            )
            .first()
        )
        
        if not conversation:
            return None
            
        # Get messages for this conversation
        query = self.db.query(Message).filter(Message.conversation_id == conversation_id)
        
        if active_branch_only:
            # Only get active branch messages
            query = query.filter(
                or_(
                    Message.is_active_branch.is_(True),
                    Message.is_active_branch.is_(None)  # For backward compatibility
                )
            )
        
        messages = query.order_by(Message.created_at.asc()).all()
        
        return conversation, messages

    def get_conversation_by_id(self, conversation_id: str, user_id: int) -> Optional[Conversation]:
        """Get conversation by ID if it belongs to the user"""
        return (
            self.db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
            .first()
        )

    def delete_conversation(self, conversation_id: str, user_id: int) -> bool:
        """Delete a conversation and all its messages"""
        conversation = (
            self.db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
            .first()
        )
        
        if not conversation:
            return False
        
        # Delete all messages first (due to foreign key constraint)
        self.db.query(Message).filter(Message.conversation_id == conversation_id).delete()
        
        # Delete the conversation
        self.db.delete(conversation)
        self.db.commit()
        
        return True