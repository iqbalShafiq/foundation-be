from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from typing import List, Optional
from app.models import Conversation, Message, ConversationResponse, MessageResponse


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
        # Get message count for this conversation
        message_count = (
            self.db.query(func.count(Message.id))
            .filter(Message.conversation_id == conversation.id)
            .scalar()
        )
        
        base_data = {
            "id": conversation.id,
            "title": conversation.title,
            "model_type": conversation.model_type,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "message_count": message_count,
        }
        
        if include_related:
            related_chats = []
            if related_messages:
                related_chats = [
                    MessageResponse.model_validate({
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat(),
                    })
                    for msg in related_messages
                ]
            base_data["related_chats"] = related_chats
        
        return ConversationResponse.model_validate(base_data)

    def get_conversation_detail(self, conversation_id: str, user_id: int):
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
            
        # Get all messages for this conversation
        messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        
        return conversation, messages