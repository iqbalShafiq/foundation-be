import uuid
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models import Message, MessageEditRequest, MessageEditResponse, MessageBranch, ConversationBranchesResponse, ModelType, Conversation
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService


class MessageService:
    def __init__(self, db: Session):
        self.db = db
    
    def edit_message(self, message_id: int, user_id: int, edit_request: MessageEditRequest) -> Optional[MessageEditResponse]:
        """
        Edit a message by creating a new conversation (branch) with edited content
        """
        # Get the original message
        original_message = self.db.query(Message).filter(
            Message.id == message_id
        ).first()
        
        if not original_message:
            return None
            
        # Verify message belongs to user's conversation
        conversation_service = ConversationService(self.db)
        original_conversation = conversation_service.get_conversation_by_id(original_message.conversation_id, user_id)
        if not original_conversation:
            return None
            
        # Only allow editing user messages
        if original_message.role != "user":
            raise ValueError("Only user messages can be edited")
            
        # Create new conversation as a branch
        new_conversation_id = str(uuid.uuid4())
        
        # Create title with (Edited) suffix only if not already present
        original_title = original_conversation.title
        if not original_title.endswith(" (Edited)"):
            new_title = f"{original_title} (Edited)"
        else:
            new_title = original_title
        
        new_conversation = Conversation(
            id=new_conversation_id,
            user_id=user_id,
            title=new_title,
            model_type=original_conversation.model_type,
            parent_conversation_id=original_conversation.id,
            edited_message_id=message_id
        )
        
        self.db.add(new_conversation)
        self.db.flush()
        
        # Copy all messages from original conversation up to the edited message
        messages_to_copy = self.db.query(Message).filter(
            and_(
                Message.conversation_id == original_message.conversation_id,
                Message.id < message_id
            )
        ).order_by(Message.id).all()
        
        # Copy previous messages to new conversation
        for msg in messages_to_copy:
            copied_message = Message(
                conversation_id=new_conversation_id,
                role=msg.role,
                content=msg.content,
                image_urls=msg.image_urls,
                document_context=msg.document_context,
                is_active_branch=True
            )
            self.db.add(copied_message)
        
        # Add the edited message to new conversation
        edited_message = Message(
            conversation_id=new_conversation_id,
            role=original_message.role,
            content=edit_request.content,
            image_urls=original_message.image_urls,
            document_context=original_message.document_context,
            is_active_branch=True
        )
        
        self.db.add(edited_message)
        self.db.flush()  # Get the ID
        
        # Regenerate AI response for the new conversation
        regenerated_messages = self._regenerate_ai_responses_for_new_conversation(
            new_conversation_id, user_id, original_conversation.model_type
        )
        
        self.db.commit()
        
        return MessageEditResponse(
            message_id=edited_message.id,
            new_branch_id=new_conversation_id,  # Now it's a conversation ID
            conversation_id=new_conversation_id,
            regenerated_messages=regenerated_messages
        )
    
    def get_conversation_branches(self, conversation_id: str, user_id: int) -> Optional[ConversationBranchesResponse]:
        """
        Get all related conversations (branches) for a conversation
        """
        # Verify conversation belongs to user
        conversation_service = ConversationService(self.db)
        conversation = conversation_service.get_conversation_by_id(conversation_id, user_id)
        if not conversation:
            return None
            
        # Find the root conversation (if current is a branch)
        root_conversation_id = conversation_id
        if conversation.parent_conversation_id:
            root_conversation_id = conversation.parent_conversation_id
        
        # Get all conversations that are branches of the root (including root itself)
        related_conversations = self.db.query(Conversation).filter(
            and_(
                Conversation.user_id == user_id,
                or_(
                    Conversation.id == root_conversation_id,
                    Conversation.parent_conversation_id == root_conversation_id
                )
            )
        ).all()
        
        branches = []
        for conv in related_conversations:
            # Get message count for this conversation
            message_count = self.db.query(Message).filter(
                Message.conversation_id == conv.id
            ).count()
            
            # Determine if this is the currently active conversation
            is_active = conv.id == conversation_id
            
            branches.append(MessageBranch(
                branch_id=conv.id,  # Use conversation ID as branch ID
                root_message_id=conv.edited_message_id or 0,  # Original message that was edited
                is_active=is_active,
                created_at=conv.created_at.isoformat(),
                message_count=message_count
            ))
        
        return ConversationBranchesResponse(
            conversation_id=conversation_id,
            branches=branches
        )
    
    def switch_active_branch(self, conversation_id: str, branch_id: str, user_id: int) -> bool:
        """
        Verify that the branch (conversation) exists and belongs to user
        In the new approach, switching branch means navigating to different conversation
        """
        # Verify both conversations belong to user
        conversation_service = ConversationService(self.db)
        
        original_conversation = conversation_service.get_conversation_by_id(conversation_id, user_id)
        target_conversation = conversation_service.get_conversation_by_id(branch_id, user_id)
        
        if not original_conversation or not target_conversation:
            return False
            
        # Verify they are related (same root or parent-child relationship)
        original_root = original_conversation.parent_conversation_id or conversation_id
        target_root = target_conversation.parent_conversation_id or branch_id
        
        return original_root == target_root or original_conversation.id == target_conversation.parent_conversation_id or target_conversation.id == original_conversation.parent_conversation_id
    
    def _deactivate_branch_from_message(self, message: Message):
        """
        Deactivate all messages in the branch from the given message onwards
        """
        # Find all messages that come after this message in the conversation
        messages_to_deactivate = self.db.query(Message).filter(
            and_(
                Message.conversation_id == message.conversation_id,
                Message.id >= message.id,
                Message.is_active_branch.is_(True)
            )
        ).all()
        
        for msg in messages_to_deactivate:
            msg.is_active_branch = False
    
    def _regenerate_ai_responses_for_new_conversation(self, conversation_id: str, user_id: int, model_type: str) -> List[dict]:
        """
        Generate AI response for the new conversation after message edit
        """
        conversation_service = ConversationService(self.db)
        
        # Get all messages in the new conversation
        all_messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.id).all()
        
        # Build conversation history for AI
        messages = []
        for msg in all_messages:
            messages.append({"role": msg.role, "content": msg.content})
        
        try:
            # Initialize chat service
            chat_service = ChatService()
            
            # Generate AI response
            response = chat_service.get_chat_response_sync(
                messages=messages,
                model_type=ModelType(model_type)
            )
            
            # Create AI response message in new conversation
            ai_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=response,
                is_active_branch=True
            )
            
            self.db.add(ai_message)
            self.db.flush()
            
            # Convert to response format
            return [conversation_service._message_to_response(ai_message)]
            
        except Exception as e:
            # If AI generation fails, return empty list
            print(f"Error regenerating AI response: {e}")
            return []