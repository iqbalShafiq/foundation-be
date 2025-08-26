import json
import uuid
import logging
from typing import AsyncGenerator, Dict, cast, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from app.models import (
    ModelType,
    Conversation,
    Message,
    UserPreferences,
    ImageData,
    ContextSource,
)
from app.database import get_db

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self._llms = {}
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}

    MODEL_MAPPING = {
        ModelType.FAST: "deepseek/deepseek-chat-v3.1",
        ModelType.STANDARD: "anthropic/claude-sonnet-4",
        ModelType.FAST_REASONING: "openai/o4-mini",
        ModelType.REASONING: "openai/o3",
    }

    def get_llm(self, model_type: ModelType) -> ChatOpenAI:
        """Get or create ChatOpenAI instance for the specified model"""
        if model_type not in self._llms:
            actual_model = self.MODEL_MAPPING[model_type]
            self._llms[model_type] = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                model=actual_model,
                temperature=0.7,
                streaming=True,
            )
        return self._llms[model_type]

    def get_memory(self, conversation_id: str) -> ConversationBufferWindowMemory:
        """Get or create memory for a conversation"""
        if conversation_id not in self._memories:
            memory = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 message pairs (20 total messages)
                return_messages=True,
                memory_key="chat_history",
            )

            # Load existing conversation history from database
            self._load_conversation_history_into_memory(conversation_id, memory)

            self._memories[conversation_id] = memory
        return self._memories[conversation_id]

    def _load_conversation_history_into_memory(
        self, conversation_id: str, memory: ConversationBufferWindowMemory
    ):
        """Load existing conversation history from database into memory"""
        try:
            db = next(get_db())
            messages = (
                db.query(Message)
                .filter(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
                .all()
            )

            print(
                f"[CHATOPENAI DEBUG] Loading {len(messages)} messages from database for conversation {conversation_id}"
            )

            # Add messages to memory in chronological order
            for msg in messages:
                if msg.role == "user":
                    memory.chat_memory.add_user_message(str(msg.content))
                elif msg.role == "assistant":
                    memory.chat_memory.add_ai_message(str(msg.content))

            db.close()

        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            # Don't fail if we can't load history - just continue with empty memory

    def _generate_conversation_title(self, message: str) -> str:
        """Generate a title from the first message"""
        # Take first 50 characters and add ellipsis if longer
        title = message.strip()[:50]
        if len(message.strip()) > 50:
            title += "..."
        return title

    def _get_user_system_prompt(self, user_id: int) -> str:
        """Generate system prompt based on user preferences"""
        db = next(get_db())
        try:
            preferences = (
                db.query(UserPreferences)
                .filter(UserPreferences.user_id == user_id)
                .first()
            )

            if not preferences:
                return "You are a helpful AI assistant."

            system_parts = ["You are a helpful AI assistant."]

            if str(preferences.nickname):
                system_parts.append(
                    f"The user prefers to be called '{preferences.nickname}'."
                )

            if str(preferences.job):
                system_parts.append(f"The user works as a {preferences.job}.")

            if str(preferences.chatbot_preference):
                system_parts.append(
                    f"Additional context: {preferences.chatbot_preference}"
                )

            return " ".join(system_parts)
        except Exception as e:
            logger.error(f"Error fetching user preferences: {e}")
            return "You are a helpful AI assistant."
        finally:
            db.close()

    def _process_images_for_openai(
        self, images: Optional[List[ImageData]]
    ) -> List[Dict[str, Any]]:
        """Convert ImageData objects to OpenAI format"""
        if not images:
            return []

        image_contents = []
        for image in images:
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.mime_type};base64,{image.data}",
                        "detail": "auto",
                    },
                }
            )
        return image_contents

    def _build_context_prompt(
        self, context_sources: List[ContextSource], max_context_length: int = 4000
    ) -> str:
        """Build context prompt from retrieved document sources"""
        if not context_sources:
            return ""

        # Sort by relevance score
        sorted_sources = sorted(
            context_sources, key=lambda x: x.relevance_score, reverse=True
        )

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
            context_prompt = "Based on the following documents:\n\n" + "".join(
                context_parts
            )
            context_prompt += "Please answer the user's question using the information from these documents. If the documents don't contain relevant information, please say so."
            return context_prompt

        return ""

    def _build_document_context_info(
        self, context_sources: List[ContextSource], collection_id: Optional[str] = None
    ) -> dict:
        """Build document context info from context sources for storage"""
        if not context_sources:
            return {}

        # Get unique documents from context sources
        documents_map = {}
        for source in context_sources:
            if source.document_id not in documents_map:
                # Extract file extension from document name
                file_extension = None
                if "." in source.document_name:
                    file_extension = source.document_name.split(".")[-1]

                documents_map[source.document_id] = {
                    "document_id": source.document_id,
                    "title": source.document_name,
                    "url": None,  # Could be enhanced to include document URL if available
                    "file_extension": file_extension,
                }

        return {
            "collection_id": collection_id,
            "documents": list(documents_map.values()),
            "context_chunks_count": len(context_sources),
        }

    def _save_conversation_and_messages_to_db(
        self,
        conversation_id: str,
        user_id: int,
        user_message: str,
        ai_message: str,
        model_type: ModelType,
        image_urls: Optional[List[str]] = None,
        document_context: Optional[dict] = None,
    ):
        """Save conversation and messages to database"""
        db = next(get_db())
        try:
            # Check if conversation already exists
            existing = (
                db.query(Conversation)
                .filter(Conversation.id == conversation_id)
                .first()
            )
            if not existing:
                # Create new conversation with title from first message
                title = self._generate_conversation_title(user_message)
                conversation = Conversation(
                    id=conversation_id,
                    user_id=user_id,
                    title=title,
                    model_type=model_type.value,
                )
                db.add(conversation)
            else:
                # SQLAlchemy will automatically update updated_at due to onupdate=func.now()
                # We just need to mark the object as modified
                db.merge(existing)

            # Save user message with image URLs and document context
            user_msg = Message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                image_urls=json.dumps(image_urls) if image_urls else None,
                document_context=json.dumps(document_context)
                if document_context
                else None,
            )
            db.add(user_msg)

            # Save AI message
            ai_msg = Message(
                conversation_id=conversation_id, role="assistant", content=ai_message
            )
            db.add(ai_msg)

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving conversation and messages: {e}")
        finally:
            db.close()

    async def generate_stream_response(
        self,
        message: str,
        model_type: ModelType = ModelType.STANDARD,
        conversation_id: str | None = None,
        user_id: int | None = None,
        images: Optional[List[ImageData]] = None,
        context_sources: Optional[List[ContextSource]] = None,
        collection_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from ChatOpenAI with conversation memory and document context"""
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Get memory for this conversation
            memory = self.get_memory(conversation_id)

            # Debug logging for ChatOpenAI
            history_messages = (
                memory.chat_memory.messages if memory.chat_memory.messages else []
            )
            print(f"[CHATOPENAI DEBUG] Conversation {conversation_id}")
            print(f"[CHATOPENAI DEBUG] Messages in memory: {len(history_messages)}")
            if history_messages:
                print(
                    f"[CHATOPENAI DEBUG] Recent messages: {[msg.content[:50] for msg in history_messages[-2:]]}"
                )

            # Get conversation history
            chat_history = (
                memory.chat_memory.messages if memory.chat_memory.messages else []
            )

            # Build context-aware user message
            enhanced_message = message
            context_metadata = None
            document_context_info = None

            if context_sources:
                # Build context prompt from retrieved sources
                context_prompt = self._build_context_prompt(context_sources)
                if context_prompt:
                    enhanced_message = f"{context_prompt}\n\nUser Question: {message}"
                    context_metadata = [
                        {
                            "document_id": source.document_id,
                            "document_name": source.document_name,
                            "page_number": source.page_number,
                            "relevance_score": source.relevance_score,
                        }
                        for source in context_sources
                    ]
                    # Build document context info for storage
                    document_context_info = self._build_document_context_info(
                        context_sources, collection_id
                    )

            # Get user system prompt if user_id is provided
            system_prompt = None
            if user_id:
                system_prompt_text = self._get_user_system_prompt(user_id)
                system_prompt = SystemMessage(content=system_prompt_text)

            # Build messages with system prompt + conversation history + new user message
            messages = []
            if system_prompt:
                messages.append(system_prompt)
            messages.extend(chat_history)

            # Create user message content with text and images
            if images:
                user_content: List[Dict[str, Any]] = [
                    {"type": "text", "text": enhanced_message}
                ]
                image_contents = self._process_images_for_openai(images)
                user_content.extend(image_contents)
                messages.append(HumanMessage(content=user_content))
            else:
                messages.append(HumanMessage(content=enhanced_message))

            # Get the appropriate LLM for the model type
            llm = self.get_llm(model_type)

            # Collect AI response content for memory storage
            ai_response_content = ""

            # Stream the response
            async for chunk in llm.astream(messages):
                if chunk.content:
                    ai_response_content += cast(str, chunk.content)
                    # Format as SSE (Server-Sent Events) with context metadata
                    data = {
                        "content": chunk.content,
                        "done": False,
                        "conversation_id": conversation_id,
                    }

                    # Include context sources in the first chunk
                    if context_metadata and ai_response_content == chunk.content:
                        data["context_sources"] = context_metadata

                    yield f"data: {json.dumps(data)}\n\n"

            # Save the conversation to memory
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(ai_response_content)

            # Save to database if user_id is provided
            if user_id:
                # Extract image URLs from images
                image_urls = [img.url for img in images if img.url] if images else None
                self._save_conversation_and_messages_to_db(
                    conversation_id,
                    user_id,
                    message,
                    ai_response_content,
                    model_type,
                    image_urls,
                    document_context_info,
                )

            # Send final message
            final_data = {
                "content": "",
                "done": True,
                "conversation_id": conversation_id,
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            error_data = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_data)}\n\n"


# Create a singleton instance
chat_service = ChatService()
