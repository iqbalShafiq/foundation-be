import json
import uuid
import logging
from typing import AsyncGenerator, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
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
from app.services.data_analysis_service import DataAnalysisService

logger = logging.getLogger(__name__)


class ReactAgentService:
    def __init__(self):
        self._agents = {}
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}

    MODEL_MAPPING = {
        ModelType.FAST: "gpt-4.1-mini",
        ModelType.STANDARD: "gpt-4.1",
        ModelType.FAST_REASONING: "o4-mini",
        ModelType.REASONING: "o3",
    }

    def get_llm(self, model_type: ModelType) -> ChatOpenAI:
        """Get ChatOpenAI instance for the specified model"""
        actual_model = self.MODEL_MAPPING[model_type]
        return ChatOpenAI(
            model=actual_model,
            temperature=0.7,
            streaming=True,
        )

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
                f"[REACT AGENT DEBUG] Loading {len(messages)} messages from database for conversation {conversation_id}"
            )

            # Add messages to memory in chronological order
            for msg in messages:
                if str(msg.role) == "user":
                    memory.chat_memory.add_user_message(str(msg.content))
                elif str(msg.role) == "assistant":
                    memory.chat_memory.add_ai_message(str(msg.content))

            db.close()

        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            # Don't fail if we can't load history - just continue with empty memory

    def _generate_conversation_title(self, message: str) -> str:
        """Generate a title from the first message"""
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

    def _build_agent_system_prompt(
        self, context_sources: List[ContextSource], user_preferences: str
    ) -> str:
        """Build comprehensive system prompt for React Agent"""
        base_prompt = user_preferences or "You are a helpful AI assistant."

        # Add document context if available
        context_prompt = self._build_context_prompt(context_sources)

        # Add data analysis capabilities description
        tools_description = """
You have access to powerful data analysis tools:

1. analyze_dataframe: Use this tool when users ask questions about CSV or Excel data. This tool can:
   - Load and analyze spreadsheet data
   - Perform statistical operations
   - Filter and group data
   - Answer questions about data content and patterns

2. generate_chart: Use this tool to create interactive visualizations. This tool can:
   - Generate various chart types (bar, line, scatter, pie, histogram, box plots)
   - Return charts as JSON data for interactive display
   - Customize chart appearance and labels

When users ask questions about data in uploaded CSV/Excel files, or request visualizations, use these tools appropriately. 

For CSV/Excel analysis questions, first use analyze_dataframe to understand the data, then optionally use generate_chart if visualization would be helpful.

For normal conversations without data analysis needs, respond directly without using tools.

Always provide clear and helpful responses. When using tools, explain what you're doing and interpret the results for the user.
"""

        # Combine all parts
        full_prompt = base_prompt

        if context_prompt:
            full_prompt += f"\n\n{context_prompt}"

        full_prompt += f"\n\n{tools_description}"

        return full_prompt

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

    def create_react_agent(
        self,
        model_type: ModelType,
        context_sources: List[ContextSource],
        user_preferences: str,
    ):
        """Create React Agent with tools and context using LangGraph"""
        # Get LLM
        llm = self.get_llm(model_type)

        # Get tools
        tools = DataAnalysisService.get_analysis_tools()

        # Build system prompt with context
        system_prompt = self._build_agent_system_prompt(
            context_sources, user_preferences
        )

        # Create React Agent using LangGraph's create_react_agent
        # This automatically handles the workflow and streaming
        agent_graph = create_react_agent(model=llm, tools=tools, prompt=system_prompt)

        return agent_graph

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
        try:
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Memory
            memory = self.get_memory(conversation_id)

            # User prefs
            user_preferences = ""
            if user_id:
                user_preferences = self._get_user_system_prompt(user_id)

            # Agent
            agent = self.create_react_agent(
                model_type=model_type,
                context_sources=context_sources or [],
                user_preferences=user_preferences,
            )

            # Build context info
            document_context_info = None
            if context_sources:
                document_context_info = self._build_document_context_info(
                    context_sources, collection_id
                )

            # Prepare conversation history for the agent
            chat_history = []
            for msg in memory.chat_memory.messages or []:
                if getattr(msg, "type", "") == "human":
                    chat_history.append({"role": "user", "content": msg.content})
                elif getattr(msg, "type", "") == "ai":
                    chat_history.append({"role": "assistant", "content": msg.content})

            # Input for LangGraph agent
            agent_input = {
                "messages": chat_history + [{"role": "user", "content": message}]
            }

            final_answer_for_storage = ""
            accumulated_thinking = ""
            accumulated_answer = ""
            chart_data_collected = None
            chart_data_sent = False

            # Stream response from LangGraph agent
            async for chunk in agent.astream(agent_input):
                logger.info(f"LangGraph chunk keys: {list(chunk.keys()) if isinstance(chunk, dict) else type(chunk)}")
                
                # LangGraph chunks have different structures depending on the node
                # Look for the final messages in the chunk
                node_messages = []
                
                # Different ways LangGraph can structure chunks:
                for key in chunk.keys() if isinstance(chunk, dict) else []:
                    if isinstance(chunk[key], dict) and "messages" in chunk[key]:
                        node_messages.extend(chunk[key]["messages"])
                    elif key == "messages":
                        node_messages.extend(chunk[key])
                
                # Process all messages found in this chunk
                for message_chunk in node_messages:
                    if hasattr(message_chunk, "content") and message_chunk.content:
                        content = message_chunk.content
                        
                        # Check message type to determine if it's tool output or AI response
                        message_type = getattr(message_chunk, "type", "")
                        message_name = getattr(message_chunk, "name", "")
                        
                        if message_type == "tool" or message_name:
                            # This is tool output - treat as thinking
                            tool_name = message_name or "tool"
                            accumulated_thinking += f"{tool_name}: {content}\n"
                            
                            if tool_name == "generate_chart":
                                try:
                                    chart_data_collected = json.loads(content) if isinstance(content, str) else content
                                except Exception as e:
                                    logger.error(f"Error parsing chart data: {e}")
                            
                            yield f"data: {json.dumps({'type': 'thinking', 'content': f'{tool_name}: {content[:200]}...', 'done': False, 'conversation_id': conversation_id})}\n\n"
                        
                        elif message_type == "ai" or message_type == "assistant" or not message_type:
                            # This is AI response - treat as final answer
                            accumulated_answer += content
                            final_answer_for_storage += content
                            
                            # Send chart data first if we have it and haven't sent it
                            if chart_data_collected and not chart_data_sent:
                                yield f"data: {json.dumps({'type': 'chart', 'content': chart_data_collected, 'done': False, 'conversation_id': conversation_id})}\n\n"
                                chart_data_sent = True
                            
                            yield f"data: {json.dumps({'type': 'answer', 'content': content, 'done': False, 'conversation_id': conversation_id})}\n\n"

            # Handle final completion  
            if not final_answer_for_storage:
                if chart_data_collected:
                    final_answer_for_storage = "I've generated the requested chart for you."
                elif accumulated_thinking:
                    final_answer_for_storage = accumulated_thinking
                else:
                    # This shouldn't happen with proper LangGraph usage
                    logger.warning("No content captured from LangGraph agent - this may indicate a structure issue")
                    final_answer_for_storage = "I apologize, but I wasn't able to generate a proper response. Please try again."

            # Clean up the final answer
            final_answer_for_storage = final_answer_for_storage.strip()

            # Send completion signal
            yield f"data: {json.dumps({'type': 'answer', 'content': '', 'done': True, 'conversation_id': conversation_id})}\n\n"

            # --- SAVE memory & DB ---
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(final_answer_for_storage)

            if user_id:
                image_urls = [img.url for img in images if img.url] if images else None

                # Include chart data in document context if available
                combined_context_info = document_context_info or {}
                if chart_data_collected:
                    combined_context_info["chart_data"] = chart_data_collected

                self._save_conversation_and_messages_to_db(
                    conversation_id,
                    user_id,
                    message,
                    final_answer_for_storage,
                    model_type,
                    image_urls,
                    combined_context_info if combined_context_info else None,
                )

        except Exception as e:
            logger.error(f"Error in React Agent stream response: {e}")
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


# Create a singleton instance
react_agent_service = ReactAgentService()
