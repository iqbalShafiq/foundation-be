import json
import uuid
import logging
from typing import AsyncGenerator, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug, set_verbose
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
from app.services.react_agent_context import set_current_context

logger = logging.getLogger(__name__)

set_debug(True)
set_verbose(True)


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
            verbose=True,
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

        # if context_parts:
        #     context_prompt = "Based on the following documents:\n\n" + "".join(
        #         context_parts
        #     )
        #     context_prompt += "Please answer the user's question using the information from these documents. If the documents don't contain relevant information, please say so."
        #     return context_prompt

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
            - Load and analyze CSV/Excel files from the user's uploaded documents
            - Filter by document IDs selected by the user in the conversation context
            - Perform statistical operations using pandas and LangChain agents
            - Answer complex questions about data content using natural language

            2. generate_chart: Use this tool to create interactive visualizations. This tool can:
            - Generate various chart types (bar, line, scatter, pie, histogram, box plots)
            - Return charts as JSON data for interactive display
            - Customize chart appearance and labels

            When users ask questions about data in uploaded CSV/Excel files, use analyze_dataframe. 
            This tool will automatically access the files based on the document context.
            For visualizations, use generate_chart after getting analysis results.

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

        # Get tools - we'll enhance analyze_dataframe with context
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
        selected_document_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Set context for tools to access
            set_current_context(
                user_id=user_id or 0, selected_document_ids=selected_document_ids
            )

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

            # Stream response from LangGraph agent using astream_events for real-time streaming
            async for event in agent.astream_events(agent_input, version="v2"):
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                event_data = event.get("data", {})

                logger.debug(f"LangGraph event: {event_type} - {event_name}")

                # Handle tool execution completion
                if event_type == "on_tool_end":
                    tool_output = event_data.get("output", "")
                    tool_name = event_name

                    # Convert tool output to string safely
                    tool_output_str = ""
                    try:
                        if isinstance(tool_output, str):
                            tool_output_str = tool_output
                        elif hasattr(tool_output, "content"):
                            # Tool output has content attribute - this is likely the case
                            tool_output_str = tool_output.content
                        else:
                            tool_output_str = str(tool_output)
                    except Exception as e:
                        logger.error(f"Error converting tool output to string: {e}")
                        tool_output_str = "Tool output could not be serialized"

                    accumulated_thinking += f"{tool_name}: {tool_output_str}\n"

                    if tool_name == "generate_chart" and tool_output_str:
                        try:
                            # Parse chart data from string output
                            if (
                                isinstance(tool_output_str, str)
                                and tool_output_str.strip()
                            ):
                                chart_data_collected = json.loads(
                                    tool_output_str.strip()
                                )
                            else:
                                chart_data_collected = tool_output_str

                            # Send chart data immediately after tool completion
                            try:
                                chart_json_str = json.dumps(
                                    {
                                        "type": "chart",
                                        "content": chart_data_collected,
                                        "done": False,
                                        "conversation_id": conversation_id,
                                    }
                                )
                                yield f"data: {chart_json_str}\n\n"
                                chart_data_sent = True
                            except (TypeError, ValueError) as e:
                                logger.error(f"Error serializing chart data: {e}")
                                yield f"data: {json.dumps({'type': 'thinking', 'content': 'Chart data generated but could not be serialized for display', 'done': False, 'conversation_id': conversation_id})}\n\n"

                            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Chart generated successfully!', 'done': False, 'conversation_id': conversation_id})}\n\n"
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing chart JSON data: {e}")
                            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Chart generation completed', 'done': False, 'conversation_id': conversation_id})}\n\n"
                        except Exception as e:
                            logger.error(f"Error processing chart data: {e}")
                            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Chart generation completed', 'done': False, 'conversation_id': conversation_id})}\n\n"
                    else:
                        # Other tool outputs
                        yield f"data: {json.dumps({'type': 'thinking', 'content': f'{tool_name}: {tool_output_str[:200]}...', 'done': False, 'conversation_id': conversation_id})}\n\n"

                # Handle LLM streaming tokens
                elif event_type == "on_chat_model_stream":
                    chunk_data = event_data.get("chunk", {})
                    # Check if chunk_data has content - could be dict or object
                    content = ""
                    if hasattr(chunk_data, "content"):
                        content = getattr(chunk_data, "content", "")
                    elif isinstance(chunk_data, dict) and "content" in chunk_data:
                        content = chunk_data["content"]

                    if content:
                        accumulated_answer += content
                        final_answer_for_storage += content

                        # Send chart data first if we have it and haven't sent it
                        if chart_data_collected and not chart_data_sent:
                            try:
                                chart_json_str = json.dumps(
                                    {
                                        "type": "chart",
                                        "content": chart_data_collected,
                                        "done": False,
                                        "conversation_id": conversation_id,
                                    }
                                )
                                yield f"data: {chart_json_str}\n\n"
                                chart_data_sent = True
                            except (TypeError, ValueError) as e:
                                logger.error(f"Error serializing chart data: {e}")
                                # Send a simplified chart notification instead
                                yield f"data: {json.dumps({'type': 'thinking', 'content': 'Chart data generated but could not be serialized', 'done': False, 'conversation_id': conversation_id})}\n\n"
                                chart_data_sent = True

                        yield f"data: {json.dumps({'type': 'answer', 'content': content, 'done': False, 'conversation_id': conversation_id})}\n\n"

                # Handle agent completion to capture any final content
                elif event_type == "on_chain_end" and event_name == "agent":
                    output_data = event_data.get("output", {})
                    if "messages" in output_data:
                        last_message = (
                            output_data["messages"][-1]
                            if output_data["messages"]
                            else None
                        )
                        if last_message and hasattr(last_message, "content"):
                            # If we haven't captured any answer content yet, use the final message
                            if not final_answer_for_storage:
                                final_answer_for_storage = last_message.content
                                yield f"data: {json.dumps({'type': 'answer', 'content': last_message.content, 'done': False, 'conversation_id': conversation_id})}\n\n"

            # Send any remaining chart data that hasn't been sent yet
            if chart_data_collected and not chart_data_sent:
                try:
                    chart_json_str = json.dumps(
                        {
                            "type": "chart",
                            "content": chart_data_collected,
                            "done": False,
                            "conversation_id": conversation_id,
                        }
                    )
                    yield f"data: {chart_json_str}\n\n"
                    chart_data_sent = True
                except (TypeError, ValueError) as e:
                    logger.error(f"Error serializing chart data at completion: {e}")
                    yield f"data: {json.dumps({'type': 'thinking', 'content': 'Chart data generated but could not be serialized', 'done': False, 'conversation_id': conversation_id})}\n\n"

            # Handle final completion
            if not final_answer_for_storage:
                if chart_data_collected:
                    final_answer_for_storage = (
                        "I've generated the requested chart for you."
                    )
                elif accumulated_thinking:
                    final_answer_for_storage = accumulated_thinking
                else:
                    # This shouldn't happen with proper LangGraph usage
                    logger.warning(
                        "No content captured from LangGraph agent - this may indicate a structure issue"
                    )
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
                    try:
                        # Test if chart data can be serialized
                        json.dumps(chart_data_collected)
                        combined_context_info["chart_data"] = chart_data_collected
                    except (TypeError, ValueError) as e:
                        logger.error(
                            f"Chart data cannot be serialized for database storage: {e}"
                        )
                        combined_context_info["chart_data"] = {
                            "error": "Chart data could not be serialized"
                        }

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
