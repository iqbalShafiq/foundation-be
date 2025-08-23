import json
import uuid
import logging
from typing import AsyncGenerator, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from app.models import ModelType, Conversation, Message, UserPreferences, ImageData, ContextSource
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
                memory_key="chat_history"
            )
            
            # Load existing conversation history from database
            self._load_conversation_history_into_memory(conversation_id, memory)
            
            self._memories[conversation_id] = memory
        return self._memories[conversation_id]

    def _load_conversation_history_into_memory(self, conversation_id: str, memory: ConversationBufferWindowMemory):
        """Load existing conversation history from database into memory"""
        try:
            db = next(get_db())
            messages = db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).all()
            
            print(f"[REACT AGENT DEBUG] Loading {len(messages)} messages from database for conversation {conversation_id}")
            
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
            preferences = db.query(UserPreferences).filter(
                UserPreferences.user_id == user_id
            ).first()
            
            if not preferences:
                return "You are a helpful AI assistant."
            
            system_parts = ["You are a helpful AI assistant."]
            
            if str(preferences.nickname):
                system_parts.append(f"The user prefers to be called '{preferences.nickname}'.")
            
            if str(preferences.job):
                system_parts.append(f"The user works as a {preferences.job}.")
            
            if str(preferences.chatbot_preference):
                system_parts.append(f"Additional context: {preferences.chatbot_preference}")
            
            return " ".join(system_parts)
        except Exception as e:
            logger.error(f"Error fetching user preferences: {e}")
            return "You are a helpful AI assistant."
        finally:
            db.close()

    def _build_context_prompt(self, context_sources: List[ContextSource], max_context_length: int = 4000) -> str:
        """Build context prompt from retrieved document sources"""
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

    def _build_agent_system_prompt(self, context_sources: List[ContextSource], user_preferences: str) -> str:
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
"""
        
        # Combine all parts
        full_prompt = base_prompt
        
        if context_prompt:
            full_prompt += f"\n\n{context_prompt}"
        
        full_prompt += f"\n\n{tools_description}"
        
        return full_prompt

    def _build_document_context_info(self, context_sources: List[ContextSource], collection_id: Optional[str] = None) -> dict:
        """Build document context info from context sources for storage"""
        if not context_sources:
            return {}
        
        # Get unique documents from context sources
        documents_map = {}
        for source in context_sources:
            if source.document_id not in documents_map:
                # Extract file extension from document name
                file_extension = None
                if '.' in source.document_name:
                    file_extension = source.document_name.split('.')[-1]
                
                documents_map[source.document_id] = {
                    "document_id": source.document_id,
                    "title": source.document_name,
                    "url": None,  # Could be enhanced to include document URL if available
                    "file_extension": file_extension
                }
        
        return {
            "collection_id": collection_id,
            "documents": list(documents_map.values()),
            "context_chunks_count": len(context_sources)
        }

    def _save_conversation_and_messages_to_db(self, conversation_id: str, user_id: int, user_message: str, ai_message: str, model_type: ModelType, image_urls: Optional[List[str]] = None, document_context: Optional[dict] = None):
        """Save conversation and messages to database"""
        db = next(get_db())
        try:
            # Check if conversation already exists
            existing = db.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not existing:
                # Create new conversation with title from first message
                title = self._generate_conversation_title(user_message)
                conversation = Conversation(
                    id=conversation_id,
                    user_id=user_id,
                    title=title,
                    model_type=model_type.value
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
                document_context=json.dumps(document_context) if document_context else None
            )
            db.add(user_msg)
            
            # Save AI message
            ai_msg = Message(
                conversation_id=conversation_id,
                role="assistant", 
                content=ai_message
            )
            db.add(ai_msg)
            
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving conversation and messages: {e}")
        finally:
            db.close()

    def create_react_agent(self, model_type: ModelType, context_sources: List[ContextSource], user_preferences: str):
        """Create React Agent with tools and context"""
        # Get LLM
        llm = self.get_llm(model_type)
        
        # Get tools
        tools = DataAnalysisService.get_analysis_tools()
        
        # Build system prompt with context
        system_prompt = self._build_agent_system_prompt(context_sources, user_preferences)
        
        # Create React Agent with proper parameters
        from langchain_core.prompts import PromptTemplate
        from langchain.agents import AgentExecutor
        from langchain import hub
        
        # Use standard React prompt with custom system message
        # First try to get from hub, fallback to manual
        try:
            prompt = hub.pull("hwchase17/react")
            # Modify the template to include our system prompt and chat history
            original_template = prompt.template
            enhanced_template = f"""{system_prompt}

Previous conversation history:
{{chat_history}}

{original_template}"""
            prompt.template = enhanced_template
        except Exception as e:
            logger.error(f"Error loading prompt from hub: {e}")
            # Fallback: Standard React format with our enhancements
            prompt = PromptTemplate.from_template(f"""{system_prompt}

Previous conversation history:
{{chat_history}}

Answer the following questions as best you can. You have access to the following tools:

{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}""")
        
        # Create React Agent with all required parameters
        agent = create_react_agent(llm, tools, prompt)
        
        # Create Agent Executor with the agent and tools
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,  # Enable verbose for debugging
            handle_parsing_errors=True,
            max_iterations=5,  # Allow more iterations
            return_intermediate_steps=True  # Return intermediate steps for better handling
        )
        
        return agent_executor
    
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

            # Chat history text
            chat_history_text = ""
            for msg in memory.chat_memory.messages or []:
                if getattr(msg, "type", "") == "human":
                    chat_history_text += f"Human: {msg.content}\n"
                elif getattr(msg, "type", "") == "ai":
                    chat_history_text += f"Assistant: {msg.content}\n"
            if not chat_history_text:
                chat_history_text = "No previous conversation."

            # Input untuk agent
            agent_input = {
                "input": message,
                "chat_history": chat_history_text.strip(),
            }

            final_answer_for_storage = ""
            accumulated_content = ""
            in_final_answer_phase = False
            chart_data_collected = None
            chart_data_sent = False

            # --- STREAMING LOOP ---
            async for event in agent.astream_events(agent_input, version="v2"):
                etype = event["event"]
                data = event.get("data", {})

                # Handle tool execution completion - collect chart data
                if etype == "on_tool_end":
                    tool_output = data.get("output")
                    tool_name = event.get("name", "")
                    
                    if tool_name == "generate_chart" and tool_output:
                        try:
                            # Store chart data for later inclusion in final answer
                            chart_data_collected = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
                            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Chart generated successfully!', 'done': False, 'conversation_id': conversation_id})}\n\n"
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse chart data: {tool_output}")
                            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Chart generation completed', 'done': False, 'conversation_id': conversation_id})}\n\n"
                    elif tool_output:
                        # Stream other tool outputs as thinking content
                        yield f"data: {json.dumps({'type': 'thinking', 'content': f'{tool_name} completed', 'done': False, 'conversation_id': conversation_id})}\n\n"

                # Main streaming logic for LLM tokens
                elif etype == "on_chat_model_stream":
                    chunk_data = data.get("chunk")
                    if chunk_data and hasattr(chunk_data, 'content') and chunk_data.content:
                        token = chunk_data.content
                        accumulated_content += token

                        # Detect final answer phase
                        if "Final Answer:" in accumulated_content and not in_final_answer_phase:
                            in_final_answer_phase = True
                            
                            # If we have chart data and haven't sent it, send it first
                            if chart_data_collected and not chart_data_sent:
                                yield f"data: {json.dumps({'type': 'chart', 'content': chart_data_collected, 'done': False, 'conversation_id': conversation_id})}\n\n"
                                chart_data_sent = True
                            
                            # Extract and stream the final answer portion
                            final_answer_start = accumulated_content.find("Final Answer:") + len("Final Answer:")
                            final_answer_portion = accumulated_content[final_answer_start:].strip()
                            
                            if final_answer_portion:
                                yield f"data: {json.dumps({'type': 'answer','content': final_answer_portion,'done': False,'conversation_id': conversation_id})}\n\n"
                        
                        elif in_final_answer_phase:
                            # Continue streaming final answer tokens
                            yield f"data: {json.dumps({'type': 'answer','content': token,'done': False,'conversation_id': conversation_id})}\n\n"
                        
                        else:
                            # Filter out React Agent parsing error messages
                            if any(error_phrase in accumulated_content for error_phrase in [
                                'Invalid Format: Missing',
                                'Action:',
                                'Thought:',
                                '_Exception result:'
                            ]) and len(accumulated_content) < 200:
                                # Skip streaming these error messages to avoid noise
                                continue
                            
                            # This is thinking phase - but check if we should switch to answer mode
                            # If we have chart data and this doesn't look like reasoning, treat as answer
                            if (chart_data_collected and 
                                not any(keyword in token.lower() for keyword in ['thought', 'action', 'observation']) and
                                len(accumulated_content.strip()) > 20):
                                # Switch to answer mode
                                in_final_answer_phase = True
                                if not chart_data_sent:
                                    yield f"data: {json.dumps({'type': 'chart', 'content': chart_data_collected, 'done': False, 'conversation_id': conversation_id})}\n\n"
                                    chart_data_sent = True
                                yield f"data: {json.dumps({'type': 'answer','content': token,'done': False,'conversation_id': conversation_id})}\n\n"
                            else:
                                # This is thinking phase (includes thoughts, actions, observations)
                                yield f"data: {json.dumps({'type': 'thinking','content': token,'done': False,'conversation_id': conversation_id})}\n\n"

                # Reset accumulated content on new LLM start
                elif etype == "on_chat_model_start":
                    # Don't reset in_final_answer_phase, but consider resetting accumulated_content for new model calls
                    pass

                # Handle completion of LLM generation
                elif etype == "on_chat_model_end":
                    output_data = data.get("output")
                    if output_data and hasattr(output_data, 'content'):
                        full_content = output_data.content
                        
                        # Extract final answer for storage
                        if "Final Answer:" in full_content:
                            final_answer_start = full_content.find("Final Answer:") + len("Final Answer:")
                            final_answer_for_storage = full_content[final_answer_start:].strip()

                # Handle agent completion
                elif etype == "on_chain_end":
                    chain_output = data.get("output")
                    if chain_output and isinstance(chain_output, dict):
                        output_content = chain_output.get("output", "")
                        if output_content and "Final Answer:" in output_content:
                            final_answer_start = output_content.find("Final Answer:") + len("Final Answer:")
                            final_answer_for_storage = output_content[final_answer_start:].strip()
                            
                        # If we still haven't sent chart data and have it, send it now
                        if chart_data_collected and not chart_data_sent and not in_final_answer_phase:
                            yield f"data: {json.dumps({'type': 'chart', 'content': chart_data_collected, 'done': False, 'conversation_id': conversation_id})}\n\n"
                            chart_data_sent = True

            # Handle case where no proper final answer was generated
            if not final_answer_for_storage and chart_data_collected:
                # If we have chart data but no final answer, send chart if not sent yet
                if not chart_data_sent:
                    yield f"data: {json.dumps({'type': 'chart', 'content': chart_data_collected, 'done': False, 'conversation_id': conversation_id})}\n\n"
                    chart_data_sent = True
                # Don't send hardcoded message, let it be empty for proper completion
                final_answer_for_storage = "Chart generated successfully."
            elif not final_answer_for_storage:
                # If no final answer at all, provide a generic completion message
                final_answer_for_storage = "Task completed."

            # Send completion signal
            yield f"data: {json.dumps({'type': 'answer','content': '','done': True,'conversation_id': conversation_id})}\n\n"

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