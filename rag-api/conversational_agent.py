"""
Conversational RAG Agent using LangChain/LangGraph
Implements short-term memory with PostgreSQL checkpointer
"""
import os
import asyncio
import structlog
from typing import List, Dict, Any, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from retrieval import RAGRetriever

logger = structlog.get_logger()

# Configuration
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://admin:admin123@postgres:5432/mydb")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Global retriever instance
_retriever = None
# Global storage for last retrieved sources (used to pass data from tool to agent)
_last_sources: List[Dict[str, Any]] = []


def get_retriever() -> RAGRetriever:
    """Get or initialize the RAG retriever"""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


@tool
def search_documents(query: str) -> str:
    """
    Search the document knowledge base for relevant information.

    Args:
        query: The search query to find relevant documents

    Returns:
        Formatted context from retrieved documents
    """
    global _last_sources

    try:
        retriever = get_retriever()

        # Retrieve relevant documents synchronously
        # Use asyncio.run to execute the async retrieve method
        results = asyncio.run(retriever.retrieve(query=query, top_k=5, min_score=0.3))

        if not results:
            _last_sources = []
            return "No relevant documents found for this query."

        # Store sources globally so they can be accessed by the agent
        _last_sources = results

        # Format results into readable context
        context_parts = []
        for i, doc in enumerate(results, 1):
            filename = doc.get('filename', 'Unknown')
            text = doc.get('text', '')
            score = doc.get('score', 0.0)

            # Truncate very long documents
            if len(text) > 1500:
                text = text[:1500] + "..."

            context_parts.append(
                f"--- Document {i} (Source: {filename}, Relevance: {score:.2f}) ---\n{text}"
            )

        context = "\n\n".join(context_parts)

        logger.info(
            "rag_tool_executed",
            query=query,
            documents_found=len(results),
            avg_score=sum(r['score'] for r in results) / len(results)
        )

        return context

    except Exception as e:
        logger.error("rag_tool_failed", query=query, error=str(e))
        _last_sources = []
        return f"Error searching documents: {str(e)}"


class ConversationalRAGAgent:
    """
    LangChain-based conversational RAG agent with PostgreSQL memory persistence
    """

    def __init__(self):
        """Initialize the conversational agent"""
        # Storage for retrieved sources (populated during tool execution)
        self.last_retrieved_sources: List[Dict[str, Any]] = []

        try:
            # Initialize Ollama LLM (langchain-ollama v0.3+)
            # Use OLLAMA_HOST env var or set base_url if needed
            os.environ.setdefault('OLLAMA_HOST', OLLAMA_URL)

            self.llm = ChatOllama(
                model=LLM_MODEL,
                temperature=TEMPERATURE,
                num_predict=MAX_TOKENS,
            )

            # Initialize PostgreSQL checkpointer with psycopg3
            self.connection_pool = ConnectionPool(
                conninfo=POSTGRES_DSN,
                kwargs={"autocommit": True}
            )

            self.checkpointer = PostgresSaver(self.connection_pool)

            # Setup checkpoint tables if they don't exist (first run)
            self.checkpointer.setup()

            # Create the agent with tools and memory (LangChain 1.0 API)
            self.agent = create_agent(
                model=self.llm,
                tools=[search_documents],
                checkpointer=self.checkpointer,
                system_prompt="""Du bist ein hilfreicher Assistent mit Zugriff auf eine Dokumenten-Wissensdatenbank.

Anweisungen:
1. Verwende das search_documents Tool, um relevante Informationen aus der Wissensdatenbank zu finden
2. Beantworte Fragen NUR basierend auf den bereitgestellten Dokumenten
3. Wenn die Dokumente nicht genügend Informationen enthalten, sage dies klar und deutlich
4. Erinnere dich an vorherige Nachrichten in der Konversation, um kontextbezogene Antworten zu geben
5. Sei präzise, aber umfassend in deinen Antworten
6. Zitiere spezifische Quellen, wenn relevant
7. Wenn mehrere Dokumente relevante Informationen enthalten, fasse die Informationen angemessen zusammen

Du kannst mehrstufige Konversationen führen und dich an den Kontext früherer Nachrichten erinnern."""
            )

            logger.info(
                "conversational_agent_initialized",
                model=LLM_MODEL,
                ollama_url=OLLAMA_URL,
                checkpointer="PostgreSQL"
            )

        except Exception as e:
            logger.error("agent_initialization_failed", error=str(e))
            raise

    async def chat(
        self,
        message: str,
        thread_id: str,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a message and get a response from the agent

        Args:
            message: User message
            thread_id: Conversation thread ID for memory persistence
            stream: Whether to stream the response (not implemented yet)

        Returns:
            Dictionary with response and metadata
        """
        global _last_sources

        try:
            logger.info("agent_chat_started", thread_id=thread_id, message=message[:100])

            # Clear previous sources before new request
            _last_sources = []

            # Configure the agent run
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }

            # Use sync invoke() wrapped in asyncio.to_thread() for async compatibility
            # This is required because PostgresSaver doesn't fully support async operations
            result = await asyncio.to_thread(
                self.agent.invoke,
                {"messages": [{"role": "user", "content": message}]},
                config
            )

            # Extract the assistant's response
            messages = result.get("messages", [])

            # Get the last AI message
            assistant_message = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    assistant_message = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "assistant":
                    assistant_message = msg.get("content", "")
                    break

            if not assistant_message:
                assistant_message = "I'm sorry, I couldn't generate a response."

            # Extract tool calls for metadata
            tool_calls = []
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls.extend([tc.get('name', 'unknown') for tc in msg.tool_calls])
                elif isinstance(msg, dict) and msg.get("tool_calls"):
                    tool_calls.extend([tc.get('name', 'unknown') for tc in msg["tool_calls"]])

            # Get sources from global variable (populated by search_documents tool)
            sources = _last_sources.copy() if _last_sources else []

            logger.info(
                "agent_chat_completed",
                thread_id=thread_id,
                response_length=len(assistant_message),
                tools_used=tool_calls,
                sources_count=len(sources)
            )

            return {
                "response": assistant_message,
                "thread_id": thread_id,
                "tools_used": list(set(tool_calls)),
                "message_count": len(messages),
                "sources": sources
            }

        except Exception as e:
            import traceback
            logger.error(
                "agent_chat_failed",
                thread_id=thread_id,
                message=message[:100],
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc()
            )
            raise

    async def get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a thread

        Args:
            thread_id: Conversation thread ID

        Returns:
            List of messages in the conversation
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}

            # Get the latest checkpoint using sync method
            state = await asyncio.to_thread(self.agent.get_state, config)

            if not state or not state.values:
                return []

            # Extract messages from state
            messages = state.values.get("messages", [])

            # Convert to simple format
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    history.append({"role": "system", "content": msg.content})
                elif isinstance(msg, dict):
                    history.append(msg)

            return history

        except Exception as e:
            logger.error("history_retrieval_failed", thread_id=thread_id, error=str(e))
            return []

    async def delete_conversation(self, thread_id: str) -> bool:
        """
        Delete a conversation thread

        Args:
            thread_id: Conversation thread ID to delete

        Returns:
            Success status
        """
        try:
            # LangGraph doesn't have built-in delete, so we manually delete from DB
            conn = self.connection_pool.getconn()
            try:
                cur = conn.cursor()

                # Delete checkpoints
                cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))

                # Delete conversation metadata
                cur.execute("DELETE FROM conversations WHERE thread_id = %s", (thread_id,))

                conn.commit()

                logger.info("conversation_deleted", thread_id=thread_id)
                return True

            finally:
                self.connection_pool.putconn(conn)

        except Exception as e:
            logger.error("conversation_deletion_failed", thread_id=thread_id, error=str(e))
            return False

    async def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent conversations

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation metadata
        """
        try:
            conn = self.connection_pool.getconn()
            try:
                cur = conn.cursor()

                # Get conversations with their latest checkpoint time
                cur.execute("""
                    SELECT
                        c.thread_id,
                        c.title,
                        c.created_at,
                        c.updated_at,
                        c.metadata,
                        COUNT(ck.checkpoint_id) as message_count
                    FROM conversations c
                    LEFT JOIN checkpoints ck ON c.thread_id = ck.thread_id
                    GROUP BY c.thread_id, c.title, c.created_at, c.updated_at, c.metadata
                    ORDER BY c.updated_at DESC
                    LIMIT %s
                """, (limit,))

                rows = cur.fetchall()

                conversations = []
                for row in rows:
                    conversations.append({
                        "thread_id": row[0],
                        "title": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "updated_at": row[3].isoformat() if row[3] else None,
                        "metadata": row[4] or {},
                        "message_count": row[5] or 0
                    })

                return conversations

            finally:
                self.connection_pool.putconn(conn)

        except Exception as e:
            logger.error("conversation_listing_failed", error=str(e))
            return []

    async def create_conversation(self, thread_id: str, title: str = None) -> Dict[str, Any]:
        """
        Create a new conversation entry

        Args:
            thread_id: Unique thread ID
            title: Optional conversation title

        Returns:
            Conversation metadata
        """
        try:
            conn = self.connection_pool.getconn()
            try:
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO conversations (thread_id, title)
                    VALUES (%s, %s)
                    ON CONFLICT (thread_id) DO UPDATE
                    SET updated_at = now()
                    RETURNING thread_id, title, created_at, updated_at
                """, (thread_id, title or f"Conversation {thread_id[:8]}"))

                row = cur.fetchone()
                conn.commit()

                return {
                    "thread_id": row[0],
                    "title": row[1],
                    "created_at": row[2].isoformat() if row[2] else None,
                    "updated_at": row[3].isoformat() if row[3] else None
                }

            finally:
                self.connection_pool.putconn(conn)

        except Exception as e:
            logger.error("conversation_creation_failed", thread_id=thread_id, error=str(e))
            raise
