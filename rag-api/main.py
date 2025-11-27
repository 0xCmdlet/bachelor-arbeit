import os
import structlog
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio

from retrieval import RAGRetriever
from generation import LLMGenerator
from conversational_agent import ConversationalRAGAgent

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

app = FastAPI(title="RAG API", version="1.0.0")

# Initialize components
retriever = RAGRetriever()
generator = LLMGenerator()
agent = None  # Lazy initialization


def get_agent() -> ConversationalRAGAgent:
    """Get or initialize the conversational agent"""
    global agent
    if agent is None:
        agent = ConversationalRAGAgent()
    return agent

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.3
    collection: Optional[str] = None  # Override QDRANT_COLLECTION env var

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    processing_time: float

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Ollama is responsive
        async with httpx.AsyncClient() as client:
            ollama_url = os.getenv('OLLAMA_URL', os.getenv('VLLM_URL', 'http://localhost:11434'))
            response = await client.get(f"{ollama_url}/v1/models", timeout=5.0)
            ollama_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        ollama_status = "unreachable"

    return {
        "status": "healthy",
        "ollama_status": ollama_status,
        "retriever": "ready",
        "generator": "ready"
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main RAG query endpoint"""
    import time
    start_time = time.time()

    logger.info("rag_query_started", query=request.query, top_k=request.top_k)

    try:
        # Step 1: Retrieve relevant documents
        retrieved_docs = await retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            collection=request.collection  # Pass collection override
        )

        if not retrieved_docs:
            logger.warning("no_documents_retrieved", query=request.query)
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query=request.query,
                processing_time=time.time() - start_time
            )

        logger.info("documents_retrieved", count=len(retrieved_docs))

        # Step 2: Generate answer using LLM
        answer = await generator.generate_answer(
            query=request.query,
            context_docs=retrieved_docs
        )

        processing_time = time.time() - start_time

        logger.info(
            "rag_query_completed",
            query=request.query,
            sources_count=len(retrieved_docs),
            processing_time=processing_time
        )

        return QueryResponse(
            answer=answer,
            sources=retrieved_docs,
            query=request.query,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error("rag_query_failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get retrieval system statistics"""
    try:
        stats = await retriever.get_stats()
        return stats
    except Exception as e:
        logger.error("stats_retrieval_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@app.get("/model")
async def get_model_info():
    """Get LLM model information"""
    try:
        model_info = await generator.get_model_info()
        return model_info
    except Exception as e:
        logger.error("model_info_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model info retrieval failed: {str(e)}")

# ============================================================================
# CONVERSATIONAL RAG ENDPOINTS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    tools_used: List[str]
    message_count: int
    sources: List[Dict[str, Any]] = []

class ConversationCreate(BaseModel):
    title: Optional[str] = None

class ConversationResponse(BaseModel):
    thread_id: str
    title: str
    created_at: Optional[str]
    updated_at: Optional[str]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message in a conversational context

    Creates a new conversation if thread_id is not provided.
    Automatically maintains conversation history using LangChain memory.
    """
    try:
        agent_instance = get_agent()

        # Generate thread_id if not provided
        thread_id = request.thread_id or str(uuid.uuid4())

        # Create conversation entry if new
        if not request.thread_id:
            await agent_instance.create_conversation(thread_id)

        logger.info("chat_request_started", thread_id=thread_id, message=request.message[:100])

        # Send message to agent
        result = await agent_instance.chat(
            message=request.message,
            thread_id=thread_id
        )

        logger.info("chat_request_completed", thread_id=thread_id)

        return ChatResponse(**result)

    except Exception as e:
        logger.error("chat_request_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreate):
    """Create a new conversation"""
    try:
        agent_instance = get_agent()
        thread_id = str(uuid.uuid4())

        conversation = await agent_instance.create_conversation(
            thread_id=thread_id,
            title=request.title
        )

        logger.info("conversation_created", thread_id=thread_id)

        return ConversationResponse(**conversation)

    except Exception as e:
        logger.error("conversation_creation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Conversation creation failed: {str(e)}")

@app.get("/conversations")
async def list_conversations(limit: int = 50):
    """List recent conversations"""
    try:
        agent_instance = get_agent()
        conversations = await agent_instance.list_conversations(limit=limit)

        return {"conversations": conversations, "count": len(conversations)}

    except Exception as e:
        logger.error("conversation_listing_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")

@app.get("/conversations/{thread_id}/history")
async def get_conversation_history(thread_id: str):
    """Get conversation history for a specific thread"""
    try:
        agent_instance = get_agent()
        history = await agent_instance.get_conversation_history(thread_id)

        return {"thread_id": thread_id, "messages": history, "count": len(history)}

    except Exception as e:
        logger.error("history_retrieval_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

@app.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """Delete a conversation and its history"""
    try:
        agent_instance = get_agent()
        success = await agent_instance.delete_conversation(thread_id)

        if success:
            return {"message": "Conversation deleted", "thread_id": thread_id}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversation_deletion_failed", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)