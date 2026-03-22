# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import sys
import os
import json
import asyncio

def safe_print(text, **kwargs):
    """Safely prints text to handle Windows console encoding issues and long strings."""
    try:
        # Avoid flush/end quirks and truncate extremely long logs
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['flush', 'end']}
        msg = str(text)
        if len(msg) > 5000:
            msg = msg[:5000] + "... [TRUNCATED]"
            
        print(msg, **clean_kwargs)
    except:
        try:
            clean_text = str(text).encode('ascii', 'ignore').decode('ascii')[:5000]
            print(clean_text)
        except:
            pass

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from .research_agent import run_research_agent
except ImportError:
    from research_agent import run_research_agent

# Define the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    messages: List[Dict[str, Any]]
    initial_search_query_count: Optional[int] = 3
    max_research_loops: Optional[int] = 3
    reasoning_model: Optional[str] = None  # Will use GEMINI_MODEL from .env
    instructions: Optional[str] = None

@app.post("/chat")
async def chat(request: QueryRequest, reasoning_model: str = None):
    """Process a chat query and stream results"""
    
    # Use GEMINI_MODEL from environment if no model specified
    if reasoning_model is None:
        reasoning_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    
    async def event_generator():
        # Create a queue to relay events from the agent task
        queue = asyncio.Queue()
        
        # Define a wrapper to put events into the queue
        async def agent_task():
            try:
                gen = run_research_agent(
                    messages=request.messages,
                    initial_search_query_count=request.initial_search_query_count or 3,
                    max_research_loops=request.max_research_loops or 3,
                    reasoning_model=request.reasoning_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
                    instructions=request.instructions
                )
                
                execution_start = asyncio.get_event_loop().time()
                while True:
                    try:
                        elapsed = asyncio.get_event_loop().time() - execution_start
                        timeout = max(0.1, 45.0 - elapsed)
                        event = await asyncio.wait_for(gen.__anext__(), timeout=timeout)
                        await queue.put(event)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        safe_print("ERROR: Global search timeout (45s) triggered.")
                        await queue.put({"event": "error", "data": "Search Timeout: Operation took longer than 45 seconds."})
                        break

                await queue.put({"event": "done", "data": {}})
            except Exception as e:
                safe_print(f"ERROR in research_agent: {str(e)}")
                await queue.put({"event": "error", "data": str(e)})
                await queue.put({"event": "done", "data": {}})

        # Start the agent task in the background
        task = asyncio.create_task(agent_task())
        
        try:
            while True:
                try:
                    # Wait for an event with a timeout for heartbeats
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    if event.get("event") == "done":
                        break
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
        finally:
            task.cancel()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    """Returns the backend configuration for the frontend."""
    use_gemini = os.getenv("USE_GEMINI", "False").lower() == "true"
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    local_model = os.getenv("LOCAL_MODEL_NAME", "qwen-opus")
    
    return {
        "use_gemini": use_gemini,
        "default_model": gemini_model if use_gemini else local_model,
        "gemini_model": gemini_model,
        "local_model": local_model
    }

@app.get("/search")
async def search(query: str, effort: str = "medium", model: str = None):
    # Use GEMINI_MODEL from environment if no model specified
    if model is None:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    """Simple text search endpoint - returns results as plain text.
    
    Args:
        query: The search query
        effort: low/medium/high (affects number of research loops and queries)
        model: The model to use for LLM calls
    
    Returns:
        Plain text response with the synthesized answer
    """
    from pydantic import BaseModel
    
    class SimpleQueryRequest(BaseModel):
        messages: list
        initial_search_query_count: int
        max_research_loops: int
        reasoning_model: str
    
    # Convert effort to query counts
    effort_map = {
        "low": (1, 1),
        "medium": (3, 3),
        "high": (5, 10),
    }
    initial_queries, max_loops = effort_map.get(effort, (3, 3))
    
    # Create the request
    request = SimpleQueryRequest(
        messages=[{"type": "human", "content": query, "id": "1"}],
        initial_search_query_count=initial_queries,
        max_research_loops=max_loops,
        reasoning_model=model,
    )
    
    # Run the research agent and collect results
    result_text = ""
    final_sources = []
    
    async for event in run_research_agent(
        messages=request.messages,
        initial_search_query_count=request.initial_search_query_count,
        max_research_loops=request.max_research_loops,
        reasoning_model=request.reasoning_model,
    ):
        if event.get("event") == "complete":
            messages = event.get("data", {}).get("messages", [])
            if messages:
                result_text = messages[0].get("content", "")
            final_sources = event.get("data", {}).get("sources_gathered", [])
        elif event.get("event") == "error":
            result_text = f"Error: {event.get('data')}"
    
    # Format as plain text with sources
    response_text = f"{result_text}\n\n{'='*60}\nSources:\n"
    for i, source in enumerate(final_sources, 1):
        response_text += f"{i}. {source.get('label', 'Source')}\n   {source.get('value', '')}\n"
    
    return Response(content=response_text, media_type="text/plain")

def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend."""
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        from starlette.routing import Route
        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )
        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)

# Mount the frontend under /app
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2024)

