import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
import requests
import urllib.parse

# Add project root to path for imports FIRST
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load .env from project root BEFORE importing anything that reads env vars
from dotenv import load_dotenv
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path, override=True)

# Now import modules that depend on environment variables
from .local_llm import create_local_llm_from_config
from .configuration import Configuration
from .prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions
)
from .utils import get_research_topic
from main import run_search_pipeline

# Configure logger
logger = logging.getLogger(__name__)

# Used for Brightdata Search API
BRIGHTDATA_API_KEY = os.getenv("BRIGHTDATA_API_KEY")
BRIGHTDATA_BASE_URL = "https://api.brightdata.com/request"

# Configuration for llama-swap API
LLAMA_SWAP_BASE_URL = os.getenv("LLAMA_SWAP_BASE_URL", "http://127.0.0.1:8080")
LLAMA_SWAP_API_URL = f"{LLAMA_SWAP_BASE_URL}/v1"

def safe_print(text, **kwargs):
    """Safely handles logging of text messages."""
    msg = str(text)
    if len(msg) > 5000:
        msg = msg[:5000] + "... [TRUNCATED]"
    
    if "ERROR" in msg.upper():
        logger.error(msg)
    elif "DEBUG" in msg.upper():
        logger.debug(msg)
    else:
        logger.info(msg)

async def run_research_agent(
    messages: List[Dict[str, Any]],
    initial_search_query_count: int = 3,
    max_research_loops: int = 3,
    reasoning_model: str = None,
    instructions: Optional[str] = None
):
    # Use GEMINI_MODEL from environment if no model specified
    if reasoning_model is None:
        reasoning_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    """
    A direct implementation of the research agent without LangGraph.
    Yields events as they happen for streaming.
    """
    import time
    start_time = time.perf_counter()
    logger.debug(f"Starting run_research_agent with topic: {messages[-1].get('content')}")
    
    # 1. Setup configuration and LLM
    config = {"configurable": {"model_name": reasoning_model}}
    configurable = Configuration() # Default config
    
    # 2. Generate Initial Queries
    yield {"event": "generate_query", "data": {"search_query": ["Generating queries..."]}}
    
    local_llm = create_local_llm_from_config(config, reasoning_model)
    current_date = get_current_date()
    # Extract research topic from messages
    history_messages = []
    for m in messages:
        if m["type"] == "human":
            history_messages.append(HumanMessage(content=m["content"]))
        else:
            history_messages.append(AIMessage(content=m["content"]))
            
    research_topic = get_research_topic(history_messages)
    
    # If explicit instructions are provided, append them to the research topic context
    if instructions:
        research_topic += f"\n\nAdditional Instructions:\n{instructions}"
    
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
        number_queries=initial_search_query_count,
    )
    
    local_llm = create_local_llm_from_config(config, reasoning_model)
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: local_llm.call([{"role": "user", "content": formatted_prompt}])
    )
    logger.debug(f"Query generator response received: {response[:100]}...")
    
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
            
        data = json.loads(json_str)
        search_queries = data.get("query", [research_topic])
    except:
        search_queries = [research_topic]
        
    yield {"event": "generate_query", "data": {"search_query": search_queries}}
    
    # 3. Web Research
    # 3. Web Research Loop
    sources_gathered = []
    all_web_research_results = []
    
    for query_idx, query in enumerate(search_queries):
        try:
            logger.info(f"Starting web research for query: {query}")
            
            loop = asyncio.get_event_loop()
            q = asyncio.Queue()
            
            def sync_log(msg):
                msg_str = str(msg)
                
                # Expanded milestones for better UI feedback
                ui_milestones = [
                    "Searching",
                    "Found",
                    "Web Research Done",
                    "content extraction",
                    "exported to search_text.md",
                    "Pipeline finished"
                ]
                
                if any(m in msg_str for m in ui_milestones):
                    display_msg = msg_str.replace("[Run] ", "").replace("[Scraper] ", "")
                    loop.call_soon_threadsafe(q.put_nowait, display_msg)
                
            task = asyncio.ensure_future(
                loop.run_in_executor(
                    None,
                    lambda: run_search_pipeline(query, max_results=3, log_callback=sync_log)
                )
            )
            
            while not task.done() or not q.empty():
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if task.done() and q.empty():
                        break
                except Exception as yield_err:
                    logger.debug(f"Yield error: {yield_err}")
                    break
                
            top_results = task.result()
            logger.debug(f"Pipeline finished. Found {len(top_results)} results.")
            
            current_sources = []
            combined_content = ""
            for r_idx, res in enumerate(top_results):
                title = res.get("title", f"Result {r_idx+1}")
                link = res.get("link", "")
                snippet = res.get("snippet", "")
                
                source = {
                    "label": title,
                    "short_url": f"[{len(sources_gathered) + 1}]",
                    "value": link
                }
                sources_gathered.append(source)
                current_sources.append(source)
                combined_content += f"\n\nSource {source['short_url']}: {title}\n{snippet}\nURL: {link}"
            
            try:
                md_path = os.path.join(project_root, "search_text.md")
                if os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        scraped_content = f.read()
                    if scraped_content and len(scraped_content) > 100:
                        combined_content += f"\n\n--- Full Scraped Content Snippet ---\n{scraped_content[:8000]}"
            except Exception as e:
                logger.error(f"Could not read search_text.md: {e}")
            
            if not current_sources:
                combined_content = f"No results retrieved for '{query}'."
            
            # [Speed] Skipping per-query synthesis, collecting raw data for final answer
            all_web_research_results.append(combined_content)
            
        except Exception as e:
            logger.error(f"CRITICAL SEARCH ERROR: {e}")
            all_web_research_results.append(f"Error: {str(e)}")
            
    summaries_text = "\n\n---\n\n".join(all_web_research_results)
    # 7. Generate Final Answer
    yield {"event": "finalize_answer", "data": {"status": "Finalizing answer..."}}
    
    final_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
        summaries=summaries_text
    )
    
    try:
        final_answer = await loop.run_in_executor(
            None,
            lambda: local_llm.call([{"role": "user", "content": final_prompt}])
        )
        logger.debug(f"Final answer generated.")
        
        # Replace [1], [2], etc. with actual links
        sorted_sources = sorted(sources_gathered, key=lambda x: len(x["short_url"]), reverse=True)
        for source in sorted_sources:
            if source["short_url"] in final_answer:
                markdown_link = f"{source['short_url']}({source['value']})"
                final_answer = final_answer.replace(source["short_url"], markdown_link)
        
        # Time taken calculation
        end_time = time.perf_counter()
        time_taken = round(end_time - start_time, 2)

        # Save the final answer to results.txt
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results_content = f"Research Topic: {research_topic}\n"
            results_content += f"Generated on: {timestamp}\n"
            results_content += f"Time Taken: {time_taken}s\n"
            results_content += f"{'='*60}\n\n"
            results_content += final_answer
            results_content += f"\n\n{'='*60}\n"
            results_content += f"Sources ({len(sources_gathered)}):\n"
            for idx, source in enumerate(sources_gathered, 1):
                results_content += f"{idx}. {source['label']}\n   {source['value']}\n"
            
            results_path = os.path.join(project_root, "results.txt")
            with open(results_path, "w", encoding="utf-8") as f:
                f.write(results_content)
            logger.info(f"Final answer saved to results.txt")
        except Exception as e:
            logger.error(f"Failed to save results to file: {e}")
                
        yield {
            "event": "complete", 
            "data": {
                "messages": [{"type": "ai", "content": final_answer, "id": "final_answer"}],
                "sources_gathered": sources_gathered,
                "time_taken": time_taken
            }
        }
    except Exception as e:
        logger.error(f"Error generating final answer: {e}")
        yield {"event": "error", "data": f"Failed to generate final answer: {str(e)}"}
