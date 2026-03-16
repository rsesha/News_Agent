import os
import requests
import urllib.parse
import json
from .tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

# Load .env from project root FIRST
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path, override=True)

from .local_llm import LocalLLM, create_local_llm_from_config, convert_messages_to_llama_format

from .state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from .configuration import Configuration
from .prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from .utils import (
    get_research_topic,
)

# Used for Brightdata Search API
BRIGHTDATA_API_KEY = os.getenv("BRIGHTDATA_API_KEY")
BRIGHTDATA_BASE_URL = "https://api.brightdata.com/request"


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses local LLM to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    # Debug print for troubleshooting
    print("=== GENERATE_QUERY FUNCTION CALLED ===")
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Create local LLM instance
    local_llm = create_local_llm_from_config(config, configurable.query_generator_model)
    
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    
    # Call local LLM
    response = local_llm.call_structured_output([
        {"role": "user", "content": formatted_prompt}
    ], SearchQueryList)
    
    # DEBUG: Print the raw response to see what we're getting
    print(f"DEBUG: Raw response type: {type(response)}")
    print(f"DEBUG: Raw response content: {response}")
    
    # Parse the response from LLM - handle both string and object responses
    try:
        if isinstance(response, str):
            # Response is a JSON string, parse it
            # Clean up the JSON string by removing markdown formatting
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]  # Remove ```json prefix
            if json_str.endswith('```'):
                json_str = json_str[:-3]  # Remove ``` suffix
            
            # Try to parse JSON
            response_data = json.loads(json_str)
            print(f"DEBUG: Parsed response_data: {response_data}")
            if 'query' in response_data:
                query_list = response_data['query']
                print(f"DEBUG: Extracted query from response: {query_list}")
            else:
                # Fallback: use the research topic directly
                research_topic = get_research_topic(state["messages"]).strip()
                query_list = [research_topic]
                print(f"DEBUG: Using research topic as fallback: {research_topic}")
        elif isinstance(response, SearchQueryList) and hasattr(response, 'query'):
            # Response is a SearchQueryList object
            query_list = response.query
            print(f"DEBUG: Extracted query from SearchQueryList: {query_list}")
        else:
            # Fallback: use the research topic directly
            research_topic = get_research_topic(state["messages"]).strip()
            query_list = [research_topic]
            print(f"DEBUG: Using research topic as fallback: {research_topic}")
            
        # Ensure we have valid queries
        if not query_list or not isinstance(query_list, list):
            research_topic = get_research_topic(state["messages"]).strip()
            query_list = [research_topic]
            print(f"DEBUG: Final fallback to research topic: {research_topic}")
            
    except Exception as e:
        print(f"Error parsing structured response: {e}")
        # Fallback: use the research topic directly
        research_topic = get_research_topic(state["messages"]).strip()
        query_list = [research_topic]
        print(f"DEBUG: Exception fallback to research topic: {research_topic}")
    
    print(f"DEBUG: Final query_list: {query_list}")
    return {"search_query": query_list}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the Brightdata Search API.

    Executes a web search using the Brightdata Search API in combination with local LLM.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # OBVIOUS DEBUG PRINT - should always appear if this function is called
    print("=== WEB_RESEARCH FUNCTION CALLED ===")
    print(f"WEB_RESEARCH DEBUG: search_query = {state['search_query']}")
    
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    
    # Use the main.py run_search_pipeline
    try:
        print(f"WEB_RESEARCH DEBUG: Starting search using main.py pipeline for query: {state['search_query']}")
        
        # Add the project root to sys.path so we can import main
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        from main import run_search_pipeline
        
        # Run the search pipeline
        top_results = run_search_pipeline(state['search_query'], max_results=5)
        
        sources_gathered = []
        combined_content = ""
        
        for i, res in enumerate(top_results):
            title = res.get('title', f'Result {i+1}')
            url = res.get('link', '')
            snippet = res.get('snippet', '')
            sources_gathered.append({
                "label": title,
                "short_url": f"[{i+1}]",
                "value": url
            })
            combined_content += f"\n\n{i+1}. {title}\n{snippet}\nURL: {url}"

        # Try to include some scraped content
        try:
            md_path = os.path.join(project_root, "search_text.md")
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    scraped_content = f.read()
                if scraped_content and len(scraped_content) > 100:
                    # Give LLM access to some of the scraped text
                    combined_content += f"\n\n--- Full Scraped Content Snippet ---\n{scraped_content[:8000]}"
        except Exception as e:
            print(f"WEB_RESEARCH DEBUG: Could not read search_text.md: {e}")        

        if not sources_gathered:
            sources_gathered = [
                {
                    "label": f"Search result for '{state['search_query']}'",
                    "short_url": "[1]",
                    "value": "https://duckduckgo.com/?q=" + urllib.parse.quote(state['search_query'])
                }
            ]
            combined_content = f"No results retrieved for '{state['search_query']}'."
        
        # Synthesis
        local_llm = create_local_llm_from_config(config, configurable.query_generator_model)
        synthesis_prompt = f"Based on the following search results and scraped content, please provide a coherent summary that answers the research topic '{state['search_query']}'.\n\nSearch Results:\n{combined_content}\n\nPlease provide a well-structured response with proper citations to the sources used."
        
        llm_response = local_llm.call([
            {"role": "user", "content": synthesis_prompt}
        ])
        
        return {
            "sources_gathered": sources_gathered,
            "search_query": [state["search_query"]],
            "web_research_result": [llm_response],
        }
    except Exception as e:
        print(f"Error in overall search pipeline: {e}")
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": [f"Error occurred during search: {str(e)}"],
        }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries using local LLM.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    
    # Use local LLM for reflection
    local_llm = create_local_llm_from_config(config, reasoning_model)
    
    # For reflection, we'll use a simpler approach since we don't have structured output handling yet
    response = local_llm.call([
        {"role": "user", "content": formatted_prompt}
    ])
    
    # Simple parsing for demonstration - in reality this would be more sophisticated
    # This is a placeholder implementation
    try:
        # Parse the response to extract reflection data
        # This is very simplified - real implementation would be more robust
        is_sufficient = True  # Placeholder
        knowledge_gap = "No specific knowledge gap identified"  # Placeholder
        follow_up_queries = ["Follow-up query based on current research"]  # Placeholder
        
        print(f"REFLECTION DEBUG: is_sufficient = {is_sufficient}")
        print(f"REFLECTION DEBUG: research_loop_count = {state['research_loop_count']}")
        print(f"REFLECTION DEBUG: web_research_result = {state['web_research_result']}")
        
        # Return the parsed data
        return {
            "is_sufficient": is_sufficient,
            "knowledge_gap": knowledge_gap,
            "follow_up_queries": follow_up_queries,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }
    except Exception as e:
        # Fallback to basic response handling
        print(f"REFLECTION DEBUG: Exception occurred: {e}")
        return {
            "is_sufficient": True,
            "knowledge_gap": "Could not determine knowledge gap",
            "follow_up_queries": ["Generic follow-up query"],
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # Use local LLM for final answer generation
    local_llm = create_local_llm_from_config(config, reasoning_model)
    
    # Generate final answer using local LLM
    response = local_llm.call([
        {"role": "user", "content": formatted_prompt}
    ])

    print(f"FINALIZE_ANSWER DEBUG: Final response before URL replacement: {response[:200]}...")
    print(f"FINALIZE_ANSWER DEBUG: sources_gathered: {state['sources_gathered']}")
    
    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in response:
            response = response.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)
            print(f"FINALIZE_ANSWER DEBUG: Replaced {source['short_url']} with {source['value']}")

    print(f"FINALIZE_ANSWER DEBUG: Final response after URL replacement: {response[:200]}...")
    
    return {
        "messages": [AIMessage(content=response)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")