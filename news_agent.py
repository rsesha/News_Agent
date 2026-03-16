#!/usr/bin/env python3
"""
Command-line interface for the News Agent research pipeline.

Usage:
    python search_and_answer.py --query "Who won the World Cup in 2022"
    python search_and_answer.py --query "What is the capital of France" --effort high

This script:
1. Reads configuration from .env file
2. Uses all 3 search engines (DuckDuckGo, Tavily, Brightdata)
3. Synthesizes results using the configured LLM (Gemini or local)
4. Outputs a cited answer to stdout
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load .env from project root
project_root = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path, override=True)

from backend.research_engine.research_agent import run_research_agent


def main():
    parser = argparse.ArgumentParser(
        description="Research any topic using AI-powered multi-engine search and synthesis"
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="The search query"
    )
    parser.add_argument(
        "--effort", "-e",
        choices=["low", "medium", "high"],
        default="medium",
        help="Research effort level (default: medium)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="LLM model to use (default: from .env USE_GEMINI setting)"
    )
    
    args = parser.parse_args()
    
    # Determine effort settings
    effort_map = {
        "low": (1, 1),
        "medium": (3, 3),
        "high": (5, 10),
    }
    initial_queries, max_loops = effort_map.get(args.effort, (3, 3))
    
    # Debug: Check .env loading
    print(f"\n{'='*60}")
    print(f"News Agent - Research Query")
    print(f"{'='*60}")
    print(f"Query: {args.query}")
    print(f"Effort: {args.effort} ({initial_queries} queries, {max_loops} loops)")
    
    # Determine model
    use_gemini_raw = os.getenv("USE_GEMINI", "NOT_SET")
    use_gemini = use_gemini_raw.lower() in ("true", "1", "yes")
    
    print(f"DEBUG: USE_GEMINI raw value: '{use_gemini_raw}'")
    print(f"DEBUG: USE_GEMINI parsed: {use_gemini}")
    
    if args.model:
        model = args.model
    elif use_gemini:
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    else:
        model = os.getenv("LOCAL_MODEL_NAME", "qwen-opus")
    
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    # Prepare messages for the research agent
    messages = [
        {"type": "human", "content": args.query, "id": "1"}
    ]
    
    # Run the research agent
    result_text = ""
    final_sources = []
    
    async def run_research():
        nonlocal result_text, final_sources
        
        async for event in run_research_agent(
            messages=messages,
            initial_search_query_count=initial_queries,
            max_research_loops=max_loops,
            reasoning_model=model,
        ):
            if event.get("event") == "generate_query":
                data = event.get("data", {})
                if data.get("search_query"):
                    queries = data["search_query"]
                    if isinstance(queries, list) and len(queries) > 0 and isinstance(queries[0], str):
                        print(f"Generating search queries: {', '.join(queries)}")
                    else:
                        print("Generating search queries...")
            elif event.get("event") == "finalize_answer":
                print("Synthesizing final answer...\n")
            elif event.get("event") == "complete":
                messages_data = event.get("data", {}).get("messages", [])
                if messages_data:
                    result_text = messages_data[0].get("content", "")
                final_sources = event.get("data", {}).get("sources_gathered", [])
            elif event.get("event") == "error":
                print(f"Error: {event.get('data')}")
    
    asyncio.run(run_research())
    
    # Print the result
    if result_text:
        print(result_text)
        print(f"\n{'='*60}")
        print("Sources:")
        for i, source in enumerate(final_sources, 1):
            label = source.get("label", "Source")
            url = source.get("value", "")
            print(f"{i}. {label}")
            if url:
                print(f"   {url}")
        print(f"{'='*60}")
        
        # Save to results.txt
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_content = f"Research Topic: {args.query}\n"
        results_content += f"Generated on: {timestamp}\n"
        results_content += f"Effort: {args.effort}\n"
        results_content += f"Model: {model}\n"
        results_content += f"{'='*60}\n\n"
        results_content += result_text
        results_content += f"\n\n{'='*60}\n"
        results_content += f"Sources ({len(final_sources)}):\n"
        for idx, source in enumerate(final_sources, 1):
            results_content += f"{idx}. {source.get('label', 'Source')}\n   {source.get('value', '')}\n"
        
        results_path = os.path.join(project_root, "results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(results_content)
        print(f"\nResults saved to: {results_path}")
    else:
        print("No result generated.")


if __name__ == "__main__":
    main()