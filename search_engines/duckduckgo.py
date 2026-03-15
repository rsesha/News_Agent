from ddgs import DDGS
import time
import logging

logger = logging.getLogger(__name__)

def safe_print(text, **kwargs):
    """Safely handles logging of text messages."""
    msg = str(text)
    if "ERROR" in msg.upper() or "FAILED" in msg.upper():
        logger.error(msg)
    elif "DEBUG" in msg.upper():
        logger.debug(msg)
    else:
        logger.info(msg)

def search(query, max_results=5):
    """
    Performs a search using the new 'ddgs' library.
    """
    results = []
    # Using a 10s timeout to prevent infinite hangs
    for attempt in range(2):
        try:
            # Note: ddgs uses httpx internally and timeout can be set on the object
            with DDGS(timeout=10) as ddgs:
                ddgs_gen = ddgs.text(query, max_results=max_results)
                if ddgs_gen:
                    for r in ddgs_gen:
                        results.append({
                            "title": r.get("title"),
                            "link": r.get("href"),
                            "snippet": r.get("body"),
                            "source": "DuckDuckGo"
                        })
                if results:
                    break
        except Exception as e:
            logger.error(f"DuckDuckGo error (attempt {attempt+1}): {e}")
            if "Ratelimit" in str(e):
                time.sleep(2)
            continue
    
    logger.debug(f"DuckDuckGo found {len(results)} results")
    return results


if __name__ == "__main__":
    import argparse
    # Manual config for CLI usage
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(description="DuckDuckGo Search")
    parser.add_argument("--search", required=True, help="Search query string")
    parser.add_argument("--max", type=int, default=5, help="Max results (default: 5)")
    args = parser.parse_args()

    results = search(args.search, max_results=args.max)
    if not results:
        print("No results found.")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['title']}")
            print(f"   URL: {r['link']}")
            if r.get('snippet'):
                print(f"   Snippet: {r['snippet'][:200]}")
