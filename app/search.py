# app/search.py

from duckduckgo_search import DDGS
from langchain.docstore.document import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_web(query: str, num_results: int = 3) -> str:
    """
    Performs a web search using DuckDuckGo and returns concatenated results.
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return

    Returns:
        str: Concatenated search results
    """
    logger.info(f"[SEARCH] Performing web search for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = [
                f"{r['title']} - {r['body']} ({r['href']})"
                for r in ddgs.text(query, max_results=num_results)
            ]
            logger.info(f"[SEARCH] Found {len(results)} results.")
            return "\n".join(results)
    except Exception as e:
        logger.error(f"[SEARCH] Web search failed: {e}")
        return ""

def search_web_as_document(query: str, num_results: int = 3) -> Document:
    """
    Perform a web search and wrap the result in a LangChain Document.
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return

    Returns:
        Document: A document containing the search results
    """
    content = search_web(query, num_results)
    return Document(
        page_content=content,
        metadata={"source": "web_search", "query": query}
    )
