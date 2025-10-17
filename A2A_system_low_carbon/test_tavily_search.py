#!/usr/bin/env python3
"""
Test Tavily Search to diagnose why 0 passages are retrieved for HotpotQA questions
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ["TAVILY_API_KEY"] = "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"

from agents.search import SearchAgent
from utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(level="DEBUG", log_to_console=True, log_to_file=False)
logger = get_logger("tavily_test")

def test_tavily_search():
    """Test Tavily search with HotpotQA-style questions"""

    # Sample HotpotQA questions from the failed results
    test_queries = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "Scott Derrickson nationality",  # Simplified query
        "Ed Wood nationality",           # Simplified query
        "Corliss Archer actress Kiss and Tell film",  # Simplified query
    ]

    logger.info("🔍 Testing Tavily Search API")

    # Initialize search agent
    try:
        search_agent = SearchAgent()
        logger.info("SearchAgent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SearchAgent: {e}")
        return

    # Test each query
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n=== Test {i}/5: {query[:50]}... ===")

        try:
            # Perform search
            results = search_agent.search(query, max_results=5)

            logger.info(f"✅ Query: {query}")
            logger.info(f"📊 Results found: {len(results)}")

            if results:
                for j, result in enumerate(results):
                    logger.info(f"  {j+1}. Title: {result.title[:80]}...")
                    logger.info(f"     URL: {result.url}")
                    logger.info(f"     Score: {result.score}")
                    logger.info(f"     Content: {result.content[:150]}...")
                    logger.info("")
            else:
                logger.warning(f"❌ No results found for query: {query}")

        except Exception as e:
            logger.error(f"❌ Search failed for query '{query}': {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("🏁 Tavily search test completed")

if __name__ == "__main__":
    test_tavily_search()