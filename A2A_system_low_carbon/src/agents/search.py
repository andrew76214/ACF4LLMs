"""
Search Agent for A2A Pipeline
Handles web search and retrieval using Tavily API + BGE embeddings + reranker
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

try:
    from FlagEmbedding import FlagModel, FlagReranker
except ImportError:
    FlagModel = None
    FlagReranker = None

import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from configs.config import config
from utils.logging import get_logger, log_search_results, PerformanceLogger
from utils.text_processing import normalize_text

logger = get_logger("search_agent")

@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    title: str
    content: str
    url: str
    score: float
    source: str = "web"
    snippet_id: Optional[str] = None

@dataclass
class RetrievalContext:
    """Context for retrieval with ranked passages"""
    query: str
    passages: List[SearchResult]
    search_time: float
    total_results: int
    reranked: bool = False

class SearchAgent:
    """
    Search Agent that performs web search and retrieval
    Uses Tavily API for search, BGE embeddings for dense retrieval, and BGE reranker
    """

    def __init__(self, search_config=None):
        """
        Initialize SearchAgent

        Args:
            search_config: Search configuration (uses global config if None)
        """
        self.config = search_config or config.search
        self.tavily_client = None
        self.embedding_model = None
        self.reranker_model = None
        self.bm25_vectorizer = None

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize search and retrieval models"""
        try:
            # Initialize Tavily client
            if self.config.tavily_api_key and TavilyClient:
                self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key)
                logger.info("Tavily client initialized successfully")
            else:
                logger.warning("Tavily client not initialized - API key missing or package not installed")

            # Initialize BGE embedding model
            if FlagModel:
                try:
                    self.embedding_model = FlagModel(
                        self.config.embedding_model,
                        query_instruction_for_retrieval="Represent this query for searching relevant passages:",
                        use_fp16=True
                    )
                    logger.info(f"BGE embedding model loaded: {self.config.embedding_model}")
                except Exception as e:
                    logger.warning(f"Failed to load FlagModel, falling back to SentenceTransformer: {e}")
                    self.embedding_model = SentenceTransformer(self.config.embedding_model)
            else:
                # Fallback to sentence-transformers
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"Sentence transformer model loaded: {self.config.embedding_model}")

            # Initialize BGE reranker
            if FlagReranker:
                try:
                    self.reranker_model = FlagReranker(
                        self.config.reranker_model,
                        use_fp16=True
                    )
                    logger.info(f"BGE reranker loaded: {self.config.reranker_model}")
                except Exception as e:
                    logger.warning(f"Failed to load BGE reranker: {e}")
                    self.reranker_model = None
            else:
                logger.warning("FlagReranker not available - reranking will be skipped")

            # Initialize BM25 vectorizer
            self.bm25_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=10000
            )

        except Exception as e:
            logger.error(f"Failed to initialize search models: {e}")
            raise

    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """
        Perform web search using Tavily API

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        if not self.tavily_client:
            logger.error("Tavily client not available")
            return []

        max_results = max_results or self.config.max_search_results

        try:
            with PerformanceLogger(f"Tavily search for '{query}'"):
                # Perform search
                response = self.tavily_client.search(
                    query=query,
                    search_depth=self.config.search_depth,
                    max_results=max_results
                )

                # Convert to SearchResult objects
                results = []
                for i, result in enumerate(response.get("results", [])):
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        url=result.get("url", ""),
                        score=result.get("score", 1.0),
                        source="tavily",
                        snippet_id=f"tavily_{i}"
                    )
                    results.append(search_result)

                log_search_results(query, len(results), [r.url for r in results])
                return results

        except Exception as e:
            logger.error(f"Tavily search failed for query '{query}': {e}")
            return []

    def retrieve_and_rerank(
        self,
        query: str,
        passages: Optional[List[SearchResult]] = None,
        top_k: Optional[int] = None
    ) -> RetrievalContext:
        """
        Retrieve relevant passages and rerank them

        Args:
            query: Query for retrieval
            passages: Pre-retrieved passages (if None, will search first)
            top_k: Number of top passages to return

        Returns:
            RetrievalContext with ranked passages
        """
        start_time = time.time()
        top_k = top_k or self.config.top_k_rerank

        try:
            # Get passages if not provided
            if passages is None:
                logger.info(f"Searching for query: '{query}'")
                passages = self.search(query, max_results=self.config.top_k_retrieve)

            if not passages:
                logger.warning("No passages found for retrieval")
                return RetrievalContext(
                    query=query,
                    passages=[],
                    search_time=time.time() - start_time,
                    total_results=0
                )

            logger.info(f"Retrieved {len(passages)} passages, reranking to top {top_k}")

            # Hybrid retrieval: BM25 + Dense
            ranked_passages = self._hybrid_retrieval(query, passages, top_k * 2)  # Get more for reranking

            # Cross-encoder reranking
            if self.reranker_model and len(ranked_passages) > 1:
                reranked_passages = self._rerank_passages(query, ranked_passages, top_k)
                reranked = True
            else:
                reranked_passages = ranked_passages[:top_k]
                reranked = False

            search_time = time.time() - start_time
            logger.info(f"Retrieval completed in {search_time:.2f}s, returned {len(reranked_passages)} passages")

            return RetrievalContext(
                query=query,
                passages=reranked_passages,
                search_time=search_time,
                total_results=len(passages),
                reranked=reranked
            )

        except Exception as e:
            logger.error(f"Retrieval and reranking failed: {e}")
            return RetrievalContext(
                query=query,
                passages=[],
                search_time=time.time() - start_time,
                total_results=0
            )

    def _hybrid_retrieval(self, query: str, passages: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        Perform hybrid retrieval combining BM25 and dense retrieval

        Args:
            query: Query text
            passages: List of passages to rank
            top_k: Number of top results to return

        Returns:
            Ranked passages
        """
        if not passages:
            return []

        try:
            # Extract text content
            passage_texts = [p.content for p in passages]

            # Dense retrieval scores
            dense_scores = self._compute_dense_scores(query, passage_texts)

            # BM25 scores
            bm25_scores = self._compute_bm25_scores(query, passage_texts)

            # Combine scores
            combined_scores = []
            for i in range(len(passages)):
                combined_score = (
                    self.config.dense_weight * dense_scores[i] +
                    self.config.bm25_weight * bm25_scores[i]
                )
                combined_scores.append(combined_score)

            # Rank by combined score
            ranked_indices = np.argsort(combined_scores)[::-1]
            ranked_passages = []

            for idx in ranked_indices[:top_k]:
                passage = passages[idx]
                passage.score = combined_scores[idx]
                ranked_passages.append(passage)

            return ranked_passages

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            # Fallback: return original passages with original scores
            return passages[:top_k]

    def _compute_dense_scores(self, query: str, passages: List[str]) -> List[float]:
        """Compute dense retrieval scores using embeddings"""
        try:
            if hasattr(self.embedding_model, 'encode_queries'):
                # FlagModel interface
                query_embedding = self.embedding_model.encode_queries([query])
                passage_embeddings = self.embedding_model.encode_corpus(passages)
            else:
                # SentenceTransformer interface
                query_embedding = self.embedding_model.encode([query])
                passage_embeddings = self.embedding_model.encode(passages)

            # Compute cosine similarities
            similarities = cosine_similarity(query_embedding, passage_embeddings)[0]
            return similarities.tolist()

        except Exception as e:
            logger.error(f"Dense scoring failed: {e}")
            return [1.0] * len(passages)

    def _compute_bm25_scores(self, query: str, passages: List[str]) -> List[float]:
        """Compute BM25 scores using TF-IDF approximation"""
        try:
            # Fit vectorizer on passages and transform
            tfidf_matrix = self.bm25_vectorizer.fit_transform(passages + [query])

            # Last row is query, others are passages
            query_vector = tfidf_matrix[-1]
            passage_vectors = tfidf_matrix[:-1]

            # Compute cosine similarities
            similarities = cosine_similarity(query_vector, passage_vectors)[0]

            # Normalize to 0-1 range
            if len(similarities) > 1:
                min_sim, max_sim = similarities.min(), similarities.max()
                if max_sim > min_sim:
                    similarities = (similarities - min_sim) / (max_sim - min_sim)

            return similarities.tolist()

        except Exception as e:
            logger.error(f"BM25 scoring failed: {e}")
            return [1.0] * len(passages)

    def _rerank_passages(self, query: str, passages: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        Rerank passages using cross-encoder

        Args:
            query: Query text
            passages: Passages to rerank
            top_k: Number of top results to return

        Returns:
            Reranked passages
        """
        if not self.reranker_model or not passages:
            return passages[:top_k]

        try:
            # Prepare query-passage pairs for reranking
            pairs = [[query, passage.content] for passage in passages]

            # Get reranking scores
            scores = self.reranker_model.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]

            # Update passage scores and rerank
            for i, passage in enumerate(passages):
                passage.score = scores[i] if i < len(scores) else passage.score

            # Sort by reranker scores
            reranked_passages = sorted(passages, key=lambda x: x.score, reverse=True)
            return reranked_passages[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return passages[:top_k]

    def multi_hop_search(self, query: str, max_hops: int = 2) -> RetrievalContext:
        """
        Perform multi-hop search for complex queries

        Args:
            query: Initial query
            max_hops: Maximum number of search hops

        Returns:
            RetrievalContext with results from all hops
        """
        start_time = time.time()
        all_passages = []
        queries_used = [query]

        logger.info(f"Starting multi-hop search with {max_hops} hops for: '{query}'")

        try:
            # First hop: direct search
            hop1_results = self.search(query, max_results=self.config.top_k_retrieve // 2)
            all_passages.extend(hop1_results)

            # Additional hops: extract entities/concepts and search
            for hop in range(1, max_hops):
                if not all_passages:
                    break

                # Extract additional search terms from current results
                additional_query = self._extract_follow_up_query(query, all_passages[-3:])
                if additional_query and additional_query not in queries_used:
                    logger.debug(f"Hop {hop + 1} query: '{additional_query}'")
                    hop_results = self.search(additional_query, max_results=self.config.top_k_retrieve // 2)
                    all_passages.extend(hop_results)
                    queries_used.append(additional_query)

            # Remove duplicates based on URL
            unique_passages = []
            seen_urls = set()
            for passage in all_passages:
                if passage.url not in seen_urls:
                    unique_passages.append(passage)
                    seen_urls.add(passage.url)

            # Final ranking
            if unique_passages:
                final_context = self.retrieve_and_rerank(
                    query=query,
                    passages=unique_passages,
                    top_k=self.config.top_k_rerank
                )
                final_context.search_time = time.time() - start_time
                return final_context

        except Exception as e:
            logger.error(f"Multi-hop search failed: {e}")

        return RetrievalContext(
            query=query,
            passages=[],
            search_time=time.time() - start_time,
            total_results=0
        )

    def _extract_follow_up_query(self, original_query: str, recent_passages: List[SearchResult]) -> Optional[str]:
        """
        Extract follow-up query from recent search results

        Args:
            original_query: Original query
            recent_passages: Recent search results

        Returns:
            Follow-up query or None
        """
        try:
            # Simple entity extraction from titles and content
            entities = set()
            for passage in recent_passages:
                # Extract capitalized words (simple NER)
                import re
                title_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', passage.title)
                content_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', passage.content[:200])
                entities.update(title_entities)
                entities.update(content_entities)

            # Remove common words
            common_words = {'The', 'This', 'That', 'And', 'But', 'For', 'With', 'From'}
            entities = entities - common_words

            # Create follow-up query
            if entities:
                # Pick most promising entity not in original query
                for entity in entities:
                    if entity.lower() not in original_query.lower():
                        return f"{entity} {original_query}"

        except Exception as e:
            logger.debug(f"Follow-up query extraction failed: {e}")

        return None

    def format_context_for_llm(self, context: RetrievalContext, max_tokens: Optional[int] = None) -> str:
        """
        Format retrieval context for LLM input

        Args:
            context: RetrievalContext to format
            max_tokens: Maximum tokens for context (rough estimate)

        Returns:
            Formatted context string
        """
        if not context.passages:
            return ""

        max_tokens = max_tokens or self.config.max_context_tokens

        formatted_passages = []
        total_length = 0

        for i, passage in enumerate(context.passages):
            # Format passage
            passage_text = f"Passage {i+1}: ({passage.title})\n{passage.content}\nSource: {passage.url}\n"

            # Rough token estimation (4 chars per token)
            estimated_tokens = len(passage_text) // 4

            if total_length + estimated_tokens > max_tokens and formatted_passages:
                break

            formatted_passages.append(passage_text)
            total_length += estimated_tokens

        context_str = "\n".join(formatted_passages)

        # Add metadata
        if context.reranked:
            context_str = f"Retrieved and reranked {len(context.passages)} passages:\n\n{context_str}"
        else:
            context_str = f"Retrieved {len(context.passages)} passages:\n\n{context_str}"

        return context_str

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of search components

        Returns:
            Dictionary of component health status
        """
        health = {
            "tavily_client": self.tavily_client is not None,
            "embedding_model": self.embedding_model is not None,
            "reranker_model": self.reranker_model is not None,
        }

        # Test Tavily connection
        if self.tavily_client:
            try:
                test_results = self.tavily_client.search("test query", max_results=1)
                health["tavily_api"] = len(test_results.get("results", [])) >= 0
            except:
                health["tavily_api"] = False

        return health