"""
Router Agent for A2A Pipeline
Classifies tasks and routes them to appropriate solving strategies
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from configs.config import config
from utils.logging import get_logger

logger = get_logger("router_agent")

@dataclass
class RoutingDecision:
    """Represents a routing decision"""
    task_type: str  # "math", "qa", "multihop"
    use_search: bool
    prompt_style: str
    decoding_strategy: str
    num_samples: int
    temperature: float
    reasoning: Optional[str] = None
    confidence: float = 1.0

class RouterAgent:
    """
    Router Agent that decides how to handle each query
    Routes between math solving, single-hop QA, and multi-hop QA
    """

    def __init__(self, router_config=None):
        """
        Initialize RouterAgent

        Args:
            router_config: Router configuration (uses global config if None)
        """
        self.config = router_config or config.router
        self.math_keywords = set(self.config.math_keywords)
        self.qa_keywords = set(self.config.qa_keywords)

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def route(self, query: str, dataset: Optional[str] = None, context: Optional[str] = None) -> RoutingDecision:
        """
        Route a query to the appropriate solving strategy

        Args:
            query: The question/query to route
            dataset: Dataset name if known
            context: Additional context if provided

        Returns:
            RoutingDecision with strategy details
        """
        logger.debug(f"Routing query: {query[:100]}...")

        # Primary routing: dataset-based (most reliable)
        if dataset:
            decision = self._route_by_dataset(dataset, query, context)
            if decision:
                logger.debug(f"Routed by dataset '{dataset}' to {decision.task_type}")
                return decision

        # Secondary routing: content-based analysis
        decision = self._route_by_content(query, context)
        logger.debug(f"Routed by content to {decision.task_type}")
        return decision

    def _route_by_dataset(self, dataset: str, query: str, context: Optional[str]) -> Optional[RoutingDecision]:
        """
        Route based on dataset name

        Args:
            dataset: Dataset name
            query: Query text
            context: Context if provided

        Returns:
            RoutingDecision or None if dataset not recognized
        """
        dataset_lower = dataset.lower()

        # Math datasets
        if any(math_ds in dataset_lower for math_ds in self.config.math_datasets):
            return RoutingDecision(
                task_type="math",
                use_search=False,
                prompt_style="math",
                decoding_strategy="self_consistency",
                num_samples=config.model.num_samples_math,
                temperature=config.model.temperature_math,
                reasoning="Dataset is mathematical"
            )

        # Multiple choice datasets
        if any(mc_ds in dataset_lower for mc_ds in self.config.multiple_choice_datasets):
            return RoutingDecision(
                task_type="multiple_choice",
                use_search=False,
                prompt_style="multiple_choice",
                decoding_strategy="deterministic",
                num_samples=1,
                temperature=config.model.temperature_qa,
                reasoning="Dataset is multiple choice"
            )

        # Multi-hop QA datasets
        if any(multihop_ds in dataset_lower for multihop_ds in self.config.multihop_datasets):
            return RoutingDecision(
                task_type="multihop",
                use_search=True,
                prompt_style="multihop_qa",
                decoding_strategy="deterministic",
                num_samples=1,
                temperature=config.model.temperature_qa,
                reasoning="Dataset requires multi-hop reasoning"
            )

        # Single-hop QA datasets
        if any(qa_ds in dataset_lower for qa_ds in self.config.qa_datasets):
            # Check if context is provided (reading comprehension vs open-domain)
            use_search = context is None
            return RoutingDecision(
                task_type="qa",
                use_search=use_search,
                prompt_style="qa" if context else "open_qa",
                decoding_strategy="deterministic",
                num_samples=1,
                temperature=config.model.temperature_qa,
                reasoning="Dataset is question-answering" + (" with context" if context else " without context")
            )

        return None

    def _route_by_content(self, query: str, context: Optional[str]) -> RoutingDecision:
        """
        Route based on query content analysis

        Args:
            query: Query text
            context: Context if provided

        Returns:
            RoutingDecision
        """
        query_lower = query.lower()

        # Check for mathematical content
        math_score = self._compute_math_score(query_lower)
        qa_score = self._compute_qa_score(query_lower)
        multihop_score = self._compute_multihop_score(query_lower)

        logger.debug(f"Content scores - Math: {math_score}, QA: {qa_score}, Multihop: {multihop_score}")

        # Decision based on highest score
        if math_score > max(qa_score, multihop_score):
            return RoutingDecision(
                task_type="math",
                use_search=False,
                prompt_style="math",
                decoding_strategy="self_consistency",
                num_samples=config.model.num_samples_math,
                temperature=config.model.temperature_math,
                reasoning=f"Content analysis: math score {math_score:.2f}",
                confidence=math_score
            )

        elif multihop_score > qa_score:
            return RoutingDecision(
                task_type="multihop",
                use_search=True,
                prompt_style="multihop_qa",
                decoding_strategy="deterministic",
                num_samples=1,
                temperature=config.model.temperature_qa,
                reasoning=f"Content analysis: multihop score {multihop_score:.2f}",
                confidence=multihop_score
            )

        else:
            # Default to QA
            use_search = context is None
            return RoutingDecision(
                task_type="qa",
                use_search=use_search,
                prompt_style="qa" if context else "open_qa",
                decoding_strategy="deterministic",
                num_samples=1,
                temperature=config.model.temperature_qa,
                reasoning=f"Content analysis: QA score {qa_score:.2f}",
                confidence=qa_score
            )

    def _compute_math_score(self, query: str) -> float:
        """
        Compute mathematical content score

        Args:
            query: Query text (lowercase)

        Returns:
            Math score (0-1)
        """
        score = 0.0

        # Math keywords
        math_keyword_matches = sum(1 for keyword in self.math_keywords if keyword in query)
        score += min(math_keyword_matches * 0.3, 0.6)

        # Mathematical symbols and expressions
        if self.math_pattern.search(query):
            score += 0.4

        # Numbers and quantities
        number_matches = len(self.number_pattern.findall(query))
        score += min(number_matches * 0.1, 0.3)

        # Mathematical operations
        if any(op in query for op in ['=', '+', '-', '*', '/', '^', 'solve', 'find', 'calculate']):
            score += 0.2

        # AIME/contest math indicators
        if any(indicator in query for indicator in ['aime', 'contest', 'olympiad', 'competition']):
            score += 0.3

        return min(score, 1.0)

    def _compute_qa_score(self, query: str) -> float:
        """
        Compute QA content score

        Args:
            query: Query text (lowercase)

        Returns:
            QA score (0-1)
        """
        score = 0.0

        # Question words
        qa_keyword_matches = sum(1 for keyword in self.qa_keywords if query.startswith(keyword))
        score += min(qa_keyword_matches * 0.4, 0.8)

        # Question mark
        if query.endswith('?'):
            score += 0.3

        # Proper nouns (entities)
        if self.proper_noun_pattern.search(query):
            score += 0.2

        # Factual indicators
        if any(indicator in query for indicator in ['name', 'capital', 'president', 'born', 'invented']):
            score += 0.3

        return min(score, 1.0)

    def _compute_multihop_score(self, query: str) -> float:
        """
        Compute multi-hop reasoning score

        Args:
            query: Query text (lowercase)

        Returns:
            Multihop score (0-1)
        """
        score = 0.0

        # Conjunctions indicating multiple facts needed
        conjunctions = ['and', 'but', 'also', 'both', 'either', 'neither']
        conjunction_matches = sum(1 for conj in conjunctions if conj in query.split())
        score += min(conjunction_matches * 0.3, 0.6)

        # Multiple question words
        question_word_count = sum(1 for qw in ['who', 'what', 'where', 'when', 'which', 'how'] if qw in query)
        if question_word_count >= 2:
            score += 0.4

        # Comparative/relational words
        if any(rel in query for rel in ['compared to', 'relationship', 'connection', 'between', 'among']):
            score += 0.3

        # Complex query structure
        if len(query.split()) > 15:  # Long queries often need multiple hops
            score += 0.2

        # Temporal/causal indicators
        if any(temp in query for temp in ['before', 'after', 'during', 'because', 'therefore', 'thus']):
            score += 0.2

        return min(score, 1.0)

    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        # Mathematical expressions
        self.math_pattern = re.compile(r'[\d\w]*[\+\-\*\/\^\=][\d\w]*|\\[a-zA-Z]+\{.*?\}|[0-9]+[\.][0-9]+')

        # Numbers
        self.number_pattern = re.compile(r'\d+(?:\.\d+)?')

        # Proper nouns (capitalized words, excluding sentence starts)
        self.proper_noun_pattern = re.compile(r'(?<!^)(?<!\. )[A-Z][a-z]+')

    def explain_routing(self, decision: RoutingDecision) -> str:
        """
        Generate explanation for routing decision

        Args:
            decision: RoutingDecision to explain

        Returns:
            Human-readable explanation
        """
        explanation = f"Routing Decision: {decision.task_type.upper()}\n"
        explanation += f"Reason: {decision.reasoning}\n"
        explanation += f"Confidence: {decision.confidence:.2f}\n"
        explanation += f"Strategy: {decision.decoding_strategy} with {decision.num_samples} samples\n"
        explanation += f"Temperature: {decision.temperature}\n"
        explanation += f"Search enabled: {'Yes' if decision.use_search else 'No'}\n"
        explanation += f"Prompt style: {decision.prompt_style}"

        return explanation

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions (would need to track over time)

        Returns:
            Routing statistics
        """
        # This would be implemented with actual tracking
        return {
            "total_routed": 0,
            "math_routed": 0,
            "qa_routed": 0,
            "multihop_routed": 0,
            "search_enabled": 0
        }

    def update_routing_rules(self, new_rules: Dict[str, Any]):
        """
        Update routing rules dynamically

        Args:
            new_rules: Dictionary of new rules to apply
        """
        logger.info(f"Updating routing rules: {new_rules}")

        if "math_keywords" in new_rules:
            self.math_keywords.update(new_rules["math_keywords"])

        if "qa_keywords" in new_rules:
            self.qa_keywords.update(new_rules["qa_keywords"])

        # Recompile patterns if needed
        self._compile_patterns()

    def validate_routing(self, query: str, expected_type: str, dataset: Optional[str] = None) -> bool:
        """
        Validate routing decision against expected type

        Args:
            query: Query to route
            expected_type: Expected task type
            dataset: Dataset name if known

        Returns:
            True if routing matches expected type
        """
        decision = self.route(query, dataset)
        return decision.task_type == expected_type