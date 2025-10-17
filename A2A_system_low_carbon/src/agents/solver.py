"""
Solver Agent for A2A Pipeline
Core reasoning agent using Qwen3-4B-Thinking with task-specific prompts
"""

import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from models.vllm_client import VLLMClient
from models.model_manager import ModelManager
from agents.search import RetrievalContext
from agents.router import RoutingDecision
from configs.config import config
from utils.logging import get_logger, PerformanceLogger
from utils.text_processing import extract_final_answer, split_reasoning_and_answer

logger = get_logger("solver_agent")

@dataclass
class SolverResponse:
    """Response from the solver agent"""
    answer: str
    reasoning: Optional[str]
    confidence: float
    raw_output: str
    metadata: Dict[str, Any]

class PromptTemplates:
    """Centralized prompt templates for different task types"""

    @staticmethod
    def get_math_system_prompt() -> str:
        return """You are a careful mathematician. You will think step by step and solve mathematical problems accurately.

Instructions:
- Show your reasoning in a <think> block
- Put your final numerical answer in \\boxed{} format
- For AIME problems, answers should be integers from 0 to 999
- Do not add extra text outside the boxed answer
- Be precise with calculations and check your work

Example:
<think>
Let me solve this step by step...
[reasoning steps]
So the answer is 42.
</think>

\\boxed{42}"""

    @staticmethod
    def get_qa_system_prompt() -> str:
        return """You are a helpful question-answering assistant. Answer questions accurately and concisely using the provided context.

Instructions:
- Use only information from the provided context
- If the answer is not in the context, respond with "I don't have enough information to answer this question"
- Be direct and factual
- For questions requiring specific facts, provide exact information"""

    @staticmethod
    def get_multihop_system_prompt() -> str:
        return """You are an expert at multi-hop question answering. You need to combine information from multiple sources to answer complex questions.

Instructions:
- Analyze all provided passages carefully
- Identify the key facts needed to answer the question
- Combine information from different passages when necessary
- Think step by step, but provide only the final answer

IMPORTANT: Provide ONLY the direct answer. Do not include explanations, reasoning, or JSON format.

Examples:
- Question: "Were Scott Derrickson and Ed Wood of the same nationality?" → Answer: "yes"
- Question: "What government position was held by the woman who portrayed Corliss Archer?" → Answer: "Chief of Protocol"
- Question: "What is the capital of France?" → Answer: "Paris"""

    @staticmethod
    def get_open_qa_system_prompt() -> str:
        return """You are a knowledgeable assistant that answers questions using the provided search results.

Instructions:
- Use the retrieved information to answer the question
- Be accurate and cite relevant sources when possible
- If the retrieved information is insufficient, say so
- Provide a clear, direct answer"""

    @staticmethod
    def get_multiple_choice_system_prompt() -> str:
        return """You are an expert at multiple choice questions. Answer based on your knowledge and reasoning.

Instructions:
- Read the question and all answer choices carefully
- Think through the problem step by step in a <think> block
- Select the best answer from the given options
- Provide ONLY the letter of your choice (A, B, C, D, or E) as your final answer

Example:
<think>
Let me analyze each option...
Option A is incorrect because...
Option B makes sense because...
</think>

B"""

class SolverAgent:
    """
    Solver Agent that generates answers using Qwen3-4B-Thinking
    Handles different task types with specialized prompts
    """

    def __init__(self, model_client: Optional[Union[VLLMClient, ModelManager]] = None):
        """
        Initialize SolverAgent

        Args:
            model_client: Model client (VLLMClient or ModelManager)
        """
        self.model_client = model_client
        self.prompt_templates = PromptTemplates()

        # Initialize model client if not provided
        if self.model_client is None:
            self._initialize_default_client()

    def _initialize_default_client(self):
        """Initialize default model client"""
        try:
            # Try VLLMClient first (assumes server is running)
            self.model_client = VLLMClient()
            if not self.model_client.health_check():
                logger.warning("VLLMClient health check failed, initializing ModelManager")
                self.model_client = ModelManager()
                if not self.model_client.load_model():
                    raise RuntimeError("Failed to load model")
        except Exception as e:
            logger.error(f"Failed to initialize model client: {e}")
            raise

    def solve(
        self,
        question: str,
        routing_decision: RoutingDecision,
        context: Optional[RetrievalContext] = None,
        **kwargs
    ) -> SolverResponse:
        """
        Solve a question using the appropriate strategy

        Args:
            question: Question to solve
            routing_decision: Routing decision with task type and parameters
            context: Retrieved context if needed
            **kwargs: Additional parameters

        Returns:
            SolverResponse with answer and metadata
        """
        logger.info(f"Solving {routing_decision.task_type} question: {question[:100]}...")

        try:
            with PerformanceLogger(f"solving {routing_decision.task_type} question"):
                # Build prompt based on task type
                system_prompt, user_prompt = self._build_prompt(
                    question, routing_decision.task_type, context
                )

                # Generate response
                if routing_decision.decoding_strategy == "self_consistency" and routing_decision.num_samples > 1:
                    response = self._solve_with_self_consistency(
                        system_prompt, user_prompt, routing_decision
                    )
                else:
                    response = self._solve_single(
                        system_prompt, user_prompt, routing_decision
                    )

                logger.info(f"Generated answer: {response.answer[:100]}...")
                return response

        except Exception as e:
            logger.error(f"Solving failed: {e}")
            return SolverResponse(
                answer="",
                reasoning="",
                confidence=0.0,
                raw_output="",
                metadata={"error": str(e)}
            )

    def _build_prompt(
        self,
        question: str,
        task_type: str,
        context: Optional[RetrievalContext] = None
    ) -> tuple[str, str]:
        """
        Build system and user prompts for the given task type

        Args:
            question: Question to solve
            task_type: Type of task (math, qa, multihop, open_qa)
            context: Retrieved context if available

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Get system prompt based on task type
        if task_type == "math":
            system_prompt = self.prompt_templates.get_math_system_prompt()
            user_prompt = question
        elif task_type == "qa":
            system_prompt = self.prompt_templates.get_qa_system_prompt()
            if context and context.passages:
                context_str = self._format_context_for_qa(context)
                user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
            else:
                user_prompt = f"Question: {question}"
        elif task_type == "multihop":
            system_prompt = self.prompt_templates.get_multihop_system_prompt()
            if context and context.passages:
                context_str = self._format_context_for_multihop(context)
                user_prompt = f"Passages:\n{context_str}\n\nQuestion: {question}"
            else:
                user_prompt = f"Question: {question}"
        elif task_type == "open_qa":
            system_prompt = self.prompt_templates.get_open_qa_system_prompt()
            if context and context.passages:
                context_str = self._format_context_for_open_qa(context)
                user_prompt = f"Search Results:\n{context_str}\n\nQuestion: {question}"
            else:
                user_prompt = f"Question: {question}"
        elif task_type == "multiple_choice":
            system_prompt = self.prompt_templates.get_multiple_choice_system_prompt()
            user_prompt = question  # Question already contains choices
        else:
            # Default to QA
            system_prompt = self.prompt_templates.get_qa_system_prompt()
            user_prompt = f"Question: {question}"

        return system_prompt, user_prompt

    def _format_context_for_qa(self, context: RetrievalContext) -> str:
        """Format context for single-hop QA tasks"""
        if not context.passages:
            return "No context provided."

        # For QA, we typically use the first passage or combine short passages
        if len(context.passages) == 1:
            return context.passages[0].content
        else:
            # Combine multiple passages
            combined = []
            for i, passage in enumerate(context.passages[:3]):  # Limit to top 3
                combined.append(f"[{i+1}] {passage.content}")
            return "\n\n".join(combined)

    def _format_context_for_multihop(self, context: RetrievalContext) -> str:
        """Format context for multi-hop QA tasks"""
        if not context.passages:
            return "No passages provided."

        formatted_passages = []
        for i, passage in enumerate(context.passages):
            passage_text = f"{i+1}. Title: {passage.title}\nContent: {passage.content}\nSource: {passage.url}"
            formatted_passages.append(passage_text)

        return "\n\n".join(formatted_passages)

    def _format_context_for_open_qa(self, context: RetrievalContext) -> str:
        """Format context for open-domain QA tasks"""
        if not context.passages:
            return "No search results found."

        formatted_results = []
        for i, passage in enumerate(context.passages):
            result_text = f"Result {i+1}:\nTitle: {passage.title}\nContent: {passage.content}\nURL: {passage.url}"
            formatted_results.append(result_text)

        return "\n\n".join(formatted_results)

    def _solve_single(
        self,
        system_prompt: str,
        user_prompt: str,
        routing_decision: RoutingDecision
    ) -> SolverResponse:
        """
        Solve with single generation

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            routing_decision: Routing parameters

        Returns:
            SolverResponse
        """
        try:
            # Generate response
            if isinstance(self.model_client, VLLMClient):
                response = self.model_client.generate_with_reasoning(
                    prompt=user_prompt,
                    system_message=system_prompt,
                    temperature=routing_decision.temperature,
                    max_tokens=config.model.max_tokens_qa if routing_decision.task_type != "math" else config.model.max_tokens_math
                )
                raw_output = response.get("content", "")
                reasoning = response.get("reasoning", "")
            else:
                # ModelManager interface
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                results = self.model_client.generate(
                    prompt=self._messages_to_prompt(messages),
                    temperature=routing_decision.temperature,
                    max_tokens=config.model.max_tokens_qa if routing_decision.task_type != "math" else config.model.max_tokens_math,
                    num_samples=1
                )
                if results:
                    result = results[0]
                    raw_output = result.get("text", "")
                    reasoning = result.get("reasoning", "")
                else:
                    raw_output = reasoning = ""

            # Parse the answer
            final_answer = extract_final_answer(raw_output, routing_decision.task_type)
            if not final_answer:
                final_answer = raw_output.strip()

            return SolverResponse(
                answer=final_answer,
                reasoning=reasoning,
                confidence=1.0,
                raw_output=raw_output,
                metadata={
                    "task_type": routing_decision.task_type,
                    "temperature": routing_decision.temperature,
                    "num_samples": 1
                }
            )

        except Exception as e:
            logger.error(f"Single solve failed: {e}")
            return SolverResponse(
                answer="",
                reasoning="",
                confidence=0.0,
                raw_output="",
                metadata={"error": str(e)}
            )

    def _solve_with_self_consistency(
        self,
        system_prompt: str,
        user_prompt: str,
        routing_decision: RoutingDecision
    ) -> SolverResponse:
        """
        Solve with self-consistency (multiple samples + majority vote)

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            routing_decision: Routing parameters

        Returns:
            SolverResponse with majority vote answer
        """
        try:
            logger.info(f"Generating {routing_decision.num_samples} samples for self-consistency")

            if isinstance(self.model_client, VLLMClient):
                # Use VLLMClient's self-consistency method
                response = self.model_client.self_consistency_generate(
                    prompt=user_prompt,
                    system_message=system_prompt,
                    num_samples=routing_decision.num_samples,
                    temperature=routing_decision.temperature
                )

                return SolverResponse(
                    answer=response["majority_answer"],
                    reasoning=response["samples"][0].get("reasoning", "") if response["samples"] else "",
                    confidence=response["agreement"],
                    raw_output=json.dumps(response["samples"]),
                    metadata={
                        "task_type": routing_decision.task_type,
                        "num_samples": routing_decision.num_samples,
                        "agreement": response["agreement"],
                        "all_answers": response["answers"]
                    }
                )

            else:
                # Use ModelManager for batch generation
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt = self._messages_to_prompt(messages)

                results = self.model_client.generate(
                    prompt=prompt,
                    temperature=routing_decision.temperature,
                    max_tokens=config.model.max_tokens_math,
                    num_samples=routing_decision.num_samples
                )

                if not results:
                    raise ValueError("No results generated")

                # Extract answers from all samples
                answers = []
                reasonings = []
                for result in results:
                    raw_output = result.get("text", "")
                    reasoning = result.get("reasoning", "")
                    answer = extract_final_answer(raw_output, routing_decision.task_type)
                    if answer:
                        answers.append(answer)
                        reasonings.append(reasoning)

                if not answers:
                    raise ValueError("No valid answers extracted from samples")

                # Majority vote
                from utils.metrics import majority_vote
                majority_answer = majority_vote(answers)

                # Calculate agreement
                agreement = answers.count(majority_answer) / len(answers)

                return SolverResponse(
                    answer=majority_answer,
                    reasoning=reasonings[0] if reasonings else "",
                    confidence=agreement,
                    raw_output=json.dumps([r.get("text", "") for r in results]),
                    metadata={
                        "task_type": routing_decision.task_type,
                        "num_samples": len(results),
                        "agreement": agreement,
                        "all_answers": answers
                    }
                )

        except Exception as e:
            logger.error(f"Self-consistency solve failed: {e}")
            # Fallback to single solve
            return self._solve_single(system_prompt, user_prompt, routing_decision)

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert messages format to single prompt for ModelManager

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts) + "\n\nAssistant:"

    def batch_solve(
        self,
        questions: List[str],
        routing_decisions: List[RoutingDecision],
        contexts: Optional[List[RetrievalContext]] = None
    ) -> List[SolverResponse]:
        """
        Solve multiple questions in batch

        Args:
            questions: List of questions
            routing_decisions: List of routing decisions
            contexts: List of contexts (optional)

        Returns:
            List of SolverResponses
        """
        if contexts is None:
            contexts = [None] * len(questions)

        results = []
        for i, (question, routing_decision, context) in enumerate(zip(questions, routing_decisions, contexts)):
            try:
                logger.info(f"Solving batch question {i+1}/{len(questions)}")
                result = self.solve(question, routing_decision, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch solve failed for question {i+1}: {e}")
                results.append(SolverResponse(
                    answer="",
                    reasoning="",
                    confidence=0.0,
                    raw_output="",
                    metadata={"error": str(e)}
                ))

        return results

    def get_solver_info(self) -> Dict[str, Any]:
        """Get information about the solver"""
        info = {
            "model_type": type(self.model_client).__name__,
            "available_task_types": ["math", "qa", "multihop", "open_qa"]
        }

        if hasattr(self.model_client, 'get_model_info'):
            info.update(self.model_client.get_model_info())

        return info