"""
Sequential Memory-Efficient Evaluator for A2A Pipeline
Only loads agents when router determines they are needed
"""

import json
import time
import gc
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from agents.router import RouterAgent
from agents.search import SearchAgent
from agents.solver import SolverAgent
from agents.judge import JudgeAgent
from models.vllm_client import VLLMClient
from models.model_manager import ModelManager
from utils.data_processing import load_dataset_by_name, save_results, sample_dataset
from utils.logging import get_logger, PerformanceLogger
from utils.metrics import format_metrics_report, compute_variance_metrics
from configs.config import config

logger = get_logger("sequential_evaluator")

class SequentialEvaluator:
    """
    Memory-efficient sequential evaluator that only loads agents when needed
    """

    def __init__(self, model_client: Optional[Any] = None):
        """
        Initialize Sequential Evaluator

        Args:
            model_client: Optional model client (will initialize default if None)
        """
        self.config = config
        self.model_client = model_client

        # Only initialize lightweight router upfront
        self.router = RouterAgent()

        # Agent instances - loaded on demand
        self.search_agent = None
        self.solver = None
        self.judge = None

        # Track what's currently loaded
        self.loaded_agents = set()

        logger.info("Sequential Memory-Efficient Evaluator initialized")

    def _clear_gpu_memory(self):
        """Clear GPU memory between operations"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def _load_search_agent(self):
        """Load search agent on demand"""
        if self.search_agent is None:
            logger.info("Loading SearchAgent on demand...")
            self.search_agent = SearchAgent()
            self.loaded_agents.add("search")
            logger.info("SearchAgent loaded")

    def _load_solver(self):
        """Load solver agent on demand"""
        if self.solver is None:
            logger.info("Loading SolverAgent on demand...")
            self.solver = SolverAgent(self.model_client)
            self.loaded_agents.add("solver")
            logger.info("SolverAgent loaded")

    def _load_judge(self):
        """Load judge agent on demand"""
        if self.judge is None:
            logger.info("Loading JudgeAgent on demand...")
            self.judge = JudgeAgent()
            self.loaded_agents.add("judge")
            logger.info("JudgeAgent loaded")

    def _unload_search_agent(self):
        """Unload search agent to free memory"""
        if self.search_agent is not None:
            logger.info("Unloading SearchAgent to free memory...")
            del self.search_agent
            self.search_agent = None
            self.loaded_agents.discard("search")
            self._clear_gpu_memory()
            logger.info("SearchAgent unloaded")

    def _unload_solver(self):
        """Unload solver agent to free memory"""
        if self.solver is not None:
            logger.info("Unloading SolverAgent to free memory...")
            del self.solver
            self.solver = None
            self.loaded_agents.discard("solver")
            self._clear_gpu_memory()
            logger.info("SolverAgent unloaded")

    def _unload_judge(self):
        """Unload judge agent to free memory"""
        if self.judge is not None:
            logger.info("Unloading JudgeAgent to free memory...")
            del self.judge
            self.judge = None
            self.loaded_agents.discard("judge")
            self._clear_gpu_memory()
            logger.info("JudgeAgent unloaded")

    def solve_question(self, question: str, dataset: Optional[str] = None) -> Dict[str, Any]:
        """
        Solve a single question using sequential agent loading

        Args:
            question: Question to solve
            dataset: Dataset type for routing

        Returns:
            Solution result
        """
        start_time = time.time()

        try:
            # Step 1: Route the question (router is always loaded)
            logger.info(f"Routing question: {question[:100]}...")
            routing_decision = self.router.route(question, dataset, None)
            logger.info(f"Routed to task_type: {routing_decision.task_type}, use_search: {routing_decision.use_search}")

            # Step 2: Load and use search agent if needed
            retrieval_context = None
            if routing_decision.use_search:
                self._load_search_agent()

                if routing_decision.task_type == "multihop":
                    retrieval_context = self.search_agent.multi_hop_search(question)
                else:
                    retrieval_context = self.search_agent.retrieve_and_rerank(question)

                logger.info(f"Retrieved {len(retrieval_context.passages)} passages")

                # Unload search agent after use to free memory
                self._unload_search_agent()

            # Step 3: Load solver and solve
            self._load_solver()
            solver_response = self.solver.solve(question, routing_decision, retrieval_context)
            logger.info(f"Generated answer: {solver_response.answer[:100]}...")

            # Keep solver loaded as it's the main model we need

            # Step 4: Load judge for evaluation (lightweight)
            self._load_judge()

            # Return result
            result = {
                "answer": solver_response.answer,
                "task_type": routing_decision.task_type,
                "reasoning": solver_response.reasoning,
                "confidence": solver_response.confidence,
                "solve_time": time.time() - start_time,
                "loaded_agents": list(self.loaded_agents),
                "retrieval_used": routing_decision.use_search,
                "passages_retrieved": len(retrieval_context.passages) if retrieval_context else 0
            }

            return result

        except Exception as e:
            logger.error(f"Question solving failed: {e}")
            return {
                "error": str(e),
                "solve_time": time.time() - start_time,
                "loaded_agents": list(self.loaded_agents)
            }

    def evaluate_single_question(
        self,
        question: str,
        gold_answer: Any,
        dataset: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a single question through sequential pipeline

        Args:
            question: Question to evaluate
            gold_answer: Ground truth answer
            dataset: Dataset name for routing
            context: Optional context
            **kwargs: Additional parameters

        Returns:
            Evaluation results
        """
        start_time = time.time()

        try:
            # Solve the question
            solve_result = self.solve_question(question, dataset)

            if "error" in solve_result:
                return solve_result

            # Judge the result
            self._load_judge()
            judgment = self.judge.judge_single(
                solve_result["answer"],
                gold_answer,
                solve_result["task_type"]
            )

            # Compile final results
            result = {
                "question": question,
                "prediction": solve_result["answer"],
                "gold_answer": gold_answer,
                "correct": judgment.correct,
                "task_type": solve_result["task_type"],
                "reasoning": solve_result.get("reasoning", ""),
                "confidence": solve_result.get("confidence", 0.0),
                "total_time": time.time() - start_time,
                "solve_time": solve_result["solve_time"],
                "loaded_agents": list(self.loaded_agents),
                "retrieval_used": solve_result.get("retrieval_used", False),
                "passages_retrieved": solve_result.get("passages_retrieved", 0),
                "judgment": {
                    "correct": judgment.correct,
                    "score": judgment.score,
                    "explanation": judgment.explanation
                }
            }

            return result

        except Exception as e:
            logger.error(f"Single question evaluation failed: {e}")
            return {
                "question": question,
                "error": str(e),
                "total_time": time.time() - start_time,
                "loaded_agents": list(self.loaded_agents)
            }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the sequential pipeline

        Returns:
            Health check results
        """
        logger.info("Running sequential pipeline health check...")

        health = {
            "router": True,  # Always loaded
            "search": False,
            "solver": False,
            "judge": False,
            "timestamp": datetime.now().isoformat(),
            "loaded_agents": list(self.loaded_agents),
            "pipeline_test": {"success": False, "time": 0}
        }

        try:
            # Test with a simple question
            test_start = time.time()
            test_result = self.solve_question("What is 2+2?", "math")
            test_time = time.time() - test_start

            if "error" not in test_result:
                health["pipeline_test"] = {
                    "success": True,
                    "time": test_time,
                    "answer": test_result.get("answer", "")
                }

            # Update health status
            health["search"] = "search" in self.loaded_agents
            health["solver"] = "solver" in self.loaded_agents
            health["judge"] = "judge" in self.loaded_agents

            if health["pipeline_test"]["success"]:
                logger.info("Sequential pipeline health check passed")
            else:
                logger.warning("Sequential pipeline health check failed")

            return health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["error"] = str(e)
            return health

    def cleanup(self):
        """Clean up all loaded agents and free memory"""
        logger.info("Cleaning up sequential evaluator...")

        self._unload_search_agent()
        self._unload_solver()
        self._unload_judge()

        # Final memory cleanup
        self._clear_gpu_memory()

        logger.info("Sequential evaluator cleanup complete")

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory usage status"""
        status = {
            "loaded_agents": list(self.loaded_agents),
            "agent_count": len(self.loaded_agents)
        }

        if torch.cuda.is_available():
            status["gpu_memory"] = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "free_mb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**2
            }

        return status