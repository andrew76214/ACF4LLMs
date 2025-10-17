"""
A2A Pipeline - Agent-to-Agent QA and Math Reasoning System

This package implements a comprehensive pipeline for question-answering and mathematical
reasoning tasks using Qwen3-4B-Thinking model with advanced retrieval and evaluation.

Architecture:
- Router Agent: Classifies tasks and routes to appropriate solver
- Search Agent: Web search and retrieval using Tavily + BGE embeddings
- Solver Agent: Core reasoning using Qwen3-4B-Thinking with chain-of-thought
- Judge Agent: Evaluation using Math-Verify and official dataset metrics

Supports:
- Math: GSM8K, AIME, MATH datasets with self-consistency decoding
- QA: SQuAD, Natural Questions, HotpotQA with retrieval augmentation
"""

__version__ = "0.1.0"
__author__ = "A2A Pipeline Team"

from .agents import RouterAgent, SearchAgent, SolverAgent, JudgeAgent
from .models import ModelManager
from .evaluation import Evaluator

__all__ = [
    "RouterAgent",
    "SearchAgent",
    "SolverAgent",
    "JudgeAgent",
    "ModelManager",
    "Evaluator"
]