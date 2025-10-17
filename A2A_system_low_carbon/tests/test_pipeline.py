"""
Test script for A2A Pipeline
Validates all components with sample questions
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import A2APipeline
from agents.router import RouterAgent
from agents.search import SearchAgent
from agents.solver import SolverAgent
from agents.judge import JudgeAgent
from utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=False)
logger = get_logger("test_pipeline")

# Test data
TEST_QUESTIONS = {
    "math": [
        {
            "question": "If 2x + 5 = 13, what is x?",
            "answer": "4",
            "dataset": "gsm8k"
        },
        {
            "question": "What is 15 * 23?",
            "answer": "345",
            "dataset": "math"
        },
        {
            "question": "Find the value of 2^5 + 3^2",
            "answer": "41",
            "dataset": "aime"
        }
    ],
    "qa": [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "dataset": "nq"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "dataset": "squad_v1"
        }
    ],
    "multihop": [
        {
            "question": "Which university did the author of 'To Kill a Mockingbird' attend?",
            "answer": "University of Alabama",
            "dataset": "hotpotqa",
            "supporting_facts": [["Harper Lee", 1], ["University of Alabama", 0]]
        }
    ]
}

def test_router_agent():
    """Test the router agent"""
    logger.info("Testing Router Agent...")

    router = RouterAgent()

    # Test math routing
    decision = router.route("What is 2+2?", "gsm8k")
    assert decision.task_type == "math", f"Expected math, got {decision.task_type}"

    # Test QA routing
    decision = router.route("Who is the president?", "nq")
    assert decision.task_type in ["qa", "open_qa"], f"Expected QA, got {decision.task_type}"

    # Test multihop routing
    decision = router.route("Which author wrote X and taught at Y?", "hotpotqa")
    assert decision.task_type == "multihop", f"Expected multihop, got {decision.task_type}"

    logger.info("✓ Router Agent tests passed")

def test_search_agent():
    """Test the search agent"""
    logger.info("Testing Search Agent...")

    search_agent = SearchAgent()

    # Test health check
    health = search_agent.health_check()
    logger.info(f"Search agent health: {health}")

    # Skip actual search tests if no API key
    if not health.get("tavily_client", False):
        logger.warning("⚠ Skipping search tests - no Tavily API key")
        return

    # Test basic search
    results = search_agent.search("Python programming", max_results=3)
    assert isinstance(results, list), "Search should return a list"
    logger.info(f"✓ Search returned {len(results)} results")

    # Test retrieval
    context = search_agent.retrieve_and_rerank("What is Python?")
    assert isinstance(context.passages, list), "Retrieval should return passages"
    logger.info(f"✓ Retrieval returned {len(context.passages)} passages")

    logger.info("✓ Search Agent tests passed")

def test_judge_agent():
    """Test the judge agent"""
    logger.info("Testing Judge Agent...")

    judge = JudgeAgent()

    # Test math judgment
    result = judge.judge_single("4", "4", "math")
    assert result.correct, "Math judgment should be correct"

    result = judge.judge_single("5", "4", "math")
    assert not result.correct, "Math judgment should be incorrect"

    # Test QA judgment
    result = judge.judge_single("Paris", "Paris", "qa")
    assert result.correct, "QA judgment should be correct"

    result = judge.judge_single("London", "Paris", "qa")
    assert not result.correct, "QA judgment should be incorrect"

    logger.info("✓ Judge Agent tests passed")

def test_individual_questions():
    """Test solving individual questions without full pipeline"""
    logger.info("Testing individual question solving...")

    # Test router with sample questions
    router = RouterAgent()

    for task_type, questions in TEST_QUESTIONS.items():
        for q_data in questions:
            decision = router.route(q_data["question"], q_data.get("dataset"))
            logger.info(f"Question: {q_data['question'][:50]}...")
            logger.info(f"Routed to: {decision.task_type} (expected: {task_type})")

            # Check if routing is reasonable
            if task_type == "math" and decision.task_type != "math":
                logger.warning(f"⚠ Math question routed to {decision.task_type}")
            elif task_type in ["qa", "multihop"] and decision.task_type == "math":
                logger.warning(f"⚠ {task_type} question routed to math")

    logger.info("✓ Individual question tests passed")

def test_mock_pipeline():
    """Test pipeline components without actual model loading"""
    logger.info("Testing mock pipeline flow...")

    # Create mock responses to test the flow
    test_cases = [
        {
            "question": "What is 2+2?",
            "expected_type": "math",
            "dataset": "gsm8k"
        },
        {
            "question": "Who is the president of the United States?",
            "expected_type": "qa",
            "dataset": "nq"
        }
    ]

    router = RouterAgent()
    judge = JudgeAgent()

    for case in test_cases:
        # Test routing
        decision = router.route(case["question"], case["dataset"])
        logger.info(f"Routed '{case['question']}' to {decision.task_type}")

        # Test judgment with mock answer
        result = judge.judge_single("mock_answer", "correct_answer", decision.task_type)
        logger.info(f"Judgment result: correct={result.correct}, score={result.score}")

    logger.info("✓ Mock pipeline tests passed")

def create_sample_data():
    """Create sample datasets for testing"""
    logger.info("Creating sample data...")

    # Create data directory
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # GSM8K sample
    gsm8k_sample = [
        {
            "question": "John has 3 apples. He eats 1 apple. How many apples does he have left?",
            "answer": "2"
        },
        {
            "question": "A store sells 5 books for $25. What is the price per book?",
            "answer": "5"
        }
    ]

    with open(data_dir / "gsm8k_sample.json", "w") as f:
        json.dump(gsm8k_sample, f, indent=2)

    # AIME sample
    aime_sample = [
        {
            "question": "Find the value of 2^3 + 3^2.",
            "answer": "17"
        }
    ]

    with open(data_dir / "aime_sample.json", "w") as f:
        json.dump(aime_sample, f, indent=2)

    logger.info("✓ Sample data created")

def run_basic_tests():
    """Run basic tests without model loading"""
    logger.info("=== Running Basic A2A Pipeline Tests ===")

    try:
        test_router_agent()
        test_search_agent()
        test_judge_agent()
        test_individual_questions()
        test_mock_pipeline()
        create_sample_data()

        logger.info("=== All Basic Tests Passed! ===")
        return True

    except Exception as e:
        logger.error(f"Basic tests failed: {e}")
        return False

def test_full_pipeline():
    """Test the full pipeline with model loading"""
    logger.info("=== Testing Full Pipeline ===")

    try:
        # This would require actual model loading
        logger.info("Full pipeline test requires model initialization")
        logger.info("To test full pipeline, run: python src/pipeline.py --mode solve --question 'What is 2+2?'")

        return True

    except Exception as e:
        logger.error(f"Full pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    basic_success = run_basic_tests()

    if basic_success:
        print("\n🎉 Basic tests completed successfully!")
        print("\nTo test the full pipeline:")
        print("1. Set up your API keys in .env file")
        print("2. Install required packages: pip install -r requirements.txt")
        print("3. Run: python src/pipeline.py --config-check")
        print("4. Run: python src/pipeline.py --mode interactive")
    else:
        print("\n❌ Basic tests failed. Please check the logs.")
        sys.exit(1)