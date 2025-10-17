#!/usr/bin/env python3
"""
Test Qwen3-4B-Thinking model with actual inference
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ["TAVILY_API_KEY"] = "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"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/home/tim/.cache/huggingface"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.logging import setup_logger, get_logger

# Setup logging
setup_logger(level="INFO", log_to_console=True, log_to_file=False)
logger = get_logger("qwen_test")

def test_qwen_download_and_tokenizer():
    """Test downloading Qwen model and tokenizer"""
    logger.info("=== Testing Qwen3-4B-Thinking Download & Tokenizer ===")

    model_name = "unsloth/Qwen3-4B-Thinking-2507"

    try:
        # Test tokenizer first (small download)
        logger.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Test tokenization
        test_text = "What is 2+2?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)

        logger.info(f"Test tokenization:")
        logger.info(f"  Original: {test_text}")
        logger.info(f"  Tokens: {len(tokens)} tokens")
        logger.info(f"  Decoded: {decoded}")

        return True, tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return False, None

def test_qwen_model_loading():
    """Test loading the actual Qwen model"""
    logger.info("=== Testing Qwen3-4B-Thinking Model Loading ===")

    model_name = "unsloth/Qwen3-4B-Thinking-2507"

    try:
        # Load model with appropriate settings for 24GB GPU
        logger.info(f"Loading model {model_name} (this may take several minutes)...")
        logger.info("Model will be loaded in FP16 to fit in 24GB VRAM...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        logger.info(f"Model loaded successfully!")
        logger.info(f"Model device: {model.device}")
        logger.info(f"Model dtype: {model.dtype}")

        # Get memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")

        return True, model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False, None

def test_qwen_inference(model, tokenizer):
    """Test actual inference with Qwen model"""
    logger.info("=== Testing Qwen3-4B-Thinking Inference ===")

    try:
        # Math problem test
        math_prompt = "Solve step by step: What is 15 * 23? Put your final answer in \\boxed{}."

        logger.info(f"Testing math inference...")
        logger.info(f"Prompt: {math_prompt}")

        # Tokenize input
        inputs = tokenizer(math_prompt, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new tokens (response)
        input_length = len(inputs.input_ids[0])
        new_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        logger.info("Generated response:")
        logger.info(f"  Full output length: {len(response)} chars")
        logger.info(f"  Generated text: {generated_text}")

        # Check if it contains reasoning or boxed answer
        has_reasoning = "<think>" in generated_text or "step" in generated_text.lower()
        has_boxed = "\\boxed{" in generated_text

        logger.info(f"  Contains reasoning: {has_reasoning}")
        logger.info(f"  Contains boxed answer: {has_boxed}")

        return True, generated_text

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return False, ""

def run_qwen_comprehensive_test():
    """Run comprehensive Qwen model test"""
    logger.info("🧠 Starting Qwen3-4B-Thinking Comprehensive Test")
    logger.info("=" * 60)

    results = {}

    # Step 1: Test tokenizer
    logger.info("Step 1: Testing tokenizer download and functionality...")
    success, tokenizer = test_qwen_download_and_tokenizer()
    results["tokenizer"] = success

    if not success:
        logger.error("❌ Tokenizer test failed - aborting")
        return results

    logger.info("✅ Tokenizer test passed!")
    print()

    # Step 2: Test model loading
    logger.info("Step 2: Testing model download and loading...")
    success, model = test_qwen_model_loading()
    results["model_loading"] = success

    if not success:
        logger.error("❌ Model loading failed - aborting")
        return results

    logger.info("✅ Model loading test passed!")
    print()

    # Step 3: Test inference
    logger.info("Step 3: Testing model inference...")
    success, response = test_qwen_inference(model, tokenizer)
    results["inference"] = success
    results["response"] = response

    if success:
        logger.info("✅ Model inference test passed!")
    else:
        logger.error("❌ Model inference failed")

    print()
    logger.info("=" * 60)
    logger.info("🎯 QWEN MODEL TEST SUMMARY")
    logger.info("=" * 60)

    passed_tests = sum(1 for k, v in results.items() if k != "response" and v)
    total_tests = len([k for k in results.keys() if k != "response"])

    logger.info(f"Tokenizer: {'✅ PASS' if results['tokenizer'] else '❌ FAIL'}")
    logger.info(f"Model Loading: {'✅ PASS' if results['model_loading'] else '❌ FAIL'}")
    logger.info(f"Inference: {'✅ PASS' if results['inference'] else '❌ FAIL'}")
    logger.info("-" * 60)
    logger.info(f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")

    if passed_tests == total_tests:
        logger.info("🎉 ALL QWEN TESTS PASSED! Model is ready for full pipeline.")
        logger.info("\n🚀 READY FOR FULL PIPELINE TESTING:")
        logger.info("  python src/pipeline.py --mode solve --question 'What is 17*19?'")
        logger.info("  python src/pipeline.py --mode interactive")
    else:
        logger.info("⚠️ Some Qwen tests failed. Check GPU memory and model availability.")

    return results

if __name__ == "__main__":
    print("🧠 Qwen3-4B-Thinking Model Test")
    print("Testing actual model download, loading, and inference...")
    print("=" * 60)

    # Check prerequisites
    if not torch.cuda.is_available():
        print("❌ CUDA not available - GPU required for Qwen model")
        sys.exit(1)

    if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
        print("⚠️  GPU has less than 20GB VRAM - may not be sufficient for Qwen3-4B")

    # Run comprehensive test
    results = run_qwen_comprehensive_test()

    # Save results
    with open("qwen_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Detailed results saved to: qwen_test_results.json")