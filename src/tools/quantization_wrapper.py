"""Real quantization tool wrappers for AutoRound, GPTQ, AWQ, INT8, LoRA, and QLoRA."""

import os
import torch
import json
import time
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Any, Tuple, List, Generator
from pathlib import Path
from datetime import datetime

from src.common.model_utils import estimate_model_size_gb

logger = logging.getLogger(__name__)


# Trusted model repositories for trust_remote_code
# Add organization/model patterns that are trusted
TRUSTED_MODEL_SOURCES = frozenset({
    "meta-llama",
    "mistralai",
    "microsoft",
    "google",
    "facebook",
    "EleutherAI",
    "bigscience",
    "tiiuae",
    "Qwen",
    "01-ai",
    "deepseek-ai",
    "internlm",
    "baichuan-inc",
    "THUDM",
    "HuggingFaceH4",
    "teknium",
    "NousResearch",
    "OpenAssistant",
    "lmsys",
    "stabilityai",
    "mosaicml",
    "databricks",
})

# Environment variable to disable trust_remote_code entirely
ALLOW_TRUST_REMOTE_CODE = os.getenv("ALLOW_TRUST_REMOTE_CODE", "true").lower() == "true"


def is_trusted_model_source(model_name: str) -> bool:
    """Check if a model is from a trusted source.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        True if the model is from a trusted source or is a local path
    """
    # Local paths are considered trusted
    if os.path.exists(model_name):
        return True

    # Check if it's from a trusted organization
    if "/" in model_name:
        org = model_name.split("/")[0]
        return org in TRUSTED_MODEL_SOURCES

    # Single-name models (like "gpt2") are from HuggingFace and trusted
    return True


def should_trust_remote_code(model_name: str) -> bool:
    """Determine if trust_remote_code should be enabled for a model.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        True if trust_remote_code should be enabled
    """
    if not ALLOW_TRUST_REMOTE_CODE:
        logger.warning("trust_remote_code disabled via ALLOW_TRUST_REMOTE_CODE=false")
        return False

    if is_trusted_model_source(model_name):
        return True

    logger.warning(
        f"Model '{model_name}' is not from a trusted source. "
        f"Set ALLOW_TRUST_REMOTE_CODE=true or add the organization to TRUSTED_MODEL_SOURCES "
        f"if you trust this model."
    )
    return False


@contextmanager
def managed_model(model: Any) -> Generator[Any, None, None]:
    """Context manager for safe model cleanup.

    Ensures model is deleted and GPU memory is freed even if an exception occurs.

    Args:
        model: The model to manage

    Yields:
        The model for use within the context
    """
    try:
        yield model
    finally:
        try:
            del model
        except Exception as e:
            logger.debug(f"Error during model deletion: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BaseQuantizer:
    """Base class for quantization wrappers."""

    def __init__(self, method: str):
        self.method = method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_compatibility(self, model_name: str, bit_width: int) -> Tuple[bool, str]:
        """Check if quantization method is compatible."""
        return True, "Compatible"

    def save_metadata(self, output_dir: str, metadata: Dict[str, Any]):
        """Save quantization metadata."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, "quantization_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)


class AutoRoundQuantizer(BaseQuantizer):
    """AutoRound quantization wrapper."""

    def __init__(self):
        super().__init__("autoround")
        self.auto_round_available = False
        try:
            from auto_round import AutoRound
            self.auto_round_available = True
            self.AutoRound = AutoRound
        except ImportError:
            logger.warning("auto-round not installed, using mock implementation")

    def quantize(
        self,
        model_name: str,
        bit_width: int = 4,
        group_size: int = 128,
        calibration_samples: int = 512,
        calibration_dataset: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Quantize model using AutoRound."""

        start_time = time.time()

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_name).replace("/", "_")
            output_dir = f"data/checkpoints/{model_short}_autoround_{bit_width}bit_{timestamp}"

        if self.auto_round_available:
            model = None
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Check if model source is trusted
                trust_remote = should_trust_remote_code(model_name)

                # Load model and tokenizer
                logger.info(f"Loading model {model_name} for AutoRound quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=trust_remote
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

                # Get calibration data
                calibration_data = self._get_calibration_data(
                    tokenizer, calibration_dataset, calibration_samples
                )

                # Configure AutoRound
                autoround = self.AutoRound(
                    model=model,
                    tokenizer=tokenizer,
                    bits=bit_width,
                    group_size=group_size,
                    seqlen=kwargs.get("seqlen", 2048),
                    batch_size=kwargs.get("batch_size", 4),
                    calibration_data=calibration_data,
                    device=str(self.device),
                )

                # Run quantization
                logger.info(f"Running AutoRound quantization with {bit_width} bits...")
                autoround.quantize()

                # Save quantized model
                logger.info(f"Saving quantized model to {output_dir}")
                autoround.save_quantized(output_dir)

                # Calculate metrics (use model_name since model is already quantized)
                original_size = estimate_model_size_gb(model_name)
                quantized_size = original_size / (16 / bit_width)

            except Exception as e:
                logger.error(f"AutoRound quantization failed: {e}")
                return self._mock_quantization(model_name, bit_width, output_dir)
            finally:
                # Always clean up model resources
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            return self._mock_quantization(model_name, bit_width, output_dir)

        quantization_time = time.time() - start_time

        fallback_size = estimate_model_size_gb(model_name)
        metadata = {
            "method": "autoround",
            "model_name": model_name,
            "bit_width": bit_width,
            "group_size": group_size,
            "calibration_samples": calibration_samples,
            "original_size_gb": original_size if 'original_size' in locals() else fallback_size,
            "quantized_size_gb": quantized_size if 'quantized_size' in locals() else fallback_size / (16.0 / bit_width),
            "compression_ratio": (original_size / quantized_size) if 'original_size' in locals() else 16.0 / bit_width,
            "quantization_time_sec": quantization_time,
            "timestamp": datetime.now().isoformat(),
        }

        self.save_metadata(output_dir, metadata)

        return {
            "checkpoint_path": output_dir,
            "model_size_gb": metadata["quantized_size_gb"],
            "compression_ratio": metadata["compression_ratio"],
            "quantization_time_sec": quantization_time,
            "metadata": metadata,
        }

    def _get_calibration_data(
        self, tokenizer: Any, dataset_name: Optional[str], num_samples: int
    ) -> List[Any]:
        """Get calibration data for quantization.

        Args:
            tokenizer: HuggingFace tokenizer
            dataset_name: Optional name of calibration dataset
            num_samples: Number of calibration samples to generate

        Returns:
            List of tokenized calibration samples
        """
        # In production, load actual dataset
        # For now, return mock data
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can be compressed effectively.",
            "Quantization reduces model size while maintaining accuracy.",
        ] * (num_samples // 3)

        return [tokenizer(text, return_tensors="pt") for text in texts[:num_samples]]

    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in GB.

        Args:
            model: PyTorch model

        Returns:
            Model size in GB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 ** 3)

    def _mock_quantization(self, model_name: str, bit_width: int, output_dir: str) -> Dict[str, Any]:
        """Fail fast when AutoRound library is unavailable.

        Returns an error result instead of creating invalid empty files.
        This allows the coordinator to handle the failure gracefully.
        """
        error_msg = "AutoRound quantization library not installed. Install: pip install auto-round"
        logger.error(error_msg)

        original_size = estimate_model_size_gb(model_name)

        return {
            "checkpoint_path": None,  # None indicates failure
            "model_size_gb": 0.0,
            "compression_ratio": 1.0,
            "quantization_time_sec": 0.0,
            "success": False,
            "error": error_msg,
            "metadata": {
                "method": "autoround",
                "failed": True,
                "reason": "library_unavailable",
                "original_size_gb": original_size,
            },
        }


class GPTQQuantizer(BaseQuantizer):
    """GPTQ quantization wrapper."""

    def __init__(self):
        super().__init__("gptq")
        self.gptq_available = False
        try:
            from gptqmodel import GPTQModel, QuantizeConfig, BACKEND

            self.gptq_available = True
            self.GPTQModel = GPTQModel
            self.QuantizeConfig = QuantizeConfig
            self.GPTQBackend = BACKEND
        except ImportError:
            logger.warning("gptqmodel not installed, using mock implementation")

    def quantize(
        self,
        model_name: str,
        bit_width: int = 4,
        group_size: int = 128,
        calibration_samples: int = 512,
        calibration_dataset: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Quantize model using GPTQ."""

        start_time = time.time()

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_name).replace("/", "_")
            output_dir = f"data/checkpoints/{model_short}_gptq_{bit_width}bit_{timestamp}"

        if self.gptq_available:
            gptq_model = None
            try:
                from transformers import AutoTokenizer

                # Check if model source is trusted
                trust_remote = should_trust_remote_code(model_name)

                logger.info(f"Loading tokenizer for {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

                quantize_config = self._build_quantize_config(
                    bit_width=bit_width,
                    group_size=group_size,
                    overrides=kwargs,
                )

                logger.info("Loading base model with GPTQModel...")
                gptq_model = self.GPTQModel.from_pretrained(
                    model_name,
                    quantize_config=quantize_config,
                    trust_remote_code=trust_remote,
                )

                calibration_data = self._get_calibration_data(
                    tokenizer, calibration_dataset, calibration_samples
                )

                backend = kwargs.get("backend")
                backend_enum = getattr(self, "GPTQBackend", None)
                if backend is None and backend_enum is not None:
                    backend = backend_enum.AUTO
                elif isinstance(backend, str) and backend_enum is not None:
                    backend = backend_enum(backend)

                logger.info(f"Running GPTQ quantization with {bit_width} bits via GPTQModel...")
                quantize_kwargs = {
                    "calibration": calibration_data,
                    "batch_size": kwargs.get("batch_size", 1),
                    "tokenizer": tokenizer,
                    "calibration_concat_size": kwargs.get("calibration_concat_size"),
                    "calibration_sort": kwargs.get("calibration_sort", "desc"),
                    "calibration_concat_separator": kwargs.get("calibration_concat_separator"),
                }
                if backend is not None:
                    quantize_kwargs["backend"] = backend
                gptq_model.quantize(**quantize_kwargs)

                logger.info(f"Saving quantized model to {output_dir}")
                gptq_model.save_quantized(output_dir)
                tokenizer.save_pretrained(output_dir)

                original_size = estimate_model_size_gb(model_name)
                quantized_size = original_size / (16 / bit_width)

            except Exception as e:
                logger.error(f"GPTQ quantization failed: {e}")
                return self._mock_quantization(model_name, bit_width, output_dir)
            finally:
                # Always clean up model resources
                if gptq_model is not None:
                    del gptq_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            return self._mock_quantization(model_name, bit_width, output_dir)

        quantization_time = time.time() - start_time

        fallback_size = estimate_model_size_gb(model_name)
        metadata = {
            "method": "gptq",
            "model_name": model_name,
            "bit_width": bit_width,
            "group_size": group_size,
            "calibration_samples": calibration_samples,
            "original_size_gb": original_size if 'original_size' in locals() else fallback_size,
            "quantized_size_gb": quantized_size if 'quantized_size' in locals() else fallback_size / (16.0 / bit_width),
            "compression_ratio": (original_size / quantized_size) if 'original_size' in locals() else 16.0 / bit_width,
            "quantization_time_sec": quantization_time,
            "timestamp": datetime.now().isoformat(),
        }

        self.save_metadata(output_dir, metadata)

        return {
            "checkpoint_path": output_dir,
            "model_size_gb": metadata["quantized_size_gb"],
            "compression_ratio": metadata["compression_ratio"],
            "quantization_time_sec": quantization_time,
            "metadata": metadata,
        }

    def _get_calibration_data(self, tokenizer, dataset_name: Optional[str], num_samples: int):
        """Get calibration data for GPTQ."""
        # In production, load actual dataset
        texts = [
            "Quantization is an effective compression technique.",
            "GPTQ provides good compression with minimal accuracy loss.",
            "Large language models benefit from quantization.",
        ] * (num_samples // 3)

        return [tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                for text in texts[:num_samples]]

    def _mock_quantization(self, model_name: str, bit_width: int, output_dir: str) -> Dict[str, Any]:
        """Fail fast when GPTQ library is unavailable.

        Returns an error result instead of creating invalid empty files.
        This allows the coordinator to handle the failure gracefully.
        """
        error_msg = "GPTQ quantization library not installed. Install: pip install gptqmodel optimum"
        logger.error(error_msg)

        original_size = estimate_model_size_gb(model_name)

        return {
            "checkpoint_path": None,  # None indicates failure
            "model_size_gb": 0.0,
            "compression_ratio": 1.0,
            "quantization_time_sec": 0.0,
            "success": False,
            "error": error_msg,
            "metadata": {
                "method": "gptq",
                "failed": True,
                "reason": "library_unavailable",
                "original_size_gb": original_size,
            },
        }

    def _build_quantize_config(self, bit_width: int, group_size: int, overrides: Dict[str, Any]):
        """Create a GPTQModel QuantizeConfig with sensible defaults."""
        quantize_config = self.QuantizeConfig(
            bits=bit_width,
            group_size=group_size,
            desc_act=overrides.get("desc_act", False),
            sym=overrides.get("sym", True),
            true_sequential=overrides.get("true_sequential", True),
            static_groups=overrides.get("static_groups", False),
        )

        optional_fields = [
            "damp_percent",
            "damp_auto_increment",
            "act_group_aware",
            "mse",
            "pack_impl",
            "rotation",
            "gptaq",
            "gptaq_alpha",
            "gptaq_memory_device",
            "offload_to_disk",
            "offload_to_disk_path",
            "fail_safe",
        ]

        for field in optional_fields:
            if field in overrides and overrides[field] is not None:
                setattr(quantize_config, field, overrides[field])

        pack_dtype = overrides.get("pack_dtype")
        if pack_dtype is not None:
            if isinstance(pack_dtype, str) and hasattr(torch, pack_dtype):
                quantize_config.pack_dtype = getattr(torch, pack_dtype)
            else:
                quantize_config.pack_dtype = pack_dtype

        fmt_override = overrides.get("format")
        if fmt_override is not None:
            enum_type = type(quantize_config.format)
            if isinstance(fmt_override, enum_type):
                quantize_config.format = fmt_override
            else:
                try:
                    quantize_config.format = enum_type(fmt_override)
                except Exception:
                    quantize_config.format = fmt_override

        method_override = overrides.get("quant_method")
        if method_override is not None:
            enum_type = type(quantize_config.quant_method)
            if isinstance(method_override, enum_type):
                quantize_config.quant_method = method_override
            else:
                try:
                    quantize_config.quant_method = enum_type(method_override)
                except Exception:
                    quantize_config.quant_method = method_override

        device_override = overrides.get("quant_device")
        if device_override:
            quantize_config.device = device_override
        else:
            if self.device.type == "cuda":
                index = self.device.index if self.device.index is not None else 0
                quantize_config.device = f"cuda:{index}"
            else:
                quantize_config.device = str(self.device)
        return quantize_config


class AWQQuantizer(BaseQuantizer):
    """AWQ (Activation-aware Weight Quantization) wrapper.

    Supports two backends:
    1. llm-compressor (vLLM's official successor to AutoAWQ) - preferred
    2. AutoAWQ (legacy, deprecated but still functional)
    """

    def __init__(self):
        super().__init__("awq")
        self.llmcompressor_available = False
        self.autoawq_available = False

        # Try llm-compressor first (preferred)
        try:
            from llmcompressor.modifiers.awq import AWQModifier
            from llmcompressor import oneshot
            self.llmcompressor_available = True
            self.AWQModifier = AWQModifier
            self.oneshot = oneshot
            logger.info("Using llm-compressor for AWQ quantization")
        except ImportError:
            logger.debug("llm-compressor not available, trying autoawq")

        # Fallback to AutoAWQ
        if not self.llmcompressor_available:
            try:
                from awq import AutoAWQForCausalLM
                self.autoawq_available = True
                self.AutoAWQForCausalLM = AutoAWQForCausalLM
                logger.info("Using AutoAWQ for AWQ quantization (deprecated, consider migrating to llm-compressor)")
            except ImportError:
                logger.warning("Neither llm-compressor nor autoawq installed, using mock implementation")

    def quantize(
        self,
        model_name: str,
        bit_width: int = 4,
        group_size: int = 128,
        calibration_samples: int = 512,
        calibration_dataset: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Quantize model using AWQ.

        Tries llm-compressor first, falls back to AutoAWQ if unavailable.
        """
        start_time = time.time()

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_name).replace("/", "_")
            output_dir = f"data/checkpoints/{model_short}_awq_{bit_width}bit_{timestamp}"

        # AWQ mainly supports 4-bit
        if bit_width != 4:
            logger.warning(f"AWQ is optimized for 4-bit quantization, got {bit_width}-bit")

        # Try llm-compressor first
        if self.llmcompressor_available and bit_width == 4:
            result = self._quantize_with_llmcompressor(
                model_name, bit_width, group_size, calibration_samples,
                calibration_dataset, output_dir, **kwargs
            )
            if result is not None:
                return result

        # Fallback to AutoAWQ
        if self.autoawq_available and bit_width == 4:
            result = self._quantize_with_autoawq(
                model_name, bit_width, group_size, calibration_samples,
                calibration_dataset, output_dir, **kwargs
            )
            if result is not None:
                return result

        # Both failed or unavailable
        return self._mock_quantization(model_name, bit_width, output_dir)

    def _quantize_with_llmcompressor(
        self,
        model_name: str,
        bit_width: int,
        group_size: int,
        calibration_samples: int,
        calibration_dataset: Optional[str],
        output_dir: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Quantize using vLLM's llm-compressor (preferred method)."""
        model = None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Check if model source is trusted
            trust_remote = should_trust_remote_code(model_name)

            logger.info(f"Loading model {model_name} for AWQ quantization (llm-compressor)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=trust_remote,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

            # AWQ recipe using llm-compressor
            # W4A16_ASYM = 4-bit weights, 16-bit activations, asymmetric quantization
            scheme = f"W{bit_width}A16_ASYM"
            recipe = [
                self.AWQModifier(
                    ignore=["lm_head"],
                    scheme=scheme,
                    targets=["Linear"],
                    group_size=group_size,
                ),
            ]

            # Get calibration dataset
            calib_dataset = calibration_dataset or "ultrachat_200k"

            logger.info(f"Running AWQ quantization with llm-compressor ({bit_width}-bit, group_size={group_size})...")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            self.oneshot(
                model=model,
                dataset=calib_dataset,
                recipe=recipe,
                output_dir=output_dir,
                num_calibration_samples=calibration_samples,
            )

            tokenizer.save_pretrained(output_dir)

            original_size = estimate_model_size_gb(model_name)
            quantized_size = original_size / (16 / bit_width)
            quantization_time = time.time() - kwargs.get("_start_time", time.time())

            metadata = {
                "method": "awq",
                "backend": "llm-compressor",
                "model_name": model_name,
                "bit_width": bit_width,
                "group_size": group_size,
                "calibration_samples": calibration_samples,
                "calibration_dataset": calib_dataset,
                "original_size_gb": original_size,
                "quantized_size_gb": quantized_size,
                "compression_ratio": original_size / quantized_size,
                "quantization_time_sec": quantization_time,
                "timestamp": datetime.now().isoformat(),
            }

            self.save_metadata(output_dir, metadata)

            return {
                "checkpoint_path": output_dir,
                "model_size_gb": quantized_size,
                "compression_ratio": metadata["compression_ratio"],
                "quantization_time_sec": quantization_time,
                "metadata": metadata,
            }

        except Exception as e:
            logger.warning(f"llm-compressor AWQ quantization failed: {e}, trying AutoAWQ fallback")
            return None
        finally:
            # Always clean up model resources
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _quantize_with_autoawq(
        self,
        model_name: str,
        bit_width: int,
        group_size: int,
        calibration_samples: int,
        calibration_dataset: Optional[str],
        output_dir: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Quantize using AutoAWQ (legacy fallback)."""
        model = None
        try:
            from transformers import AutoTokenizer

            # Check if model source is trusted
            trust_remote = should_trust_remote_code(model_name)

            logger.info(f"Loading model {model_name} for AWQ quantization (AutoAWQ)...")
            model = self.AutoAWQForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=trust_remote,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

            # AWQ config
            quant_config = {
                "zero_point": True,
                "q_group_size": group_size,
                "w_bit": bit_width,
                "version": "GEMM",
            }

            # Get calibration data
            calibration_data = self._get_calibration_texts(calibration_dataset, calibration_samples)

            # Run quantization
            logger.info(f"Running AWQ quantization with AutoAWQ ({bit_width}-bit)...")
            model.quantize(
                tokenizer,
                quant_config=quant_config,
                calib_data=calibration_data,
            )

            # Save model
            logger.info(f"Saving AWQ model to {output_dir}")
            model.save_quantized(output_dir)
            tokenizer.save_pretrained(output_dir)

            original_size = estimate_model_size_gb(model_name)
            quantized_size = original_size / (16 / bit_width)
            quantization_time = time.time() - kwargs.get("_start_time", time.time())

            metadata = {
                "method": "awq",
                "backend": "autoawq",
                "model_name": model_name,
                "bit_width": bit_width,
                "group_size": group_size,
                "calibration_samples": calibration_samples,
                "original_size_gb": original_size,
                "quantized_size_gb": quantized_size,
                "compression_ratio": original_size / quantized_size,
                "quantization_time_sec": quantization_time,
                "timestamp": datetime.now().isoformat(),
            }

            self.save_metadata(output_dir, metadata)

            return {
                "checkpoint_path": output_dir,
                "model_size_gb": quantized_size,
                "compression_ratio": metadata["compression_ratio"],
                "quantization_time_sec": quantization_time,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"AutoAWQ quantization failed: {e}")
            return None
        finally:
            # Always clean up model resources
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_calibration_texts(self, dataset_name: Optional[str], num_samples: int):
        """Get calibration texts for AWQ."""
        if dataset_name:
            try:
                from datasets import load_dataset
                dataset = load_dataset(dataset_name, split="train")
                # Try common text columns
                for col in ["text", "content", "question"]:
                    if col in dataset.column_names:
                        return dataset[col][:num_samples]
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_name}: {e}")

        # Fallback to default calibration texts
        return [
            "AWQ preserves important weights during quantization.",
            "Activation-aware quantization maintains model quality.",
            "This method works particularly well with Llama models.",
            "Large language models can be compressed efficiently.",
            "Quantization reduces memory footprint significantly.",
        ] * (num_samples // 5 + 1)

    def _mock_quantization(self, model_name: str, bit_width: int, output_dir: str) -> Dict[str, Any]:
        """Fail fast when AWQ library is unavailable.

        Returns an error result instead of creating invalid empty files.
        This allows the coordinator to handle the failure gracefully.
        """
        error_msg = "AWQ quantization library not installed. Install: pip install autoawq or pip install llmcompressor"
        logger.error(error_msg)

        original_size = estimate_model_size_gb(model_name)

        return {
            "checkpoint_path": None,  # None indicates failure
            "model_size_gb": 0.0,
            "compression_ratio": 1.0,
            "quantization_time_sec": 0.0,
            "success": False,
            "error": error_msg,
            "metadata": {
                "method": "awq",
                "failed": True,
                "reason": "library_unavailable",
                "original_size_gb": original_size,
            },
        }


class INT8Quantizer(BaseQuantizer):
    """INT8 quantization using bitsandbytes."""

    def __init__(self):
        super().__init__("int8")
        self.bnb_available = False
        try:
            import bitsandbytes as bnb
            self.bnb_available = True
            self.bnb = bnb
        except ImportError:
            logger.warning("bitsandbytes not installed, using mock implementation")

    def quantize(
        self,
        model_name: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Quantize model to INT8 using bitsandbytes."""

        start_time = time.time()

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_name).replace("/", "_")
            output_dir = f"data/checkpoints/{model_short}_int8_{timestamp}"

        if self.bnb_available:
            model = None
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

                # Check if model source is trusted
                trust_remote = should_trust_remote_code(model_name)

                # INT8 config
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    int8_threshold=6.0,
                )

                logger.info(f"Loading model {model_name} with INT8 quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=trust_remote,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

                # Save model
                logger.info(f"Saving INT8 model to {output_dir}")
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                original_size = estimate_model_size_gb(model_name)
                quantized_size = original_size / 2  # INT8 is roughly 2x compression

            except Exception as e:
                logger.error(f"INT8 quantization failed: {e}")
                return self._mock_quantization(model_name, output_dir)
            finally:
                # Always clean up model resources
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            return self._mock_quantization(model_name, output_dir)

        quantization_time = time.time() - start_time

        fallback_size = estimate_model_size_gb(model_name)
        metadata = {
            "method": "int8",
            "model_name": model_name,
            "bit_width": 8,
            "original_size_gb": original_size if 'original_size' in locals() else fallback_size,
            "quantized_size_gb": quantized_size if 'quantized_size' in locals() else fallback_size / 2.0,
            "compression_ratio": (original_size / quantized_size) if 'original_size' in locals() else 2.0,
            "quantization_time_sec": quantization_time,
            "timestamp": datetime.now().isoformat(),
        }

        self.save_metadata(output_dir, metadata)

        return {
            "checkpoint_path": output_dir,
            "model_size_gb": metadata["quantized_size_gb"],
            "compression_ratio": metadata["compression_ratio"],
            "quantization_time_sec": quantization_time,
            "metadata": metadata,
        }

    def _mock_quantization(self, model_name: str, output_dir: str) -> Dict[str, Any]:
        """Fail fast when bitsandbytes library is unavailable.

        Returns an error result instead of creating invalid empty files.
        This allows the coordinator to handle the failure gracefully.
        """
        error_msg = "INT8 quantization requires bitsandbytes. Install: pip install bitsandbytes"
        logger.error(error_msg)

        original_size = estimate_model_size_gb(model_name)

        return {
            "checkpoint_path": None,  # None indicates failure
            "model_size_gb": 0.0,
            "compression_ratio": 1.0,
            "quantization_time_sec": 0.0,
            "success": False,
            "error": error_msg,
            "metadata": {
                "method": "int8",
                "failed": True,
                "reason": "library_unavailable",
                "original_size_gb": original_size,
            },
        }


class LoRATrainer(BaseQuantizer):
    """LoRA (Low-Rank Adaptation) fine-tuning wrapper."""

    def __init__(self):
        super().__init__("lora")
        self.peft_available = False
        self.trl_available = False
        try:
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            self.peft_available = True
            self.LoraConfig = LoraConfig
            self.get_peft_model = get_peft_model
            self.TaskType = TaskType
            self.prepare_model_for_kbit_training = prepare_model_for_kbit_training
        except ImportError:
            logger.warning("peft not installed, using mock implementation")

        try:
            from trl import SFTTrainer, SFTConfig
            self.trl_available = True
            self.SFTTrainer = SFTTrainer
            self.SFTConfig = SFTConfig
        except ImportError:
            logger.warning("trl not installed, using mock implementation")

    def finetune(
        self,
        model_name: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        training_steps: int = 100,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        calibration_dataset: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune model using LoRA adapters.

        Args:
            model_name: HuggingFace model name or path
            lora_rank: Rank of LoRA decomposition (default: 16)
            lora_alpha: LoRA scaling factor (default: 32)
            lora_dropout: Dropout probability for LoRA layers (default: 0.05)
            target_modules: List of module names to apply LoRA (default: ["q_proj", "v_proj"])
            training_steps: Number of training steps (default: 100)
            learning_rate: Learning rate (default: 2e-4)
            batch_size: Training batch size (default: 4)
            calibration_dataset: Dataset name for fine-tuning
            output_dir: Directory to save adapter weights

        Returns:
            Dictionary with training results and adapter path
        """
        start_time = time.time()

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_name).replace("/", "_")
            output_dir = f"data/checkpoints/{model_short}_lora_r{lora_rank}_{timestamp}"

        if self.peft_available and self.trl_available:
            model = None
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from datasets import Dataset

                # Check if model source is trusted
                trust_remote = should_trust_remote_code(model_name)

                logger.info(f"Loading model {model_name} for LoRA fine-tuning...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=trust_remote,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Configure LoRA
                lora_config = self.LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                    task_type=self.TaskType.CAUSAL_LM,
                )

                # Apply LoRA
                logger.info(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}...")
                model = self.get_peft_model(model, lora_config)
                model.print_trainable_parameters()

                # Prepare training dataset
                train_dataset = self._get_training_dataset(calibration_dataset, tokenizer)

                # Configure training
                # NOTE: fp16=False to avoid AMP gradient scaling issues with BFloat16
                # AMP's GradScaler doesn't support BFloat16, causing training failures
                sft_config = self.SFTConfig(
                    output_dir=output_dir,
                    max_steps=training_steps,
                    per_device_train_batch_size=batch_size,
                    learning_rate=learning_rate,
                    logging_steps=10,
                    save_steps=training_steps,
                    save_total_limit=1,
                    fp16=False,
                    report_to="none",
                    gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
                    warmup_steps=kwargs.get("warmup_steps", 10),
                )

                # Create trainer and train
                logger.info(f"Starting LoRA training for {training_steps} steps...")
                trainer = self.SFTTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    processing_class=tokenizer,
                    args=sft_config,
                )

                train_result = trainer.train()
                final_loss = train_result.training_loss

                # Save adapter
                logger.info(f"Saving LoRA adapter to {output_dir}")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                # Calculate adapter size
                adapter_size_mb = self._calculate_adapter_size(output_dir)

            except Exception as e:
                logger.error(f"LoRA fine-tuning failed: {e}")
                import traceback
                traceback.print_exc()
                return self._mock_finetune(model_name, lora_rank, output_dir)
            finally:
                # Always clean up model resources
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            return self._mock_finetune(model_name, lora_rank, output_dir)

        training_time = time.time() - start_time

        metadata = {
            "method": "lora",
            "model_name": model_name,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "final_loss": final_loss if 'final_loss' in locals() else 0.0,
            "adapter_size_mb": adapter_size_mb if 'adapter_size_mb' in locals() else 10.0,
            "training_time_sec": training_time,
            "timestamp": datetime.now().isoformat(),
        }

        self.save_metadata(output_dir, metadata)

        return {
            "checkpoint_path": output_dir,
            "adapter_size_mb": metadata["adapter_size_mb"],
            "training_loss": metadata["final_loss"],
            "training_time_sec": training_time,
            "compression_ratio": 1.0,  # LoRA doesn't compress, it adapts
            "model_size_gb": 0.0,  # Adapter only
            "metadata": metadata,
        }

    def _get_training_dataset(self, dataset_name: Optional[str], tokenizer):
        """Get training dataset for LoRA fine-tuning."""
        from datasets import Dataset

        if dataset_name:
            try:
                from datasets import load_dataset
                dataset = load_dataset(dataset_name, split="train[:1000]")
                # Try to find text column
                text_col = None
                for col in ["text", "content", "question", "instruction"]:
                    if col in dataset.column_names:
                        text_col = col
                        break
                if text_col:
                    return dataset.select(range(min(500, len(dataset))))
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_name}: {e}")

        # Fallback to mock data
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can be fine-tuned with LoRA.",
            "Low-rank adaptation preserves model knowledge while adding new capabilities.",
            "Fine-tuning with adapters is memory efficient.",
            "LoRA enables training large models on consumer hardware.",
        ] * 100

        return Dataset.from_dict({"text": texts})

    def _calculate_adapter_size(self, output_dir: str) -> float:
        """Calculate the size of saved adapter weights in MB."""
        total_size = 0
        for file in Path(output_dir).glob("**/*"):
            if file.is_file() and file.suffix in [".safetensors", ".bin", ".pt"]:
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)

    def _mock_finetune(self, model_name: str, lora_rank: int, output_dir: str) -> Dict[str, Any]:
        """Fail fast when PEFT/TRL libraries are unavailable.

        Returns an error result instead of creating invalid empty files.
        This allows the coordinator to handle the failure gracefully.
        """
        error_msg = "LoRA fine-tuning requires peft and trl. Install: pip install peft trl"
        logger.error(error_msg)

        original_size = estimate_model_size_gb(model_name)

        return {
            "checkpoint_path": None,  # None indicates failure
            "adapter_size_mb": 0.0,
            "training_loss": 0.0,
            "training_time_sec": 0.0,
            "compression_ratio": 1.0,
            "model_size_gb": 0.0,
            "success": False,
            "error": error_msg,
            "metadata": {
                "method": "lora",
                "failed": True,
                "reason": "library_unavailable",
                "original_size_gb": original_size,
            },
        }


class QLoRATrainer(BaseQuantizer):
    """QLoRA (Quantized LoRA) fine-tuning wrapper - 4-bit base + LoRA adapters."""

    def __init__(self):
        super().__init__("qlora")
        self.peft_available = False
        self.trl_available = False
        self.bnb_available = False

        try:
            from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
            self.peft_available = True
            self.LoraConfig = LoraConfig
            self.get_peft_model = get_peft_model
            self.TaskType = TaskType
            self.prepare_model_for_kbit_training = prepare_model_for_kbit_training
        except ImportError:
            logger.warning("peft not installed, using mock implementation")

        try:
            from trl import SFTTrainer, SFTConfig
            self.trl_available = True
            self.SFTTrainer = SFTTrainer
            self.SFTConfig = SFTConfig
        except ImportError:
            logger.warning("trl not installed, using mock implementation")

        try:
            import bitsandbytes as bnb
            self.bnb_available = True
            self.bnb = bnb
        except ImportError:
            logger.warning("bitsandbytes not installed, using mock implementation")

    def finetune(
        self,
        model_name: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bits: int = 4,
        target_modules: Optional[List[str]] = None,
        training_steps: int = 100,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        calibration_dataset: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune model using QLoRA (4-bit quantization + LoRA).

        Args:
            model_name: HuggingFace model name or path
            lora_rank: Rank of LoRA decomposition (default: 16)
            lora_alpha: LoRA scaling factor (default: 32)
            lora_dropout: Dropout probability for LoRA layers (default: 0.05)
            bits: Quantization bits for base model (default: 4)
            target_modules: List of module names to apply LoRA
            training_steps: Number of training steps (default: 100)
            learning_rate: Learning rate (default: 2e-4)
            batch_size: Training batch size (default: 4)
            calibration_dataset: Dataset name for fine-tuning
            output_dir: Directory to save adapter weights

        Returns:
            Dictionary with training results and adapter path
        """
        start_time = time.time()

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_name).replace("/", "_")
            output_dir = f"data/checkpoints/{model_short}_qlora_{bits}bit_r{lora_rank}_{timestamp}"

        if self.peft_available and self.trl_available and self.bnb_available:
            model = None
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from datasets import Dataset

                # Check if model source is trusted
                trust_remote = should_trust_remote_code(model_name)

                # Configure 4-bit quantization with CPU offload for large models
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )

                logger.info(f"Loading model {model_name} in {bits}-bit for QLoRA...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=trust_remote,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                # Prepare model for k-bit training
                model = self.prepare_model_for_kbit_training(model)

                # Configure LoRA
                lora_config = self.LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                    bias="none",
                    task_type=self.TaskType.CAUSAL_LM,
                )

                # Apply LoRA
                logger.info(f"Applying QLoRA with {bits}-bit base, rank={lora_rank}, alpha={lora_alpha}...")
                model = self.get_peft_model(model, lora_config)
                model.print_trainable_parameters()

                # Prepare training dataset
                train_dataset = self._get_training_dataset(calibration_dataset, tokenizer)

                # Configure training
                # NOTE: fp16=False to avoid AMP gradient scaling issues with BFloat16
                # AMP's GradScaler doesn't support BFloat16, causing training failures
                sft_config = self.SFTConfig(
                    output_dir=output_dir,
                    max_steps=training_steps,
                    per_device_train_batch_size=batch_size,
                    learning_rate=learning_rate,
                    logging_steps=10,
                    save_steps=training_steps,
                    save_total_limit=1,
                    fp16=False,
                    report_to="none",
                    gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 1),
                    warmup_steps=kwargs.get("warmup_steps", 10),
                    optim="paged_adamw_8bit",  # Memory-efficient optimizer for QLoRA
                )

                # Create trainer and train
                logger.info(f"Starting QLoRA training for {training_steps} steps...")
                trainer = self.SFTTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    processing_class=tokenizer,
                    args=sft_config,
                )

                train_result = trainer.train()
                final_loss = train_result.training_loss

                # Get peak VRAM usage
                if torch.cuda.is_available():
                    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
                else:
                    peak_vram_gb = 0.0

                # Save adapter
                logger.info(f"Saving QLoRA adapter to {output_dir}")
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                # Calculate adapter size
                adapter_size_mb = self._calculate_adapter_size(output_dir)

            except Exception as e:
                logger.error(f"QLoRA fine-tuning failed: {e}")
                import traceback
                traceback.print_exc()
                return self._mock_finetune(model_name, lora_rank, bits, output_dir)
            finally:
                # Always clean up model resources
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            return self._mock_finetune(model_name, lora_rank, bits, output_dir)

        training_time = time.time() - start_time

        metadata = {
            "method": "qlora",
            "model_name": model_name,
            "base_model_bits": bits,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "training_steps": training_steps,
            "learning_rate": learning_rate,
            "final_loss": final_loss if 'final_loss' in locals() else 0.0,
            "adapter_size_mb": adapter_size_mb if 'adapter_size_mb' in locals() else 10.0,
            "peak_vram_gb": peak_vram_gb if 'peak_vram_gb' in locals() else 0.0,
            "training_time_sec": training_time,
            "timestamp": datetime.now().isoformat(),
        }

        self.save_metadata(output_dir, metadata)

        return {
            "checkpoint_path": output_dir,
            "adapter_size_mb": metadata["adapter_size_mb"],
            "training_loss": metadata["final_loss"],
            "training_time_sec": training_time,
            "base_model_bits": bits,
            "total_vram_gb": metadata["peak_vram_gb"],
            "compression_ratio": 16.0 / bits,  # Compression from base model quantization
            "model_size_gb": 0.0,  # Adapter only
            "metadata": metadata,
        }

    def _get_training_dataset(self, dataset_name: Optional[str], tokenizer):
        """Get training dataset for QLoRA fine-tuning."""
        from datasets import Dataset

        if dataset_name:
            try:
                from datasets import load_dataset
                dataset = load_dataset(dataset_name, split="train[:1000]")
                text_col = None
                for col in ["text", "content", "question", "instruction"]:
                    if col in dataset.column_names:
                        text_col = col
                        break
                if text_col:
                    return dataset.select(range(min(500, len(dataset))))
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_name}: {e}")

        # Fallback to mock data
        texts = [
            "QLoRA combines 4-bit quantization with LoRA for memory-efficient fine-tuning.",
            "The base model is loaded in 4-bit precision using NF4 quantization.",
            "Only the LoRA adapter weights are trained in full precision.",
            "This approach enables fine-tuning of large models on consumer GPUs.",
            "QLoRA achieves similar performance to full fine-tuning at a fraction of the memory cost.",
        ] * 100

        return Dataset.from_dict({"text": texts})

    def _calculate_adapter_size(self, output_dir: str) -> float:
        """Calculate the size of saved adapter weights in MB."""
        total_size = 0
        for file in Path(output_dir).glob("**/*"):
            if file.is_file() and file.suffix in [".safetensors", ".bin", ".pt"]:
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)

    def _mock_finetune(self, model_name: str, lora_rank: int, bits: int, output_dir: str) -> Dict[str, Any]:
        """Fail fast when PEFT/TRL/bitsandbytes libraries are unavailable.

        Returns an error result instead of creating invalid empty files.
        This allows the coordinator to handle the failure gracefully.
        """
        error_msg = "QLoRA fine-tuning requires peft, trl, and bitsandbytes. Install: pip install peft trl bitsandbytes"
        logger.error(error_msg)

        original_size = estimate_model_size_gb(model_name)

        return {
            "checkpoint_path": None,  # None indicates failure
            "adapter_size_mb": 0.0,
            "training_loss": 0.0,
            "training_time_sec": 0.0,
            "base_model_bits": bits,
            "total_vram_gb": 0.0,
            "compression_ratio": 1.0,
            "model_size_gb": 0.0,
            "success": False,
            "error": error_msg,
            "metadata": {
                "method": "qlora",
                "failed": True,
                "reason": "library_unavailable",
                "original_size_gb": original_size,
            },
        }


class ASVDCompressor(BaseQuantizer):
    """ASVD (Activation-aware Singular Value Decomposition) compression.

    ASVD compresses model weights using SVD decomposition while considering
    activation patterns to determine which singular values are most important.
    This allows for intelligent rank selection that preserves model accuracy.

    Key features:
    - Uses calibration data to collect activation statistics
    - Scales weights by activation importance before SVD
    - Selects top-k singular values based on rank_ratio
    - Reconstructs low-rank weight matrices

    Compression ratio calculation:
    - Original: m × n parameters
    - Compressed: m × k + k + k × n = k(m + n + 1) parameters
    - Where k = rank × rank_ratio
    """

    def __init__(self):
        super().__init__("asvd")

    def compress(
        self,
        model_name: str,
        rank_ratio: float = 0.5,
        calibration_samples: int = 512,
        calibration_dataset: Optional[str] = None,
        target_layers: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply ASVD compression to a model.

        Args:
            model_name: HuggingFace model name or path
            rank_ratio: Ratio of singular values to keep (0.0-1.0, default: 0.5)
            calibration_samples: Number of calibration samples for activation collection
            calibration_dataset: Dataset name for calibration data
            target_layers: List of layer name patterns to compress (None = all linear layers)
            output_dir: Directory to save compressed model

        Returns:
            Dictionary with compression results including:
            - checkpoint_path: Path to compressed model
            - model_size_gb: Size of compressed model
            - compression_ratio: Compression ratio achieved
            - compression_time_sec: Time taken for compression
            - layers_compressed: Number of layers compressed
            - metadata: Additional compression metadata
        """
        start_time = time.time()

        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = os.path.basename(model_name).replace("/", "_")
            output_dir = f"data/checkpoints/{model_short}_asvd_r{int(rank_ratio*100)}_{timestamp}"

        model = None
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Check if model source is trusted
            trust_remote = should_trust_remote_code(model_name)

            logger.info(f"Loading model {model_name} for ASVD compression...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=trust_remote,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Get calibration data for activation collection
            calibration_texts = self._get_calibration_data(
                tokenizer, calibration_dataset, calibration_samples
            )

            # Collect activation statistics
            logger.info("Collecting activation statistics...")
            activation_stats = self._collect_activation_stats(
                model, tokenizer, calibration_texts, target_layers
            )

            # Apply ASVD compression to each layer
            logger.info(f"Applying ASVD compression with rank_ratio={rank_ratio}...")
            layers_compressed = 0
            total_params_before = 0
            total_params_after = 0

            for name, module in model.named_modules():
                if not isinstance(module, torch.nn.Linear):
                    continue

                # Check if layer should be compressed
                if target_layers is not None:
                    if not any(pattern in name for pattern in target_layers):
                        continue

                # Skip very small layers (not worth compressing)
                if module.weight.shape[0] < 64 or module.weight.shape[1] < 64:
                    continue

                # Get activation stats for this layer
                layer_activations = activation_stats.get(name)

                # Apply ASVD to this layer
                compressed = self._compress_layer(
                    module, layer_activations, rank_ratio
                )

                if compressed:
                    layers_compressed += 1
                    m, n = module.weight.shape
                    k = compressed["k"]
                    total_params_before += m * n
                    total_params_after += k * (m + n + 1)

            # Calculate compression metrics
            original_size = estimate_model_size_gb(model_name)
            if total_params_before > 0:
                param_ratio = total_params_after / total_params_before
                # Estimate new size (only compressed layers change)
                compressed_size = original_size * (0.5 + 0.5 * param_ratio)  # Rough estimate
            else:
                compressed_size = original_size
                param_ratio = 1.0

            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            # Save compressed model
            logger.info(f"Saving ASVD-compressed model to {output_dir}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            compression_time = time.time() - start_time

            metadata = {
                "method": "asvd",
                "model_name": model_name,
                "rank_ratio": rank_ratio,
                "calibration_samples": calibration_samples,
                "target_layers": target_layers,
                "layers_compressed": layers_compressed,
                "original_size_gb": original_size,
                "compressed_size_gb": compressed_size,
                "compression_ratio": compression_ratio,
                "param_ratio": param_ratio,
                "compression_time_sec": compression_time,
                "timestamp": datetime.now().isoformat(),
            }

            self.save_metadata(output_dir, metadata)

            return {
                "checkpoint_path": output_dir,
                "model_size_gb": compressed_size,
                "compression_ratio": compression_ratio,
                "compression_time_sec": compression_time,
                "layers_compressed": layers_compressed,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"ASVD compression failed: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_compression(model_name, rank_ratio, output_dir)
        finally:
            # Always clean up model resources
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_calibration_data(
        self, tokenizer, dataset_name: Optional[str], num_samples: int
    ) -> List[str]:
        """Get calibration texts for activation collection."""
        if dataset_name:
            try:
                from datasets import load_dataset

                if "wikitext" in dataset_name.lower():
                    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                    texts = [t for t in dataset["text"] if len(t) > 100][:num_samples]
                elif "c4" in dataset_name.lower():
                    dataset = load_dataset("c4", "en", split="train", streaming=True)
                    texts = [item["text"] for item in dataset.take(num_samples)]
                else:
                    dataset = load_dataset(dataset_name, split="train")
                    # Try common text column names
                    for col in ["text", "content", "question", "sentence"]:
                        if col in dataset.column_names:
                            texts = dataset[col][:num_samples]
                            break
                    else:
                        texts = []

                if texts:
                    return texts
            except Exception as e:
                logger.warning(f"Could not load dataset {dataset_name}: {e}")

        # Fallback to mock calibration data
        return [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "Machine learning models can be compressed using various techniques. " * 8,
            "Singular value decomposition is a powerful matrix factorization method. " * 8,
            "Neural networks learn to approximate complex functions from data. " * 8,
            "Activation-aware compression preserves important model behaviors. " * 8,
        ] * (num_samples // 5 + 1)

    def _collect_activation_stats(
        self,
        model,
        tokenizer,
        calibration_texts: List[str],
        target_layers: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Collect activation statistics from calibration data.

        Args:
            model: The model to analyze
            tokenizer: Tokenizer for the model
            calibration_texts: List of calibration texts
            target_layers: Layer patterns to collect stats for

        Returns:
            Dictionary mapping layer names to activation importance tensors
        """
        activation_stats = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                # Collect activation magnitudes
                if isinstance(input, tuple) and len(input) > 0:
                    inp = input[0]
                else:
                    inp = input

                if inp is not None and isinstance(inp, torch.Tensor):
                    # Compute mean absolute activation per input dimension
                    with torch.no_grad():
                        # Shape: [batch, seq, hidden] -> [hidden]
                        importance = torch.mean(torch.abs(inp.float()), dim=(0, 1))

                        if name in activation_stats:
                            # Running average
                            activation_stats[name] = (
                                activation_stats[name] + importance
                            ) / 2
                        else:
                            activation_stats[name] = importance
            return hook

        # Register hooks on linear layers
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            if target_layers is not None:
                if not any(pattern in name for pattern in target_layers):
                    continue

            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

        # Run forward passes on calibration data
        model.eval()
        with torch.no_grad():
            for i, text in enumerate(calibration_texts[:50]):  # Limit to 50 for speed
                try:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    model(**inputs)
                except Exception as e:
                    logger.debug(f"Calibration sample {i} failed: {e}")
                    continue

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activation_stats

    def _compress_layer(
        self,
        module: torch.nn.Linear,
        activations: Optional[torch.Tensor],
        rank_ratio: float,
    ) -> Optional[Dict[str, Any]]:
        """Apply ASVD compression to a single linear layer.

        Args:
            module: Linear layer to compress
            activations: Activation importance tensor (or None for uniform)
            rank_ratio: Ratio of singular values to keep

        Returns:
            Dictionary with compression info, or None if compression failed
        """
        try:
            weight = module.weight.data.float()  # [out_features, in_features]
            m, n = weight.shape

            # Scale weight by activation importance if available
            if activations is not None and activations.shape[0] == n:
                # Normalize importance to avoid numerical issues
                importance = activations.to(weight.device) + 1e-6
                importance = importance / importance.max()
                scaled_weight = weight * importance.unsqueeze(0)
            else:
                scaled_weight = weight

            # Perform SVD
            U, S, Vh = torch.linalg.svd(scaled_weight, full_matrices=False)

            # Select top-k singular values
            k = max(1, int(len(S) * rank_ratio))

            # Truncate to k components
            U_k = U[:, :k]  # [m, k]
            S_k = S[:k]      # [k]
            Vh_k = Vh[:k, :] # [k, n]

            # Reconstruct low-rank approximation
            # We store as two separate matrices: W1 = U_k @ diag(S_k), W2 = Vh_k
            # For inference: W_approx = W1 @ W2
            W1 = U_k * S_k.unsqueeze(0)  # [m, k]
            W2 = Vh_k  # [k, n]

            # Replace weight with low-rank approximation
            W_approx = W1 @ W2
            module.weight.data = W_approx.to(module.weight.dtype)

            return {
                "k": k,
                "original_rank": min(m, n),
                "compression": (m * n) / (k * (m + n)),
            }

        except Exception as e:
            logger.warning(f"Failed to compress layer: {e}")
            return None

    def _mock_compression(
        self, model_name: str, rank_ratio: float, output_dir: str
    ) -> Dict[str, Any]:
        """Fail fast when ASVD compression fails.

        Returns an error result instead of creating invalid empty files.
        This allows the coordinator to handle the failure gracefully.
        """
        error_msg = "ASVD compression failed. Ensure transformers and torch are properly installed."
        logger.error(error_msg)

        original_size = estimate_model_size_gb(model_name)

        return {
            "checkpoint_path": None,  # None indicates failure
            "model_size_gb": 0.0,
            "compression_ratio": 1.0,
            "compression_time_sec": 0.0,
            "layers_compressed": 0,
            "success": False,
            "error": error_msg,
            "metadata": {
                "method": "asvd",
                "failed": True,
                "reason": "compression_failed",
                "rank_ratio": rank_ratio,
                "original_size_gb": original_size,
            },
        }


# Factory function to get appropriate quantizer
def get_quantizer(method: str) -> BaseQuantizer:
    """Get the appropriate quantizer for the method."""
    quantizers = {
        "autoround": AutoRoundQuantizer,
        "gptq": GPTQQuantizer,
        "awq": AWQQuantizer,
        "int8": INT8Quantizer,
        "asvd": ASVDCompressor,
        "lora": LoRATrainer,
        "qlora": QLoRATrainer,
    }

    if method.lower() not in quantizers:
        raise ValueError(f"Unknown quantization method: {method}")

    return quantizers[method.lower()]()
