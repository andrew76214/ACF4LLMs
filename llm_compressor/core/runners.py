"""Model runners and serving backends (vLLM, TensorRT-LLM abstractions)."""

import asyncio
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path
import json


class ModelRunner(ABC):
    """Abstract base class for model runners."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"runner.{self.__class__.__name__}")
        self.is_running = False
        self.server_process = None
        
    @abstractmethod
    def start_server(self, model_path: str, **kwargs) -> bool:
        """Start the model server."""
        pass
    
    @abstractmethod
    def stop_server(self) -> bool:
        """Stop the model server."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text from the model."""
        pass
    
    @abstractmethod
    def benchmark(self, prompts: List[str], **kwargs) -> Dict[str, Any]:
        """Run benchmark on a list of prompts."""
        pass
    
    def health_check(self) -> bool:
        """Check if the server is healthy."""
        return self.is_running


class VLLMRunner(ModelRunner):
    """Runner for vLLM backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_url = config.get("api_url", "http://localhost:8000")
        self.port = config.get("port", 8000)
        
    def start_server(self, model_path: str, **kwargs) -> bool:
        """Start vLLM server."""
        
        try:
            # Build vLLM command
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--port", str(self.port),
                "--host", "0.0.0.0"
            ]
            
            # Add optional parameters
            if "gpu_memory_utilization" in kwargs:
                cmd.extend(["--gpu-memory-utilization", str(kwargs["gpu_memory_utilization"])])
            
            if "max_model_len" in kwargs:
                cmd.extend(["--max-model-len", str(kwargs["max_model_len"])])
            
            if "quantization" in kwargs:
                cmd.extend(["--quantization", kwargs["quantization"]])
            
            # Add tensor parallel if specified
            tensor_parallel = kwargs.get("tensor_parallel_size", 1)
            if tensor_parallel > 1:
                cmd.extend(["--tensor-parallel-size", str(tensor_parallel)])
            
            self.logger.info(f"Starting vLLM server: {' '.join(cmd)}")
            
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            max_wait = 120  # 2 minutes
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if self._check_server_ready():
                    self.is_running = True
                    self.logger.info("vLLM server started successfully")
                    return True
                time.sleep(5)
            
            self.logger.error("vLLM server failed to start within timeout")
            self.stop_server()
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start vLLM server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop vLLM server."""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=30)
                self.is_running = False
                self.logger.info("vLLM server stopped")
                return True
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
                self.is_running = False
                self.logger.warning("vLLM server forcefully killed")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping vLLM server: {e}")
                return False
        return True
    
    def _check_server_ready(self) -> bool:
        """Check if vLLM server is ready."""
        try:
            import requests
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using vLLM."""
        if not self.is_running:
            raise RuntimeError("vLLM server is not running")
        
        try:
            import requests
            
            payload = {
                "model": self.config.get("model_name", "default"),
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 100),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "stream": False
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/v1/completions",
                json=payload,
                timeout=60
            )
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result["choices"][0]["text"],
                    "latency_ms": latency * 1000,
                    "tokens_generated": result.get("usage", {}).get("completion_tokens", 0),
                    "success": True
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def benchmark(self, prompts: List[str], **kwargs) -> Dict[str, Any]:
        """Run benchmark on multiple prompts."""
        if not self.is_running:
            raise RuntimeError("vLLM server is not running")
        
        results = []
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Benchmarking prompt {i+1}/{len(prompts)}")
            
            result = self.generate(prompt, **kwargs)
            results.append(result)
            
            if result.get("success", False):
                total_tokens += result.get("tokens_generated", 0)
                total_time += result.get("latency_ms", 0) / 1000
        
        successful_results = [r for r in results if r.get("success", False)]
        
        if successful_results:
            latencies = [r["latency_ms"] for r in successful_results]
            
            benchmark_metrics = {
                "total_prompts": len(prompts),
                "successful_prompts": len(successful_results),
                "success_rate": len(successful_results) / len(prompts),
                "total_tokens": total_tokens,
                "total_time_seconds": total_time,
                "throughput_tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
                "average_latency_ms": sum(latencies) / len(latencies),
                "p50_latency_ms": sorted(latencies)[len(latencies)//2],
                "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)],
                "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)]
            }
        else:
            benchmark_metrics = {
                "total_prompts": len(prompts),
                "successful_prompts": 0,
                "success_rate": 0.0,
                "error": "No successful generations"
            }
        
        return {
            "metrics": benchmark_metrics,
            "results": results
        }


class TensorRTLLMRunner(ModelRunner):
    """Runner for TensorRT-LLM backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine_path = None
        
    def start_server(self, model_path: str, **kwargs) -> bool:
        """Start TensorRT-LLM server."""
        
        try:
            # First, build TensorRT engine if needed
            engine_path = self._build_engine(model_path, **kwargs)
            if not engine_path:
                return False
            
            self.engine_path = engine_path
            
            # Start TensorRT-LLM server
            cmd = [
                "python", "-m", "tensorrt_llm.serve",
                "--engine_dir", engine_path,
                "--tokenizer_dir", model_path,
                "--port", str(self.port)
            ]
            
            # Add optional parameters
            if "max_batch_size" in kwargs:
                cmd.extend(["--max_batch_size", str(kwargs["max_batch_size"])])
            
            if "max_input_len" in kwargs:
                cmd.extend(["--max_input_len", str(kwargs["max_input_len"])])
            
            self.logger.info(f"Starting TensorRT-LLM server: {' '.join(cmd)}")
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            max_wait = 180  # 3 minutes (TensorRT takes longer)
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if self._check_server_ready():
                    self.is_running = True
                    self.logger.info("TensorRT-LLM server started successfully")
                    return True
                time.sleep(10)
            
            self.logger.error("TensorRT-LLM server failed to start within timeout")
            self.stop_server()
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start TensorRT-LLM server: {e}")
            return False
    
    def _build_engine(self, model_path: str, **kwargs) -> Optional[str]:
        """Build TensorRT engine from model."""
        
        engine_dir = Path(f"{model_path}_trt_engine")
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        # Build command for TensorRT-LLM build
        cmd = [
            "trtllm-build",
            "--checkpoint_dir", model_path,
            "--output_dir", str(engine_dir),
            "--gemm_plugin", "float16"
        ]
        
        # Add quantization if specified
        if kwargs.get("quantization"):
            quant_type = kwargs["quantization"]
            if quant_type in ["int8", "fp8"]:
                cmd.extend(["--use_weight_only", "--weight_only_precision", quant_type])
        
        # Add parallelism settings
        tp_size = kwargs.get("tensor_parallel_size", 1)
        if tp_size > 1:
            cmd.extend(["--tp_size", str(tp_size)])
        
        # Add max sequence length
        max_seq_len = kwargs.get("max_input_len", 4096)
        cmd.extend(["--max_input_len", str(max_seq_len)])
        
        self.logger.info(f"Building TensorRT engine: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                self.logger.info("TensorRT engine built successfully")
                return str(engine_dir)
            else:
                self.logger.error(f"Engine build failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("Engine build timed out")
            return None
        except Exception as e:
            self.logger.error(f"Engine build error: {e}")
            return None
    
    def stop_server(self) -> bool:
        """Stop TensorRT-LLM server."""
        return super().stop_server()  # Use parent implementation
    
    def _check_server_ready(self) -> bool:
        """Check if TensorRT-LLM server is ready."""
        # TensorRT-LLM specific health check
        try:
            import requests
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using TensorRT-LLM."""
        # Similar to vLLM but with TensorRT-LLM specific API
        return self._call_api(prompt, **kwargs)
    
    def benchmark(self, prompts: List[str], **kwargs) -> Dict[str, Any]:
        """Run benchmark using TensorRT-LLM."""
        # Similar to vLLM implementation
        return super().benchmark(prompts, **kwargs)
    
    def _call_api(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Make API call to TensorRT-LLM server."""
        if not self.is_running:
            raise RuntimeError("TensorRT-LLM server is not running")
        
        # Implementation similar to vLLM but with TensorRT-LLM API
        try:
            import requests
            
            payload = {
                "text_input": prompt,
                "max_tokens": kwargs.get("max_tokens", 100),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9)
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=60
            )
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result.get("text_output", ""),
                    "latency_ms": latency * 1000,
                    "tokens_generated": len(result.get("text_output", "").split()),
                    "success": True
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class RunnerFactory:
    """Factory for creating model runners."""
    
    @staticmethod
    def create_runner(backend: str, config: Dict[str, Any]) -> ModelRunner:
        """Create a model runner for the specified backend."""
        
        if backend.lower() == "vllm":
            return VLLMRunner(config)
        elif backend.lower() == "tensorrt":
            return TensorRTLLMRunner(config)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    @staticmethod
    def get_available_backends() -> List[str]:
        """Get list of available backends."""
        backends = []
        
        # Check vLLM availability
        try:
            import vllm
            backends.append("vllm")
        except ImportError:
            pass
        
        # Check TensorRT-LLM availability
        try:
            import tensorrt_llm
            backends.append("tensorrt")
        except ImportError:
            pass
        
        return backends


class ModelServingManager:
    """Manager for handling multiple model runners."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("serving_manager")
        self.active_runners = {}
        
    def start_model(self, model_id: str, model_path: str, 
                   backend: str, **kwargs) -> bool:
        """Start a model with specified backend."""
        
        if model_id in self.active_runners:
            self.logger.warning(f"Model {model_id} is already running")
            return True
        
        try:
            runner = RunnerFactory.create_runner(backend, self.config.get(backend, {}))
            
            if runner.start_server(model_path, **kwargs):
                self.active_runners[model_id] = runner
                self.logger.info(f"Started model {model_id} with {backend} backend")
                return True
            else:
                self.logger.error(f"Failed to start model {model_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting model {model_id}: {e}")
            return False
    
    def stop_model(self, model_id: str) -> bool:
        """Stop a running model."""
        
        if model_id not in self.active_runners:
            self.logger.warning(f"Model {model_id} is not running")
            return True
        
        try:
            runner = self.active_runners[model_id]
            if runner.stop_server():
                del self.active_runners[model_id]
                self.logger.info(f"Stopped model {model_id}")
                return True
            else:
                self.logger.error(f"Failed to stop model {model_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping model {model_id}: {e}")
            return False
    
    def get_runner(self, model_id: str) -> Optional[ModelRunner]:
        """Get runner for a model."""
        return self.active_runners.get(model_id)
    
    def list_active_models(self) -> List[str]:
        """List all active model IDs."""
        return list(self.active_runners.keys())
    
    def stop_all_models(self) -> bool:
        """Stop all running models."""
        success = True
        
        for model_id in list(self.active_runners.keys()):
            if not self.stop_model(model_id):
                success = False
        
        return success