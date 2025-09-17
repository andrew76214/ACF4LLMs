"""Metrics collection and analysis utilities."""

import time
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class MetricSample:
    """A single metric sample with timestamp."""
    timestamp: float
    value: Union[float, int, str]
    metadata: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """Collect and aggregate various metrics during optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("metrics")
        self.metrics_history = {}
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: Union[float, int, str], 
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a single metric value."""
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        sample = MetricSample(
            timestamp=time.time(),
            value=value,
            metadata=metadata or {}
        )
        
        self.metrics_history[name].append(sample)
        
    def get_metric_history(self, name: str) -> List[MetricSample]:
        """Get history for a specific metric."""
        return self.metrics_history.get(name, [])
    
    def get_latest_metric(self, name: str) -> Optional[MetricSample]:
        """Get the most recent value for a metric."""
        history = self.get_metric_history(name)
        return history[-1] if history else None
    
    def calculate_statistics(self, name: str) -> Dict[str, float]:
        """Calculate statistics for a metric."""
        history = self.get_metric_history(name)
        
        if not history:
            return {}
        
        # Extract numeric values only
        numeric_values = []
        for sample in history:
            if isinstance(sample.value, (int, float)):
                numeric_values.append(float(sample.value))
        
        if not numeric_values:
            return {}
        
        values = np.array(numeric_values)
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))
        }
    
    def aggregate_metrics(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Aggregate all metrics over a time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else self.start_time
        
        aggregated = {}
        
        for metric_name, history in self.metrics_history.items():
            # Filter by time window
            filtered_samples = [
                sample for sample in history 
                if sample.timestamp >= cutoff_time
            ]
            
            if not filtered_samples:
                continue
            
            # Get latest value
            latest = filtered_samples[-1]
            aggregated[f"{metric_name}_latest"] = latest.value
            
            # Calculate statistics for numeric metrics
            numeric_values = [
                sample.value for sample in filtered_samples
                if isinstance(sample.value, (int, float))
            ]
            
            if numeric_values:
                values = np.array(numeric_values)
                aggregated.update({
                    f"{metric_name}_mean": float(np.mean(values)),
                    f"{metric_name}_std": float(np.std(values)),
                    f"{metric_name}_min": float(np.min(values)),
                    f"{metric_name}_max": float(np.max(values))
                })
        
        return aggregated
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {}
        
        # CPU metrics
        metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
        metrics["cpu_count"] = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics["memory_total_gb"] = memory.total / (1024**3)
        metrics["memory_used_gb"] = memory.used / (1024**3)
        metrics["memory_percent"] = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics["disk_total_gb"] = disk.total / (1024**3)
        metrics["disk_used_gb"] = disk.used / (1024**3)
        metrics["disk_percent"] = (disk.used / disk.total) * 100
        
        # GPU metrics (if available)
        gpu_metrics = self._collect_gpu_metrics()
        metrics.update(gpu_metrics)
        
        # Record all system metrics
        for name, value in metrics.items():
            self.record_metric(f"system_{name}", value)
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics if available."""
        gpu_metrics = {}
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_metrics.update({
                    "gpu_utilization": gpu.load * 100,
                    "gpu_memory_used_gb": gpu.memoryUsed / 1024,
                    "gpu_memory_total_gb": gpu.memoryTotal / 1024,
                    "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_temperature": gpu.temperature
                })
        except ImportError:
            # Try nvidia-ml-py as fallback
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_metrics.update({
                    "gpu_memory_used_gb": mem_info.used / (1024**3),
                    "gpu_memory_total_gb": mem_info.total / (1024**3),
                    "gpu_memory_percent": (mem_info.used / mem_info.total) * 100
                })
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_metrics["gpu_utilization"] = util.gpu
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_metrics["gpu_temperature"] = temp
                
            except Exception as e:
                self.logger.debug(f"Could not collect GPU metrics: {e}")
        
        return gpu_metrics
    
    def estimate_energy_consumption(self, duration_seconds: float, 
                                   gpu_utilization: float = 0.8) -> Dict[str, Any]:
        """Estimate energy consumption for a time period."""
        
        # Base power consumption estimates (watts)
        cpu_base_power = 100  # Typical server CPU
        gpu_base_power = 300  # A100 80GB
        system_base_power = 50  # Other components
        
        # Calculate actual consumption based on utilization
        cpu_power = cpu_base_power * 0.3  # Assume 30% CPU utilization
        gpu_power = gpu_base_power * gpu_utilization
        system_power = system_base_power
        
        total_power = cpu_power + gpu_power + system_power
        
        # Energy consumption
        energy_kwh = (total_power * duration_seconds) / (3600 * 1000)  # Convert to kWh
        
        # Carbon footprint (using grid emission factor)
        grid_factor = self.config.get("grid_emission_factor_kg_per_kwh", 0.4)
        co2_emissions_kg = energy_kwh * grid_factor
        
        energy_metrics = {
            "duration_seconds": duration_seconds,
            "cpu_power_watts": cpu_power,
            "gpu_power_watts": gpu_power,
            "system_power_watts": system_power,
            "total_power_watts": total_power,
            "energy_consumption_kwh": energy_kwh,
            "co2_emissions_kg": co2_emissions_kg,
            "grid_emission_factor": grid_factor
        }
        
        # Record energy metrics
        for name, value in energy_metrics.items():
            self.record_metric(f"energy_{name}", value)
        
        return energy_metrics
    
    def calculate_efficiency_metrics(self, accuracy: float, latency: float,
                                   energy: float) -> Dict[str, Any]:
        """Calculate efficiency metrics."""
        
        # Performance per watt
        perf_per_watt = accuracy / energy if energy > 0 else 0
        
        # Throughput efficiency (higher is better)
        throughput_efficiency = 1.0 / latency if latency > 0 else 0
        
        # Energy efficiency score (accuracy per unit energy)
        energy_efficiency = accuracy / energy if energy > 0 else 0
        
        # Overall efficiency score
        overall_efficiency = (accuracy * throughput_efficiency) / energy if energy > 0 else 0
        
        efficiency_metrics = {
            "performance_per_watt": perf_per_watt,
            "throughput_efficiency": throughput_efficiency,
            "energy_efficiency": energy_efficiency,
            "overall_efficiency": overall_efficiency
        }
        
        # Record efficiency metrics
        for name, value in efficiency_metrics.items():
            self.record_metric(f"efficiency_{name}", value)
        
        return efficiency_metrics
    
    def export_metrics(self, output_path: str, format: str = "json") -> str:
        """Export collected metrics to file."""
        
        # Prepare export data
        export_data = {
            "collection_start": self.start_time,
            "collection_end": time.time(),
            "total_duration": time.time() - self.start_time,
            "metrics": {}
        }
        
        # Export metric histories
        for metric_name, history in self.metrics_history.items():
            export_data["metrics"][metric_name] = {
                "samples": [
                    {
                        "timestamp": sample.timestamp,
                        "value": sample.value,
                        "metadata": sample.metadata
                    }
                    for sample in history
                ],
                "statistics": self.calculate_statistics(metric_name)
            }
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported metrics to: {output_path}")
        return str(output_path)
    
    def create_metrics_dashboard(self, output_path: str = "reports/metrics_dashboard.html"):
        """Create an interactive metrics dashboard."""
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots for different metric categories
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("System Metrics", "Energy Consumption", 
                               "Performance Metrics", "Efficiency Trends"),
                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                       [{"secondary_y": True}, {"secondary_y": True}]]
            )
            
            # Plot system metrics
            self._add_system_metrics_to_plot(fig, row=1, col=1)
            
            # Plot energy metrics
            self._add_energy_metrics_to_plot(fig, row=1, col=2)
            
            # Plot performance metrics
            self._add_performance_metrics_to_plot(fig, row=2, col=1)
            
            # Plot efficiency metrics
            self._add_efficiency_metrics_to_plot(fig, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title="LLM Compression Metrics Dashboard",
                showlegend=True,
                height=800
            )
            
            # Save dashboard
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path))
            
            self.logger.info(f"Created metrics dashboard: {output_path}")
            return str(output_path)
            
        except ImportError:
            self.logger.warning("Plotly not available, skipping dashboard creation")
            return ""
    
    def _add_system_metrics_to_plot(self, fig, row: int, col: int):
        """Add system metrics to subplot."""
        # Implementation would add CPU, memory, GPU utilization over time
        pass
    
    def _add_energy_metrics_to_plot(self, fig, row: int, col: int):
        """Add energy metrics to subplot."""
        # Implementation would add power consumption and CO2 emissions over time
        pass
    
    def _add_performance_metrics_to_plot(self, fig, row: int, col: int):
        """Add performance metrics to subplot."""
        # Implementation would add latency, throughput over time
        pass
    
    def _add_efficiency_metrics_to_plot(self, fig, row: int, col: int):
        """Add efficiency metrics to subplot."""
        # Implementation would add efficiency scores over time
        pass
    
    def reset(self):
        """Reset all collected metrics."""
        self.metrics_history.clear()
        self.start_time = time.time()
        self.logger.info("Metrics collector reset")