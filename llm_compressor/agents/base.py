"""Base agent class for all compression and optimization agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
import time


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    

class BaseAgent(ABC):
    """Base class for all compression/optimization agents."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"agent.{name}")
        
    @abstractmethod
    def execute(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Execute the agent with given recipe and context."""
        pass
    
    @abstractmethod  
    def validate_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Validate if the recipe is compatible with this agent."""
        pass
    
    def get_search_space(self) -> Dict[str, Any]:
        """Return the hyperparameter search space for this agent."""
        return {}
    
    def estimate_cost(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        """Estimate computational cost (time, memory, energy) for a recipe."""
        return {"time": 1.0, "memory": 1.0, "energy": 1.0}
    
    def _log_execution(self, recipe: Dict[str, Any], result: AgentResult):
        """Log execution details."""
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(
            f"Agent {self.name} {status} in {result.execution_time:.2f}s - "
            f"Recipe: {recipe.get('name', 'unknown')}"
        )
        if not result.success and result.error:
            self.logger.error(f"Error: {result.error}")
            
    def run(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Run the agent with timing and logging."""
        start_time = time.time()
        
        try:
            if not self.validate_recipe(recipe):
                return AgentResult(
                    success=False, 
                    metrics={}, 
                    artifacts={},
                    error=f"Invalid recipe for agent {self.name}"
                )
            
            result = self.execute(recipe, context)
            result.execution_time = time.time() - start_time
            
        except Exception as e:
            result = AgentResult(
                success=False,
                metrics={},
                artifacts={},
                error=str(e),
                execution_time=time.time() - start_time
            )
            
        self._log_execution(recipe, result)
        return result