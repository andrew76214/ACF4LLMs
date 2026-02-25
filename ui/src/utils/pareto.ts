import type { ParetoPoint, ParetoSolution } from '../types';

/** Convert Pareto solutions to chart points */
export function solutionsToPoints(solutions: ParetoSolution[]): ParetoPoint[] {
  return solutions.map((sol) => ({
    strategyId: sol.strategy.strategy_id,
    accuracy: sol.result.accuracy,
    latency: sol.result.latency_ms,
    memory: sol.result.memory_gb,
    size: sol.result.model_size_gb,
    co2: sol.result.co2_grams || 0,
    method: sol.strategy.methods[0] || 'unknown',
    bits: sol.strategy.quantization_bits,
  }));
}
