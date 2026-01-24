// TypeScript types for Green AI Dashboard

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export type CompressionMethod =
  | 'autoround'
  | 'gptq'
  | 'int8'
  | 'awq'
  | 'pruning'
  | 'distillation'
  | 'lora'
  | 'qlora';

export type Benchmark =
  | 'gsm8k'
  | 'commonsenseqa'
  | 'truthfulqa'
  | 'humaneval'
  | 'bigbench_hard'
  | 'mmlu'
  | 'hellaswag'
  | 'arc_easy'
  | 'arc_challenge'
  | 'winogrande';

export interface Job {
  job_id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  progress: JobProgress;
  result: JobResult | null;
  error: string | null;
}

export interface JobProgress {
  current_episode: number;
  max_episodes: number;
  pareto_solutions: number;
}

export interface JobResult {
  best_solution: ParetoSolution | null;
  pareto_frontier_size: number;
  total_strategies_tried: number;
  compression_achieved: number;
}

export interface CompressionRequest {
  model_name: string;
  dataset: string;
  max_episodes: number;
  target_metric?: string;
  compression_methods?: CompressionMethod[];
  constraints?: Record<string, number>;
  use_mock?: boolean;
}

export interface ModelSpec {
  model_name: string;
  model_size_gb: number;
  model_family: string;
  parameter_count: string | null;
  preferred_methods: CompressionMethod[];
  calibration_samples: number;
  max_sequence_length: number;
  min_vram_gb: number;
  recommended_vram_gb: number;
  primary_objective: string;
  accuracy_threshold: number;
  target_speedup: number;
  target_memory_reduction: number;
}

export interface CompressionStrategy {
  episode_id: number;
  strategy_id: string;
  methods: CompressionMethod[];
  quantization_bits: number | null;
  quantization_method: string | null;
  pruning_ratio: number | null;
  pruning_method: string | null;
  pruning_granularity: string | null;
  distillation_teacher: string | null;
  lora_rank: number | null;
  do_finetune: boolean;
  finetune_steps: number | null;
  calibration_dataset: string | null;
  created_at: string;
}

export interface EvaluationResult {
  strategy_id: string;
  checkpoint_path: string;
  model_size_gb: number;
  compression_ratio: number;
  accuracy: number;
  latency_ms: number;
  throughput_tokens_per_sec: number;
  memory_gb: number;
  energy_joules: number | null;
  energy_kwh: number | null;
  co2_grams: number | null;
  co2_kg: number | null;
  num_inferences_measured: number | null;
  is_carbon_mock: boolean;
  benchmark_scores: Record<string, number>;
  is_pareto_optimal: boolean;
  evaluation_time_sec: number;
  evaluated_at: string;
}

export interface ParetoSolution {
  strategy: CompressionStrategy;
  result: EvaluationResult;
  dominates: string[];
}

export interface ParetoFrontier {
  solutions: ParetoSolution[];
  message?: string;
}

export interface HealthCheck {
  status: string;
  cuda_available: string;
  gpu_count: string;
}

export interface GpuInfo {
  index: number;
  name: string;
  memory_total_gb: number;
  memory_used_gb: number;
  memory_reserved_gb: number;
  memory_utilization: number;
  compute_capability: string;
}

export interface GpuStatus {
  available: boolean;
  gpus: GpuInfo[];
}

// Logs response from API
export interface LogsResponse {
  logs: string[];
  total: number;
  offset: number;
  limit: number;
}

// Chart data types
export interface ParetoPoint {
  strategyId: string;
  accuracy: number;
  latency: number;
  memory: number;
  size: number;
  co2: number;
  method: string;
  bits: number | null;
}
