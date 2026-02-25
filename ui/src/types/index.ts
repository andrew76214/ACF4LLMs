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

export interface BestSolutions {
  accuracy: ParetoSolution | null;
  latency: ParetoSolution | null;
  size: ParetoSolution | null;
  carbon: ParetoSolution | null;
  balanced: ParetoSolution | null;
}

export interface JobResult {
  best_solutions: BestSolutions | null;
  pareto_frontier_size: number;
  total_strategies_tried: number;
  compression_achieved: number;
}

// Advanced config section interfaces (all fields optional for partial overrides)
export interface CoordinatorConfig {
  llm_model?: string;
  llm_temperature?: number;
  worker_model?: string;
  worker_temperature?: number;
  max_retries?: number;
}

export interface QuantizationConfig {
  default_bit_width?: number;
  default_method?: 'autoround' | 'gptq' | 'awq' | 'int8';
  group_size?: number;
  calibration_samples?: number;
  calibration_seq_length?: number;
  sym?: boolean;
}

export interface EvaluationConfig {
  use_proxy?: boolean;
  proxy_samples?: number;
  full_eval_samples?: number | null;
  batch_size?: number;
  measure_carbon?: boolean;
  carbon_inference_count?: number;
  device?: string | null;
}

export interface SearchConfig {
  method?: 'random' | 'bayesian' | 'evolutionary' | 'bandit';
  exploration_ratio?: number;
  ucb_exploration_param?: number;
  mutation_rate?: number;
  population_size?: number;
}

export interface RewardConfig {
  accuracy_weight?: number;
  latency_weight?: number;
  memory_weight?: number;
  energy_weight?: number;
  min_accuracy?: number;
  max_latency_ms?: number | null;
  max_memory_gb?: number | null;
}

export interface FinetuningConfig {
  lora_rank?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  learning_rate?: number;
  num_train_steps?: number;
  warmup_steps?: number;
  gradient_accumulation_steps?: number;
  target_modules?: string[] | null;
}

export interface TerminationConfig {
  max_episodes?: number;
  budget_hours?: number;
  convergence_patience?: number;
  min_improvement?: number;
}

export interface AdvancedConfig {
  coordinator?: CoordinatorConfig;
  quantization?: QuantizationConfig;
  evaluation?: EvaluationConfig;
  search?: SearchConfig;
  reward?: RewardConfig;
  finetuning?: FinetuningConfig;
  termination?: TerminationConfig;
}

export type PresetName =
  | 'accuracy_focused'
  | 'latency_focused'
  | 'balanced'
  | 'memory_constrained'
  | 'low_carbon'
  | 'low_cost';

export interface PresetInfo {
  name: PresetName;
  label: string;
  description: string;
  icon: string;
}

export interface CompressionRequest {
  model_name: string;
  dataset: string;
  max_episodes: number;
  target_metric?: string;
  compression_methods?: CompressionMethod[];
  constraints?: Record<string, number>;
  use_mock?: boolean;
  preset?: PresetName;
  advanced_config?: AdvancedConfig;
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

/** Raw pareto_frontier.json shape written by pareto.py (uses 'frontier' key) */
export interface FilesystemParetoData {
  frontier: ParetoSolution[];
  history?: Record<string, unknown>[];
  metadata?: {
    num_solutions: number;
    num_evaluated: number;
    last_updated: string;
  };
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

// Episode types for coordinator reasoning display
export interface EpisodeDecisionParams {
  bits?: number;
  lora_rank?: number;
  pruning_ratio?: number;
  pruning_method?: string;
  asvd_rank_ratio?: number;
  pipeline_name?: string;
}

export interface EpisodeDecision {
  episode_id: number;
  action: string;
  method: string | null;
  reasoning: string;
  params: EpisodeDecisionParams;
  timestamp: string;
  skill_recommendations?: Array<{
    skill_name: string;
    reasoning: string;
    confidence: number;
  }> | null;
}

export interface Episode {
  episode_id: number;
  decision: EpisodeDecision;
  strategy: CompressionStrategy & {
    coordinator_reasoning?: string;
    coordinator_decision_timestamp?: string;
    pipeline_name?: string;
  };
  result: EvaluationResult | null;
  is_pareto: boolean;
}

export interface EpisodesResponse {
  episodes: Episode[];
  total: number;
  message?: string;
}

// Experiment from filesystem (CLI-run experiments)
export interface Experiment {
  experiment_id: string;
  experiment_dir: string;
  model_name: string;
  dataset: string;
  status: 'completed' | 'running' | 'failed';
  created_at: string;
  episodes_completed: number;
  pareto_solutions: number;
  best_accuracy: number | null;
  best_compression: number | null;
  best_co2_grams: number | null;
  visualization_path: string | null;
}

export interface ExperimentDetail {
  experiment_id: string;
  experiment_dir: string;
  results: {
    experiment_name: string;
    model: string;
    dataset: string;
    episodes_completed: number;
    frontier_summary: {
      num_solutions: number;
      best_accuracy: number;
      best_latency: number;
      best_size: number;
      best_co2_grams: number | null;
    };
    best_solutions: BestSolutions;
    visualization: string;
  };
  model_spec: ModelSpec;
  pareto_frontier: FilesystemParetoData | null;
}
