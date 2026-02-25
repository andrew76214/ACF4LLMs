import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { jobsApi, specApi, systemApi, experimentsApi } from '../api/client';
import type { CompressionRequest, Job } from '../types';

// Query keys
export const queryKeys = {
  jobs: ['jobs'] as const,
  job: (id: string) => ['job', id] as const,
  pareto: (id: string) => ['pareto', id] as const,
  logs: (id: string) => ['logs', id] as const,
  episodes: (id: string) => ['episodes', id] as const,
  experiments: ['experiments'] as const,
  experiment: (id: string) => ['experiment', id] as const,
  experimentEpisodes: (id: string) => ['experimentEpisodes', id] as const,
  health: ['health'] as const,
  gpu: ['gpu'] as const,
  methods: ['methods'] as const,
  benchmarks: ['benchmarks'] as const,
  presets: ['presets'] as const,
  presetConfig: (name: string) => ['presetConfig', name] as const,
  spec: (model: string, dataset: string) => ['spec', model, dataset] as const,
};

// Fetch all jobs
export function useJobs(status?: string) {
  return useQuery({
    queryKey: [...queryKeys.jobs, status],
    queryFn: () => jobsApi.list(status),
    refetchInterval: 10000, // Refetch every 10 seconds
  });
}

// Fetch single job
export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.job(jobId || ''),
    queryFn: () => jobsApi.get(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const job = query.state.data as Job | undefined;
      // Poll more frequently for running jobs
      if (job?.status === 'running' || job?.status === 'pending') {
        return 5000;
      }
      return false; // Don't poll for completed/failed jobs
    },
  });
}

// Fetch Pareto frontier
export function usePareto(jobId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.pareto(jobId || ''),
    queryFn: () => jobsApi.getPareto(jobId!),
    enabled: !!jobId,
  });
}

// Fetch job logs with polling for running jobs
export function useLogs(jobId: string | undefined, isRunning: boolean = false) {
  return useQuery({
    queryKey: queryKeys.logs(jobId || ''),
    queryFn: () => jobsApi.getLogs(jobId!),
    enabled: !!jobId,
    refetchInterval: isRunning ? 2000 : false, // Poll every 2 seconds for running jobs
  });
}

// Fetch episode history for a job
export function useEpisodes(jobId: string | undefined, isRunning: boolean = false) {
  return useQuery({
    queryKey: queryKeys.episodes(jobId || ''),
    queryFn: () => jobsApi.getEpisodes(jobId!),
    enabled: !!jobId,
    refetchInterval: isRunning ? 5000 : false, // Poll every 5 seconds for running jobs
  });
}

// Create new job
export function useCreateJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CompressionRequest) => jobsApi.create(request),
    onSuccess: () => {
      // Invalidate jobs list to refetch
      queryClient.invalidateQueries({ queryKey: queryKeys.jobs });
    },
  });
}

// Health check
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: systemApi.health,
    refetchInterval: 30000, // Check every 30 seconds
    retry: 1,
  });
}

// GPU status with detailed info
export function useGpu() {
  return useQuery({
    queryKey: queryKeys.gpu,
    queryFn: systemApi.gpu,
    refetchInterval: 10000, // Update every 10 seconds
    retry: 1,
  });
}

// Fetch available methods
export function useMethods() {
  return useQuery({
    queryKey: queryKeys.methods,
    queryFn: systemApi.methods,
    staleTime: Infinity, // Methods don't change
  });
}

// Fetch available benchmarks
export function useBenchmarks() {
  return useQuery({
    queryKey: queryKeys.benchmarks,
    queryFn: systemApi.benchmarks,
    staleTime: Infinity, // Benchmarks don't change
  });
}

// Fetch available presets
export function usePresets() {
  return useQuery({
    queryKey: queryKeys.presets,
    queryFn: systemApi.presets,
    staleTime: Infinity, // Presets don't change at runtime
  });
}

// Fetch resolved config for a preset
export function usePresetConfig(name: string | null) {
  return useQuery({
    queryKey: queryKeys.presetConfig(name || ''),
    queryFn: () => systemApi.presetConfig(name!),
    enabled: !!name,
    staleTime: Infinity,
  });
}

// Infer model spec
export function useModelSpec(modelName: string, dataset: string) {
  return useQuery({
    queryKey: queryKeys.spec(modelName, dataset),
    queryFn: () => specApi.infer(modelName, dataset),
    enabled: !!modelName && !!dataset,
    retry: 1,
  });
}

// Fetch all experiments from filesystem (includes CLI-run experiments)
export function useExperiments() {
  return useQuery({
    queryKey: queryKeys.experiments,
    queryFn: () => experimentsApi.list(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });
}

// Fetch single experiment details
export function useExperiment(experimentId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.experiment(experimentId || ''),
    queryFn: () => experimentsApi.get(experimentId!),
    enabled: !!experimentId,
  });
}

// Fetch episodes for an experiment
export function useExperimentEpisodes(experimentId: string | undefined) {
  return useQuery({
    queryKey: queryKeys.experimentEpisodes(experimentId || ''),
    queryFn: () => experimentsApi.getEpisodes(experimentId!),
    enabled: !!experimentId,
  });
}
