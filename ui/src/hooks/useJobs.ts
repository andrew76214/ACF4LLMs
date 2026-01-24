import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { jobsApi, specApi, systemApi } from '../api/client';
import type { CompressionRequest, Job } from '../types';

// Query keys
export const queryKeys = {
  jobs: ['jobs'] as const,
  job: (id: string) => ['job', id] as const,
  pareto: (id: string) => ['pareto', id] as const,
  logs: (id: string) => ['logs', id] as const,
  health: ['health'] as const,
  gpu: ['gpu'] as const,
  methods: ['methods'] as const,
  benchmarks: ['benchmarks'] as const,
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

// Infer model spec
export function useModelSpec(modelName: string, dataset: string) {
  return useQuery({
    queryKey: queryKeys.spec(modelName, dataset),
    queryFn: () => specApi.infer(modelName, dataset),
    enabled: !!modelName && !!dataset,
    retry: 1,
  });
}
