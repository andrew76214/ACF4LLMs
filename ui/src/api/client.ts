import axios, { type AxiosInstance } from 'axios';
import type {
  Job,
  CompressionRequest,
  ModelSpec,
  ParetoFrontier,
  HealthCheck,
  GpuStatus,
  CompressionMethod,
  Benchmark,
  LogsResponse,
} from '../types';

// API base URL - uses proxy in development, env variable in production
const API_URL = import.meta.env.VITE_API_URL || '/api';

// Create axios instance with defaults
const api: AxiosInstance = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'Unknown error';
    console.error(`[API Error] ${message}`);
    return Promise.reject(new Error(message));
  }
);

// Jobs API
export const jobsApi = {
  // List all jobs with optional filtering
  list: async (status?: string, limit: number = 100): Promise<Job[]> => {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('limit', limit.toString());
    const response = await api.get<Job[]>(`/jobs?${params}`);
    return response.data;
  },

  // Get single job by ID
  get: async (jobId: string): Promise<Job> => {
    const response = await api.get<Job>(`/jobs/${jobId}`);
    return response.data;
  },

  // Create new compression job
  create: async (request: CompressionRequest): Promise<Job> => {
    const response = await api.post<Job>('/compress', request);
    return response.data;
  },

  // Get Pareto frontier for a job
  getPareto: async (jobId: string): Promise<ParetoFrontier> => {
    const response = await api.get<ParetoFrontier>(`/pareto/${jobId}`);
    return response.data;
  },

  // Get logs for a job
  getLogs: async (jobId: string, offset: number = 0, limit: number = 200): Promise<LogsResponse> => {
    const params = new URLSearchParams();
    params.append('offset', offset.toString());
    params.append('limit', limit.toString());
    const response = await api.get<LogsResponse>(`/jobs/${jobId}/logs?${params}`);
    return response.data;
  },
};

// Model specification API
export const specApi = {
  // Infer model specification
  infer: async (modelName: string, dataset: string): Promise<ModelSpec> => {
    const response = await api.post<ModelSpec>('/spec/infer', {
      model_name: modelName,
      dataset,
    });
    return response.data;
  },
};

// System API
export const systemApi = {
  // Health check
  health: async (): Promise<HealthCheck> => {
    const response = await api.get<HealthCheck>('/health');
    return response.data;
  },

  // GPU status with detailed info
  gpu: async (): Promise<GpuStatus> => {
    const response = await api.get<GpuStatus>('/gpu');
    return response.data;
  },

  // List available compression methods
  methods: async (): Promise<CompressionMethod[]> => {
    const response = await api.get<CompressionMethod[]>('/methods');
    return response.data;
  },

  // List available benchmarks
  benchmarks: async (): Promise<Benchmark[]> => {
    const response = await api.get<Benchmark[]>('/benchmarks');
    return response.data;
  },
};

export default api;
