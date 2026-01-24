import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, Sparkles, AlertCircle } from 'lucide-react';
import { useCreateJob, useMethods, useBenchmarks, useModelSpec } from '../hooks/useJobs';
import type { CompressionMethod, CompressionRequest } from '../types';

export function NewJobForm() {
  const navigate = useNavigate();
  const createJob = useCreateJob();
  const { data: methods } = useMethods();
  const { data: benchmarks } = useBenchmarks();

  // Form state
  const [modelName, setModelName] = useState('');
  const [dataset, setDataset] = useState('gsm8k');
  const [maxEpisodes, setMaxEpisodes] = useState(10);
  const [selectedMethods, setSelectedMethods] = useState<CompressionMethod[]>([]);
  const [useMock, setUseMock] = useState(false);

  // Model spec inference
  const {
    data: modelSpec,
    isLoading: specLoading,
    error: specError,
  } = useModelSpec(modelName, dataset);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const request: CompressionRequest = {
      model_name: modelName,
      dataset,
      max_episodes: maxEpisodes,
      compression_methods: selectedMethods.length > 0 ? selectedMethods : undefined,
      use_mock: useMock,
    };

    try {
      const job = await createJob.mutateAsync(request);
      navigate(`/experiments/${job.job_id}`);
    } catch (error) {
      // Error is handled by React Query
      console.error('Failed to create job:', error);
    }
  };

  const toggleMethod = (method: CompressionMethod) => {
    setSelectedMethods((prev) =>
      prev.includes(method)
        ? prev.filter((m) => m !== method)
        : [...prev, method]
    );
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Model Name */}
      <div>
        <label
          htmlFor="modelName"
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Model Name
        </label>
        <input
          type="text"
          id="modelName"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          placeholder="e.g., gpt2, meta-llama/Meta-Llama-3-8B-Instruct"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
          required
        />
        <p className="mt-1 text-xs text-gray-500">
          Enter a HuggingFace model path or local model name
        </p>
      </div>

      {/* Model Spec Preview */}
      {modelName && (
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
            <Sparkles className="w-4 h-4" />
            Inferred Model Spec
          </h4>
          {specLoading ? (
            <div className="flex items-center gap-2 text-gray-500">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Analyzing model...</span>
            </div>
          ) : specError ? (
            <div className="flex items-center gap-2 text-yellow-600">
              <AlertCircle className="w-4 h-4" />
              <span className="text-sm">Could not infer model specification</span>
            </div>
          ) : modelSpec ? (
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-500">Family:</span>{' '}
                <span className="font-medium">{modelSpec.model_family}</span>
              </div>
              <div>
                <span className="text-gray-500">Size:</span>{' '}
                <span className="font-medium">{modelSpec.model_size_gb.toFixed(1)} GB</span>
              </div>
              <div>
                <span className="text-gray-500">Parameters:</span>{' '}
                <span className="font-medium">{modelSpec.parameter_count || 'Unknown'}</span>
              </div>
              <div>
                <span className="text-gray-500">Min VRAM:</span>{' '}
                <span className="font-medium">{modelSpec.min_vram_gb.toFixed(1)} GB</span>
              </div>
            </div>
          ) : null}
        </div>
      )}

      {/* Dataset */}
      <div>
        <label
          htmlFor="dataset"
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Evaluation Dataset
        </label>
        <select
          id="dataset"
          value={dataset}
          onChange={(e) => setDataset(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
        >
          {(Array.isArray(benchmarks) ? benchmarks : ['gsm8k', 'commonsenseqa', 'truthfulqa', 'mmlu']).map(
            (b) => (
              <option key={b} value={b}>
                {b.toUpperCase()}
              </option>
            )
          )}
        </select>
      </div>

      {/* Max Episodes */}
      <div>
        <label
          htmlFor="maxEpisodes"
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Max Episodes
        </label>
        <input
          type="number"
          id="maxEpisodes"
          value={maxEpisodes}
          onChange={(e) => setMaxEpisodes(parseInt(e.target.value, 10))}
          min={1}
          max={100}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
        />
        <p className="mt-1 text-xs text-gray-500">
          Number of compression strategies to try (recommended: 5-20)
        </p>
      </div>

      {/* Compression Methods */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Compression Methods (optional)
        </label>
        <div className="flex flex-wrap gap-2">
          {(Array.isArray(methods) ? methods : ['autoround', 'gptq', 'int8', 'awq', 'pruning', 'lora', 'qlora']).map(
            (method) => (
              <button
                key={method}
                type="button"
                onClick={() => toggleMethod(method as CompressionMethod)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  selectedMethods.includes(method as CompressionMethod)
                    ? 'bg-primary-100 text-primary-700 border-2 border-primary-500'
                    : 'bg-gray-100 text-gray-600 border-2 border-transparent hover:bg-gray-200'
                }`}
              >
                {method.toUpperCase()}
              </button>
            )
          )}
        </div>
        <p className="mt-1 text-xs text-gray-500">
          Leave empty to let the AI decide automatically
        </p>
      </div>

      {/* Mock Mode */}
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="useMock"
          checked={useMock}
          onChange={(e) => setUseMock(e.target.checked)}
          className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
        />
        <label htmlFor="useMock" className="text-sm text-gray-600">
          Use mock mode (for testing without GPU)
        </label>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={createJob.isPending || !modelName}
        className="w-full py-3 px-4 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
      >
        {createJob.isPending ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Starting Compression...
          </>
        ) : (
          <>
            <Sparkles className="w-5 h-5" />
            Start Compression
          </>
        )}
      </button>

      {/* Error message */}
      {createJob.isError && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm flex items-center gap-2">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{createJob.error?.message || 'Failed to start compression job'}</span>
        </div>
      )}
    </form>
  );
}
