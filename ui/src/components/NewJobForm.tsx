import { useState, useReducer, useEffect, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, Sparkles, AlertCircle } from 'lucide-react';
import { useCreateJob, useMethods, useBenchmarks, useModelSpec, usePresetConfig } from '../hooks/useJobs';
import { PresetSelector } from './PresetSelector';
import { AdvancedConfigForm } from './AdvancedConfigForm';
import type { CompressionMethod, CompressionRequest, PresetName, AdvancedConfig } from '../types';

// State
interface FormState {
  modelName: string;
  dataset: string;
  maxEpisodes: number;
  selectedMethods: CompressionMethod[];
  useMock: boolean;
  preset: PresetName | null;
  presetModified: boolean;
  advancedConfig: AdvancedConfig;
}

type FormAction =
  | { type: 'SET_FIELD'; field: keyof FormState; value: unknown }
  | { type: 'SELECT_PRESET'; preset: PresetName | null }
  | { type: 'APPLY_PRESET_CONFIG'; config: AdvancedConfig }
  | { type: 'SET_ADVANCED_CONFIG'; config: AdvancedConfig }
  | { type: 'TOGGLE_METHOD'; method: CompressionMethod };

const initialState: FormState = {
  modelName: '',
  dataset: 'gsm8k',
  maxEpisodes: 10,
  selectedMethods: [],
  useMock: false,
  preset: null,
  presetModified: false,
  advancedConfig: {},
};

function formReducer(state: FormState, action: FormAction): FormState {
  switch (action.type) {
    case 'SET_FIELD':
      return { ...state, [action.field]: action.value };
    case 'SELECT_PRESET':
      return {
        ...state,
        preset: action.preset,
        presetModified: false,
        // Clear advanced config when deselecting preset
        advancedConfig: action.preset ? state.advancedConfig : {},
      };
    case 'APPLY_PRESET_CONFIG':
      return {
        ...state,
        advancedConfig: action.config,
        maxEpisodes: action.config.termination?.max_episodes ?? state.maxEpisodes,
        presetModified: false,
      };
    case 'SET_ADVANCED_CONFIG':
      return {
        ...state,
        advancedConfig: action.config,
        presetModified: state.preset !== null,
      };
    case 'TOGGLE_METHOD': {
      const methods = state.selectedMethods.includes(action.method)
        ? state.selectedMethods.filter((m) => m !== action.method)
        : [...state.selectedMethods, action.method];
      return { ...state, selectedMethods: methods };
    }
    default:
      return state;
  }
}

export function NewJobForm() {
  const navigate = useNavigate();
  const createJob = useCreateJob();
  const { data: methods } = useMethods();
  const { data: benchmarks } = useBenchmarks();

  const [state, dispatch] = useReducer(formReducer, initialState);

  // Fetch preset config when preset changes
  const { data: presetConfig } = usePresetConfig(state.preset);

  // Apply preset config when it loads
  const [appliedPreset, setAppliedPreset] = useState<string | null>(null);
  useEffect(() => {
    if (presetConfig && state.preset && state.preset !== appliedPreset) {
      dispatch({ type: 'APPLY_PRESET_CONFIG', config: presetConfig });
      setAppliedPreset(state.preset);
    }
  }, [presetConfig, state.preset, appliedPreset]);

  // Model spec inference
  const {
    data: modelSpec,
    isLoading: specLoading,
    error: specError,
  } = useModelSpec(state.modelName, state.dataset);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    const request: CompressionRequest = {
      model_name: state.modelName,
      dataset: state.dataset,
      max_episodes: state.maxEpisodes,
      compression_methods: state.selectedMethods.length > 0 ? state.selectedMethods : undefined,
      use_mock: state.useMock,
      preset: state.preset ?? undefined,
      advanced_config: state.presetModified || !state.preset
        ? Object.keys(state.advancedConfig).length > 0
          ? state.advancedConfig
          : undefined
        : undefined,
    };

    try {
      const job = await createJob.mutateAsync(request);
      navigate(`/experiments/${job.job_id}`);
    } catch (error) {
      console.error('Failed to create job:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Preset Selector */}
      <PresetSelector
        selected={state.preset}
        onSelect={(preset) => {
          dispatch({ type: 'SELECT_PRESET', preset });
          if (!preset) setAppliedPreset(null);
        }}
        modified={state.presetModified}
      />

      {/* Divider */}
      <div className="border-t border-gray-200" />

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
          value={state.modelName}
          onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'modelName', value: e.target.value })}
          placeholder="e.g., gpt2, meta-llama/Meta-Llama-3-8B-Instruct"
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
          required
        />
        <p className="mt-1 text-xs text-gray-500">
          Enter a HuggingFace model path or local model name
        </p>
      </div>

      {/* Model Spec Preview */}
      {state.modelName && (
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
          value={state.dataset}
          onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'dataset', value: e.target.value })}
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
          value={state.maxEpisodes}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10);
            dispatch({ type: 'SET_FIELD', field: 'maxEpisodes', value: val });
            // Sync with advanced config termination
            dispatch({
              type: 'SET_ADVANCED_CONFIG',
              config: {
                ...state.advancedConfig,
                termination: { ...state.advancedConfig.termination, max_episodes: val },
              },
            });
          }}
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
                onClick={() => dispatch({ type: 'TOGGLE_METHOD', method: method as CompressionMethod })}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  state.selectedMethods.includes(method as CompressionMethod)
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
          checked={state.useMock}
          onChange={(e) => dispatch({ type: 'SET_FIELD', field: 'useMock', value: e.target.checked })}
          className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
        />
        <label htmlFor="useMock" className="text-sm text-gray-600">
          Use mock mode (for testing without GPU)
        </label>
      </div>

      {/* Advanced Config */}
      <div className="border-t border-gray-200 pt-4">
        <AdvancedConfigForm
          config={state.advancedConfig}
          onChange={(config) => dispatch({ type: 'SET_ADVANCED_CONFIG', config })}
        />
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={createJob.isPending || !state.modelName}
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
