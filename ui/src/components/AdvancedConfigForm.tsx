import { useState } from 'react';
import { Settings } from 'lucide-react';
import { ConfigSection } from './ConfigSection';
import type { AdvancedConfig } from '../types';

interface AdvancedConfigFormProps {
  config: AdvancedConfig;
  onChange: (config: AdvancedConfig) => void;
}

// Helper: update a nested section
function updateSection<K extends keyof AdvancedConfig>(
  config: AdvancedConfig,
  section: K,
  field: string,
  value: unknown,
): AdvancedConfig {
  return {
    ...config,
    [section]: {
      ...config[section],
      [field]: value,
    },
  };
}

// Reusable field components
function SliderField({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  format,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: (v: number) => string;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs font-medium text-gray-600">{label}</label>
        <span className="text-xs text-gray-500 font-mono">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
      />
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
      />
    </div>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

function ToggleField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between">
      <label className="text-xs font-medium text-gray-600">{label}</label>
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
          value ? 'bg-primary-600' : 'bg-gray-300'
        }`}
      >
        <span
          className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
            value ? 'translate-x-4.5' : 'translate-x-0.5'
          }`}
        />
      </button>
    </div>
  );
}

function ButtonGroupField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string | number;
  onChange: (v: string | number) => void;
  options: { value: string | number; label: string }[];
}) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
      <div className="flex flex-wrap gap-1">
        {options.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            onClick={() => onChange(o.value)}
            className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
              value === o.value
                ? 'bg-primary-100 text-primary-700 border border-primary-500'
                : 'bg-gray-100 text-gray-600 border border-transparent hover:bg-gray-200'
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function AdvancedConfigForm({ config, onChange }: AdvancedConfigFormProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({});

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const c = config;

  return (
    <div>
      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
      >
        <Settings className="w-4 h-4" />
        {showAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
      </button>

      {showAdvanced && (
        <div className="mt-4 space-y-3">
          {/* 1. Coordinator */}
          <ConfigSection
            title="Coordinator"
            subtitle="LLM settings"
            isOpen={!!openSections.coordinator}
            onToggle={() => toggleSection('coordinator')}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <SelectField
                label="LLM Model"
                value={c.coordinator?.llm_model ?? 'gpt-4o'}
                onChange={(v) => onChange(updateSection(c, 'coordinator', 'llm_model', v))}
                options={[
                  { value: 'gpt-4o', label: 'GPT-4o' },
                  { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
                  { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
                ]}
              />
              <SliderField
                label="LLM Temperature"
                value={c.coordinator?.llm_temperature ?? 0.7}
                onChange={(v) => onChange(updateSection(c, 'coordinator', 'llm_temperature', v))}
                min={0}
                max={2}
                step={0.1}
              />
              <SelectField
                label="Worker Model"
                value={c.coordinator?.worker_model ?? 'gpt-4o'}
                onChange={(v) => onChange(updateSection(c, 'coordinator', 'worker_model', v))}
                options={[
                  { value: 'gpt-4o', label: 'GPT-4o' },
                  { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
                  { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo' },
                ]}
              />
              <SliderField
                label="Worker Temperature"
                value={c.coordinator?.worker_temperature ?? 0.0}
                onChange={(v) => onChange(updateSection(c, 'coordinator', 'worker_temperature', v))}
                min={0}
                max={2}
                step={0.1}
              />
              <NumberField
                label="Max Retries"
                value={c.coordinator?.max_retries ?? 3}
                onChange={(v) => onChange(updateSection(c, 'coordinator', 'max_retries', v))}
                min={1}
                max={10}
              />
            </div>
          </ConfigSection>

          {/* 2. Quantization */}
          <ConfigSection
            title="Quantization"
            subtitle="Compression settings"
            isOpen={!!openSections.quantization}
            onToggle={() => toggleSection('quantization')}
          >
            <div className="space-y-4">
              <ButtonGroupField
                label="Bit Width"
                value={c.quantization?.default_bit_width ?? 4}
                onChange={(v) => onChange(updateSection(c, 'quantization', 'default_bit_width', v))}
                options={[
                  { value: 2, label: '2-bit' },
                  { value: 3, label: '3-bit' },
                  { value: 4, label: '4-bit' },
                  { value: 8, label: '8-bit' },
                ]}
              />
              <ButtonGroupField
                label="Method"
                value={c.quantization?.default_method ?? 'gptq'}
                onChange={(v) => onChange(updateSection(c, 'quantization', 'default_method', v))}
                options={[
                  { value: 'autoround', label: 'AutoRound' },
                  { value: 'gptq', label: 'GPTQ' },
                  { value: 'awq', label: 'AWQ' },
                  { value: 'int8', label: 'INT8' },
                ]}
              />
              <div className="grid grid-cols-2 gap-4">
                <NumberField
                  label="Group Size"
                  value={c.quantization?.group_size ?? 128}
                  onChange={(v) => onChange(updateSection(c, 'quantization', 'group_size', v))}
                  min={1}
                />
                <NumberField
                  label="Calibration Samples"
                  value={c.quantization?.calibration_samples ?? 512}
                  onChange={(v) => onChange(updateSection(c, 'quantization', 'calibration_samples', v))}
                  min={32}
                />
                <NumberField
                  label="Calibration Seq Length"
                  value={c.quantization?.calibration_seq_length ?? 2048}
                  onChange={(v) => onChange(updateSection(c, 'quantization', 'calibration_seq_length', v))}
                  min={128}
                />
              </div>
              <ToggleField
                label="Symmetric Quantization"
                value={c.quantization?.sym ?? false}
                onChange={(v) => onChange(updateSection(c, 'quantization', 'sym', v))}
              />
            </div>
          </ConfigSection>

          {/* 3. Evaluation */}
          <ConfigSection
            title="Evaluation"
            subtitle="Benchmarking settings"
            isOpen={!!openSections.evaluation}
            onToggle={() => toggleSection('evaluation')}
          >
            <div className="space-y-4">
              <ToggleField
                label="Use Proxy Evaluation"
                value={c.evaluation?.use_proxy ?? true}
                onChange={(v) => onChange(updateSection(c, 'evaluation', 'use_proxy', v))}
              />
              <div className="grid grid-cols-2 gap-4">
                <NumberField
                  label="Proxy Samples"
                  value={c.evaluation?.proxy_samples ?? 200}
                  onChange={(v) => onChange(updateSection(c, 'evaluation', 'proxy_samples', v))}
                  min={10}
                />
                <NumberField
                  label="Batch Size"
                  value={c.evaluation?.batch_size ?? 8}
                  onChange={(v) => onChange(updateSection(c, 'evaluation', 'batch_size', v))}
                  min={1}
                />
              </div>
              <ToggleField
                label="Measure Carbon"
                value={c.evaluation?.measure_carbon ?? true}
                onChange={(v) => onChange(updateSection(c, 'evaluation', 'measure_carbon', v))}
              />
              <NumberField
                label="Carbon Inference Count"
                value={c.evaluation?.carbon_inference_count ?? 500}
                onChange={(v) => onChange(updateSection(c, 'evaluation', 'carbon_inference_count', v))}
                min={10}
              />
              <SelectField
                label="Device"
                value={c.evaluation?.device ?? 'auto'}
                onChange={(v) => onChange(updateSection(c, 'evaluation', 'device', v === 'auto' ? null : v))}
                options={[
                  { value: 'auto', label: 'Auto-detect' },
                  { value: 'cuda', label: 'CUDA (GPU)' },
                  { value: 'cpu', label: 'CPU' },
                ]}
              />
            </div>
          </ConfigSection>

          {/* 4. Search */}
          <ConfigSection
            title="Search"
            subtitle="Strategy exploration"
            isOpen={!!openSections.search}
            onToggle={() => toggleSection('search')}
          >
            <div className="space-y-4">
              <ButtonGroupField
                label="Search Method"
                value={c.search?.method ?? 'bandit'}
                onChange={(v) => onChange(updateSection(c, 'search', 'method', v))}
                options={[
                  { value: 'random', label: 'Random' },
                  { value: 'bayesian', label: 'Bayesian' },
                  { value: 'evolutionary', label: 'Evolutionary' },
                  { value: 'bandit', label: 'Bandit' },
                ]}
              />
              <SliderField
                label="Exploration Ratio"
                value={c.search?.exploration_ratio ?? 0.2}
                onChange={(v) => onChange(updateSection(c, 'search', 'exploration_ratio', v))}
                min={0}
                max={1}
                step={0.05}
              />
              {(c.search?.method ?? 'bandit') === 'bandit' && (
                <NumberField
                  label="UCB Exploration Param"
                  value={c.search?.ucb_exploration_param ?? 2.0}
                  onChange={(v) => onChange(updateSection(c, 'search', 'ucb_exploration_param', v))}
                  min={0}
                  step={0.1}
                />
              )}
              {(c.search?.method) === 'evolutionary' && (
                <div className="grid grid-cols-2 gap-4">
                  <SliderField
                    label="Mutation Rate"
                    value={c.search?.mutation_rate ?? 0.1}
                    onChange={(v) => onChange(updateSection(c, 'search', 'mutation_rate', v))}
                    min={0}
                    max={1}
                    step={0.05}
                  />
                  <NumberField
                    label="Population Size"
                    value={c.search?.population_size ?? 10}
                    onChange={(v) => onChange(updateSection(c, 'search', 'population_size', v))}
                    min={2}
                  />
                </div>
              )}
            </div>
          </ConfigSection>

          {/* 5. Reward */}
          <ConfigSection
            title="Reward"
            subtitle="Multi-objective weights"
            isOpen={!!openSections.reward}
            onToggle={() => toggleSection('reward')}
          >
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <SliderField
                  label="Accuracy Weight"
                  value={c.reward?.accuracy_weight ?? 1.0}
                  onChange={(v) => onChange(updateSection(c, 'reward', 'accuracy_weight', v))}
                  min={0}
                  max={5}
                  step={0.1}
                />
                <SliderField
                  label="Latency Weight"
                  value={c.reward?.latency_weight ?? 0.3}
                  onChange={(v) => onChange(updateSection(c, 'reward', 'latency_weight', v))}
                  min={0}
                  max={5}
                  step={0.1}
                />
                <SliderField
                  label="Memory Weight"
                  value={c.reward?.memory_weight ?? 0.3}
                  onChange={(v) => onChange(updateSection(c, 'reward', 'memory_weight', v))}
                  min={0}
                  max={5}
                  step={0.1}
                />
                <SliderField
                  label="Energy Weight"
                  value={c.reward?.energy_weight ?? 0.1}
                  onChange={(v) => onChange(updateSection(c, 'reward', 'energy_weight', v))}
                  min={0}
                  max={5}
                  step={0.1}
                />
              </div>
              <SliderField
                label="Min Accuracy Threshold"
                value={c.reward?.min_accuracy ?? 0.9}
                onChange={(v) => onChange(updateSection(c, 'reward', 'min_accuracy', v))}
                min={0}
                max={1}
                step={0.01}
                format={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <input
                      type="checkbox"
                      checked={c.reward?.max_latency_ms != null}
                      onChange={(e) =>
                        onChange(updateSection(c, 'reward', 'max_latency_ms', e.target.checked ? 100 : null))
                      }
                      className="w-3.5 h-3.5 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                    />
                    <label className="text-xs font-medium text-gray-600">Max Latency (ms)</label>
                  </div>
                  {c.reward?.max_latency_ms != null && (
                    <input
                      type="number"
                      value={c.reward.max_latency_ms}
                      onChange={(e) =>
                        onChange(updateSection(c, 'reward', 'max_latency_ms', parseFloat(e.target.value)))
                      }
                      min={1}
                      className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                    />
                  )}
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <input
                      type="checkbox"
                      checked={c.reward?.max_memory_gb != null}
                      onChange={(e) =>
                        onChange(updateSection(c, 'reward', 'max_memory_gb', e.target.checked ? 8 : null))
                      }
                      className="w-3.5 h-3.5 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                    />
                    <label className="text-xs font-medium text-gray-600">Max Memory (GB)</label>
                  </div>
                  {c.reward?.max_memory_gb != null && (
                    <input
                      type="number"
                      value={c.reward.max_memory_gb}
                      onChange={(e) =>
                        onChange(updateSection(c, 'reward', 'max_memory_gb', parseFloat(e.target.value)))
                      }
                      min={0.5}
                      step={0.5}
                      className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                    />
                  )}
                </div>
              </div>
            </div>
          </ConfigSection>

          {/* 6. Fine-tuning */}
          <ConfigSection
            title="Fine-tuning"
            subtitle="LoRA / QLoRA"
            isOpen={!!openSections.finetuning}
            onToggle={() => toggleSection('finetuning')}
          >
            <div className="space-y-4">
              <SelectField
                label="LoRA Rank"
                value={String(c.finetuning?.lora_rank ?? 16)}
                onChange={(v) => onChange(updateSection(c, 'finetuning', 'lora_rank', parseInt(v)))}
                options={[
                  { value: '4', label: '4' },
                  { value: '8', label: '8' },
                  { value: '16', label: '16' },
                  { value: '32', label: '32' },
                  { value: '64', label: '64' },
                ]}
              />
              <div className="grid grid-cols-2 gap-4">
                <NumberField
                  label="LoRA Alpha"
                  value={c.finetuning?.lora_alpha ?? 32}
                  onChange={(v) => onChange(updateSection(c, 'finetuning', 'lora_alpha', v))}
                  min={1}
                />
                <SliderField
                  label="LoRA Dropout"
                  value={c.finetuning?.lora_dropout ?? 0.05}
                  onChange={(v) => onChange(updateSection(c, 'finetuning', 'lora_dropout', v))}
                  min={0}
                  max={1}
                  step={0.01}
                />
                <NumberField
                  label="Learning Rate"
                  value={c.finetuning?.learning_rate ?? 0.0002}
                  onChange={(v) => onChange(updateSection(c, 'finetuning', 'learning_rate', v))}
                  min={0}
                  step={0.0001}
                />
                <NumberField
                  label="Train Steps"
                  value={c.finetuning?.num_train_steps ?? 100}
                  onChange={(v) => onChange(updateSection(c, 'finetuning', 'num_train_steps', v))}
                  min={1}
                />
                <NumberField
                  label="Warmup Steps"
                  value={c.finetuning?.warmup_steps ?? 10}
                  onChange={(v) => onChange(updateSection(c, 'finetuning', 'warmup_steps', v))}
                  min={0}
                />
                <NumberField
                  label="Gradient Accum Steps"
                  value={c.finetuning?.gradient_accumulation_steps ?? 4}
                  onChange={(v) => onChange(updateSection(c, 'finetuning', 'gradient_accumulation_steps', v))}
                  min={1}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Target Modules</label>
                <input
                  type="text"
                  value={
                    c.finetuning?.target_modules
                      ? c.finetuning.target_modules.join(', ')
                      : ''
                  }
                  onChange={(e) => {
                    const val = e.target.value.trim();
                    onChange(
                      updateSection(
                        c,
                        'finetuning',
                        'target_modules',
                        val ? val.split(',').map((s) => s.trim()).filter(Boolean) : null,
                      ),
                    );
                  }}
                  placeholder="Auto (leave empty)"
                  className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
                />
                <p className="mt-0.5 text-xs text-gray-400">Comma-separated module names, or leave empty for auto-detection</p>
              </div>
            </div>
          </ConfigSection>

          {/* 7. Termination */}
          <ConfigSection
            title="Termination"
            subtitle="Stop conditions"
            isOpen={!!openSections.termination}
            onToggle={() => toggleSection('termination')}
          >
            <div className="grid grid-cols-2 gap-4">
              <NumberField
                label="Max Episodes"
                value={c.termination?.max_episodes ?? 10}
                onChange={(v) => onChange(updateSection(c, 'termination', 'max_episodes', v))}
                min={1}
              />
              <NumberField
                label="Budget (hours)"
                value={c.termination?.budget_hours ?? 2.0}
                onChange={(v) => onChange(updateSection(c, 'termination', 'budget_hours', v))}
                min={0.1}
                step={0.5}
              />
              <NumberField
                label="Convergence Patience"
                value={c.termination?.convergence_patience ?? 5}
                onChange={(v) => onChange(updateSection(c, 'termination', 'convergence_patience', v))}
                min={1}
              />
              <NumberField
                label="Min Improvement"
                value={c.termination?.min_improvement ?? 0.001}
                onChange={(v) => onChange(updateSection(c, 'termination', 'min_improvement', v))}
                min={0}
                step={0.001}
              />
            </div>
          </ConfigSection>
        </div>
      )}
    </div>
  );
}
