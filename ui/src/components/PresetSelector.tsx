import { Target, Zap, Scale, HardDrive, Leaf, DollarSign } from 'lucide-react';
import type { PresetName } from '../types';

interface PresetSelectorProps {
  selected: PresetName | null;
  onSelect: (preset: PresetName | null) => void;
  modified?: boolean;
}

const PRESETS: {
  name: PresetName;
  label: string;
  description: string;
  icon: typeof Target;
}[] = [
  {
    name: 'accuracy_focused',
    label: 'Accuracy',
    description: '8-bit, 95% min accuracy',
    icon: Target,
  },
  {
    name: 'latency_focused',
    label: 'Latency',
    description: '4-bit GPTQ, fast inference',
    icon: Zap,
  },
  {
    name: 'balanced',
    label: 'Balanced',
    description: '4-bit, 90% min accuracy',
    icon: Scale,
  },
  {
    name: 'memory_constrained',
    label: 'Memory',
    description: '4-bit AWQ, LoRA rank 8',
    icon: HardDrive,
  },
  {
    name: 'low_carbon',
    label: 'Low Carbon',
    description: 'Minimize CO2 emissions',
    icon: Leaf,
  },
  {
    name: 'low_cost',
    label: 'Low Cost',
    description: 'Minimize hardware cost',
    icon: DollarSign,
  },
];

export function PresetSelector({
  selected,
  onSelect,
  modified = false,
}: PresetSelectorProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Preset
        {selected && modified && (
          <span className="ml-2 text-xs text-amber-600">(modified)</span>
        )}
      </label>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {PRESETS.map(({ name, label, description, icon: Icon }) => {
          const isSelected = selected === name;
          return (
            <button
              key={name}
              type="button"
              onClick={() => onSelect(isSelected ? null : name)}
              className={`flex flex-col items-center gap-1.5 p-3 rounded-lg border-2 transition-colors text-center ${
                isSelected
                  ? 'border-primary-500 bg-primary-50 text-primary-700'
                  : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300 hover:bg-gray-50'
              }`}
            >
              <Icon className={`w-5 h-5 ${isSelected ? 'text-primary-600' : 'text-gray-400'}`} />
              <span className="text-sm font-medium">{label}</span>
              <span className="text-xs text-gray-500 leading-tight">{description}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
