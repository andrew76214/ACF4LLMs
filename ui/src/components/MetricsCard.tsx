import type { ReactNode } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricsCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon?: ReactNode;
  change?: number;
  changeLabel?: string;
  description?: string;
  variant?: 'default' | 'success' | 'warning' | 'error';
}

const variantStyles = {
  default: 'bg-white border-gray-200',
  success: 'bg-green-50 border-green-200',
  warning: 'bg-yellow-50 border-yellow-200',
  error: 'bg-red-50 border-red-200',
};

export function MetricsCard({
  title,
  value,
  unit,
  icon,
  change,
  changeLabel,
  description,
  variant = 'default',
}: MetricsCardProps) {
  const getTrendIcon = () => {
    if (change === undefined || change === 0) {
      return <Minus className="w-4 h-4 text-gray-400" />;
    }
    if (change > 0) {
      return <TrendingUp className="w-4 h-4 text-green-500" />;
    }
    return <TrendingDown className="w-4 h-4 text-red-500" />;
  };

  const getTrendColor = () => {
    if (change === undefined || change === 0) return 'text-gray-500';
    if (change > 0) return 'text-green-600';
    return 'text-red-600';
  };

  return (
    <div
      className={`rounded-lg border p-4 ${variantStyles[variant]}`}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-600">{title}</span>
        {icon && <div className="text-gray-400">{icon}</div>}
      </div>

      <div className="mt-2 flex items-baseline gap-1">
        <span className="text-2xl font-bold text-gray-900">
          {typeof value === 'number' ? value.toLocaleString() : value}
        </span>
        {unit && <span className="text-sm text-gray-500">{unit}</span>}
      </div>

      {(change !== undefined || description) && (
        <div className="mt-2 flex items-center gap-2">
          {change !== undefined && (
            <div className={`flex items-center gap-1 text-sm ${getTrendColor()}`}>
              {getTrendIcon()}
              <span>
                {change > 0 ? '+' : ''}
                {change.toFixed(1)}%
              </span>
              {changeLabel && (
                <span className="text-gray-500">{changeLabel}</span>
              )}
            </div>
          )}
          {description && (
            <span className="text-sm text-gray-500">{description}</span>
          )}
        </div>
      )}
    </div>
  );
}

export function CompressionRatioMetric({ value }: { value: number }) {
  return (
    <MetricsCard
      title="Compression"
      value={`${value.toFixed(2)}x`}
      description="size reduction"
      variant={value >= 2 ? 'success' : value >= 1.5 ? 'warning' : 'default'}
    />
  );
}
