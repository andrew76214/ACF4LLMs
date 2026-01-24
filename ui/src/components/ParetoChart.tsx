import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import type { ParetoPoint } from '../types';

interface ParetoChartProps {
  data: ParetoPoint[];
  xKey: keyof ParetoPoint;
  yKey: keyof ParetoPoint;
  xLabel: string;
  yLabel: string;
  title?: string;
  showBaseline?: boolean;
  baselineX?: number;
  baselineY?: number;
}

// Color map for different compression methods
const methodColors: Record<string, string> = {
  autoround: '#3b82f6', // blue
  gptq: '#10b981', // green
  int8: '#f59e0b', // amber
  awq: '#8b5cf6', // purple
  pruning: '#ef4444', // red
  lora: '#ec4899', // pink
  qlora: '#06b6d4', // cyan
  default: '#6b7280', // gray
};

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: ParetoPoint;
  }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) return null;

  const point = payload[0].payload;

  return (
    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg text-sm">
      <p className="font-medium text-gray-900 mb-2">
        Strategy: {point.strategyId.slice(0, 8)}...
      </p>
      <div className="space-y-1 text-gray-600">
        <p>Method: <span className="font-medium">{point.method}</span></p>
        {point.bits && <p>Bits: <span className="font-medium">{point.bits}</span></p>}
        <p>Accuracy: <span className="font-medium">{(point.accuracy * 100).toFixed(1)}%</span></p>
        <p>Latency: <span className="font-medium">{point.latency.toFixed(1)} ms</span></p>
        <p>Memory: <span className="font-medium">{point.memory.toFixed(2)} GB</span></p>
        <p>Size: <span className="font-medium">{point.size.toFixed(2)} GB</span></p>
        {point.co2 > 0 && (
          <p>CO2: <span className="font-medium">{point.co2.toFixed(2)} g</span></p>
        )}
      </div>
    </div>
  );
}

export function ParetoChart({
  data,
  xKey,
  yKey,
  xLabel,
  yLabel,
  title,
  showBaseline = false,
  baselineX,
  baselineY,
}: ParetoChartProps) {
  // Group data by method
  const dataByMethod = data.reduce(
    (acc, point) => {
      const method = point.method || 'default';
      if (!acc[method]) acc[method] = [];
      acc[method].push(point);
      return acc;
    },
    {} as Record<string, ParetoPoint[]>
  );

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      )}
      <ResponsiveContainer width="100%" height={350}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            type="number"
            dataKey={xKey}
            name={xLabel}
            tick={{ fontSize: 12 }}
            label={{
              value: xLabel,
              position: 'insideBottom',
              offset: -10,
              fontSize: 14,
            }}
          />
          <YAxis
            type="number"
            dataKey={yKey}
            name={yLabel}
            tick={{ fontSize: 12 }}
            label={{
              value: yLabel,
              angle: -90,
              position: 'insideLeft',
              fontSize: 14,
            }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* Reference lines for baseline */}
          {showBaseline && baselineX !== undefined && (
            <ReferenceLine
              x={baselineX}
              stroke="#9ca3af"
              strokeDasharray="5 5"
              label={{ value: 'Baseline', fontSize: 10 }}
            />
          )}
          {showBaseline && baselineY !== undefined && (
            <ReferenceLine
              y={baselineY}
              stroke="#9ca3af"
              strokeDasharray="5 5"
            />
          )}

          {/* Scatter plots for each method */}
          {Object.entries(dataByMethod).map(([method, points]) => (
            <Scatter
              key={method}
              name={method.toUpperCase()}
              data={points}
              fill={methodColors[method] || methodColors.default}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

// Preset chart configurations
export function AccuracyVsLatencyChart({ data }: { data: ParetoPoint[] }) {
  return (
    <ParetoChart
      data={data}
      xKey="latency"
      yKey="accuracy"
      xLabel="Latency (ms)"
      yLabel="Accuracy"
      title="Accuracy vs Latency"
    />
  );
}

export function AccuracyVsMemoryChart({ data }: { data: ParetoPoint[] }) {
  return (
    <ParetoChart
      data={data}
      xKey="memory"
      yKey="accuracy"
      xLabel="Memory (GB)"
      yLabel="Accuracy"
      title="Accuracy vs Memory"
    />
  );
}

export function AccuracyVsCO2Chart({ data }: { data: ParetoPoint[] }) {
  return (
    <ParetoChart
      data={data}
      xKey="co2"
      yKey="accuracy"
      xLabel="CO2 Emissions (g)"
      yLabel="Accuracy"
      title="Accuracy vs CO2 Emissions"
    />
  );
}

export function AccuracyVsSizeChart({ data }: { data: ParetoPoint[] }) {
  return (
    <ParetoChart
      data={data}
      xKey="size"
      yKey="accuracy"
      xLabel="Model Size (GB)"
      yLabel="Accuracy"
      title="Accuracy vs Model Size"
    />
  );
}
