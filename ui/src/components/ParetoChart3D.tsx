import { useState, useMemo } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
// eslint-disable-next-line @typescript-eslint/no-require-imports
import * as PlotlyModule from 'plotly.js-dist-min';
import type { ParetoPoint } from '../types';

// plotly.js-dist-min uses default export at runtime
const Plotly = (PlotlyModule as unknown as { default: typeof PlotlyModule }).default || PlotlyModule;
const Plot = createPlotlyComponent(Plotly as Parameters<typeof createPlotlyComponent>[0]);

type MetricKey = 'accuracy' | 'latency' | 'memory' | 'size' | 'co2';

interface ParetoChart3DProps {
  data: ParetoPoint[];
  defaultX?: MetricKey;
  defaultY?: MetricKey;
  defaultZ?: MetricKey;
  height?: number;
}

const AXIS_META: Record<MetricKey, { label: string; axisTitle: string; format: (v: number) => string }> = {
  accuracy: { label: 'Accuracy', axisTitle: 'Accuracy (%)', format: (v) => `${(v * 100).toFixed(1)}%` },
  latency: { label: 'Latency', axisTitle: 'Latency (ms)', format: (v) => `${v.toFixed(1)} ms` },
  memory: { label: 'Memory', axisTitle: 'Memory (GB)', format: (v) => `${v.toFixed(2)} GB` },
  size: { label: 'Size', axisTitle: 'Model Size (GB)', format: (v) => `${v.toFixed(2)} GB` },
  co2: { label: 'CO2', axisTitle: 'CO2 Emissions (g)', format: (v) => `${v.toFixed(2)} g` },
};

const METHOD_COLORS: Record<string, string> = {
  autoround: '#3b82f6',
  gptq: '#10b981',
  int8: '#f59e0b',
  awq: '#8b5cf6',
  pruning: '#ef4444',
  lora: '#ec4899',
  qlora: '#06b6d4',
  default: '#6b7280',
};

const METRIC_KEYS: MetricKey[] = ['accuracy', 'latency', 'memory', 'size', 'co2'];

export function ParetoChart3D({
  data,
  defaultX = 'accuracy',
  defaultY = 'latency',
  defaultZ = 'memory',
  height = 550,
}: ParetoChart3DProps) {
  const [xAxis, setXAxis] = useState<MetricKey>(defaultX);
  const [yAxis, setYAxis] = useState<MetricKey>(defaultY);
  const [zAxis, setZAxis] = useState<MetricKey>(defaultZ);

  const traces = useMemo(() => {
    // Group data by method
    const byMethod: Record<string, ParetoPoint[]> = {};
    for (const point of data) {
      const method = point.method || 'default';
      if (!byMethod[method]) byMethod[method] = [];
      byMethod[method].push(point);
    }

    return Object.entries(byMethod).map(([method, points]) => ({
      type: 'scatter3d' as const,
      mode: 'markers' as const,
      name: method.toUpperCase(),
      x: points.map((p) => p[xAxis]),
      y: points.map((p) => p[yAxis]),
      z: points.map((p) => p[zAxis]),
      text: points.map((p) =>
        `<b>${method.toUpperCase()}</b>${p.bits ? ` (${p.bits}-bit)` : ''}<br>` +
        `Accuracy: ${AXIS_META.accuracy.format(p.accuracy)}<br>` +
        `Latency: ${AXIS_META.latency.format(p.latency)}<br>` +
        `Memory: ${AXIS_META.memory.format(p.memory)}<br>` +
        `Size: ${AXIS_META.size.format(p.size)}<br>` +
        `CO2: ${AXIS_META.co2.format(p.co2)}`
      ),
      hoverinfo: 'text' as const,
      marker: {
        size: 7,
        color: METHOD_COLORS[method] || METHOD_COLORS.default,
        opacity: 0.9,
        line: { width: 1, color: '#ffffff' },
      },
    }));
  }, [data, xAxis, yAxis, zAxis]);

  const layout = useMemo(
    () => ({
      autosize: true,
      height,
      margin: { l: 0, r: 0, t: 30, b: 0 },
      scene: {
        xaxis: { title: { text: AXIS_META[xAxis].axisTitle } },
        yaxis: { title: { text: AXIS_META[yAxis].axisTitle } },
        zaxis: { title: { text: AXIS_META[zAxis].axisTitle } },
        camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } },
      },
      legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)' },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
    }),
    [xAxis, yAxis, zAxis, height]
  );

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        3D Pareto Frontier
      </h3>

      {/* Axis selectors */}
      <div className="flex flex-wrap gap-4 mb-4">
        {(['X', 'Y', 'Z'] as const).map((label) => {
          const value = label === 'X' ? xAxis : label === 'Y' ? yAxis : zAxis;
          const setter = label === 'X' ? setXAxis : label === 'Y' ? setYAxis : setZAxis;
          return (
            <div key={label} className="flex items-center gap-2">
              <label className="text-sm font-medium text-gray-600">{label}-Axis:</label>
              <select
                value={value}
                onChange={(e) => setter(e.target.value as MetricKey)}
                className="px-2 py-1 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none"
              >
                {METRIC_KEYS.map((k) => (
                  <option key={k} value={k}>
                    {AXIS_META[k].label}
                  </option>
                ))}
              </select>
            </div>
          );
        })}
      </div>

      {/* 3D Plot */}
      <Plot
        data={traces}
        layout={layout}
        useResizeHandler={true}
        style={{ width: '100%' }}
        config={{ responsive: true, displayModeBar: true }}
      />
    </div>
  );
}
