import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Loader2,
  CheckCircle2,
  Download,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';
import { useExperiment, useExperimentEpisodes } from '../hooks/useJobs';
import { MetricsCard, CompressionRatioMetric } from '../components/MetricsCard';
import {
  AccuracyVsLatencyChart,
  AccuracyVsMemoryChart,
  AccuracyVsCO2Chart,
  AccuracyVsSizeChart,
} from '../components/ParetoChart';
import { EpisodeTimeline } from '../components/EpisodeTimeline';
import { solutionsToPoints } from '../utils/pareto';
import type { ParetoSolution } from '../types';

export function FilesystemExperimentDetail() {
  const { experimentId } = useParams<{ experimentId: string }>();
  const {
    data: experiment,
    isLoading: experimentLoading,
    refetch,
  } = useExperiment(experimentId);
  const { data: episodesData, isLoading: episodesLoading } =
    useExperimentEpisodes(experimentId);

  if (experimentLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="text-center py-24">
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Experiment not found
        </h2>
        <Link
          to="/experiments"
          className="text-primary-600 hover:text-primary-700"
        >
          Back to experiments
        </Link>
      </div>
    );
  }

  const results = experiment.results;
  const paretoFrontier = experiment.pareto_frontier;
  const solutions = paretoFrontier?.frontier || [];
  const paretoPoints = solutions.length > 0
    ? solutionsToPoints(solutions)
    : [];

  // Find best solution (highest accuracy)
  const bestSolution = solutions.reduce(
    (best: ParetoSolution | null, curr: ParetoSolution) =>
      !best || curr.result.accuracy > best.result.accuracy ? curr : best,
    null as ParetoSolution | null
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <Link
            to="/experiments"
            className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-700 text-sm mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to experiments
          </Link>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
            {results.model}
            <CheckCircle2 className="w-5 h-5 text-green-500" />
          </h1>
          <p className="text-gray-500 mt-1">
            Dataset: {results.dataset} | {results.episodes_completed} episodes
          </p>
        </div>

        <div className="flex items-center gap-2">
          {results.visualization && (
            <a
              href={`/${results.visualization}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-3 py-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
              title="Open Interactive Visualization"
            >
              <ExternalLink className="w-5 h-5" />
              Visualization
            </a>
          )}
          <button
            onClick={() => refetch()}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Summary metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {bestSolution && (
          <CompressionRatioMetric value={bestSolution.result.compression_ratio} />
        )}
        <MetricsCard
          title="Pareto Solutions"
          value={results.frontier_summary?.num_solutions || 0}
          description="optimal trade-offs"
        />
        <MetricsCard
          title="Episodes"
          value={results.episodes_completed}
        />
        {bestSolution && (
          <MetricsCard
            title="Best Accuracy"
            value={(bestSolution.result.accuracy * 100).toFixed(1)}
            unit="%"
            variant="success"
          />
        )}
      </div>

      {/* Best solution details */}
      {bestSolution && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Best Solution
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <span className="text-sm text-gray-500">Method</span>
              <p className="font-medium text-gray-900">
                {bestSolution.strategy.methods[0]?.toUpperCase() || 'N/A'}
              </p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Bits</span>
              <p className="font-medium text-gray-900">
                {bestSolution.strategy.quantization_bits || 'N/A'}
              </p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Accuracy</span>
              <p className="font-medium text-gray-900">
                {(bestSolution.result.accuracy * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Latency</span>
              <p className="font-medium text-gray-900">
                {bestSolution.result.latency_ms.toFixed(1)} ms
              </p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Memory</span>
              <p className="font-medium text-gray-900">
                {bestSolution.result.memory_gb.toFixed(2)} GB
              </p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Model Size</span>
              <p className="font-medium text-gray-900">
                {bestSolution.result.model_size_gb.toFixed(2)} GB
              </p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Compression</span>
              <p className="font-medium text-gray-900">
                {bestSolution.result.compression_ratio.toFixed(2)}x
              </p>
            </div>
            {bestSolution.result.co2_grams !== null && (
              <div>
                <span className="text-sm text-gray-500">CO2</span>
                <p className="font-medium text-gray-900">
                  {bestSolution.result.co2_grams.toFixed(2)} g
                </p>
              </div>
            )}
          </div>

          {/* Download button */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <button
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              onClick={() => {
                alert(`Checkpoint path: ${bestSolution.result.checkpoint_path}`);
              }}
            >
              <Download className="w-5 h-5" />
              View Checkpoint Path
            </button>
          </div>
        </div>
      )}

      {/* Episode History Timeline */}
      <EpisodeTimeline
        episodes={episodesData?.episodes || []}
        isLoading={episodesLoading}
      />

      {/* Pareto Charts */}
      {paretoPoints.length > 0 ? (
        <div className="space-y-6">
          <h2 className="text-lg font-semibold text-gray-900">
            Pareto Frontier Visualization
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <AccuracyVsLatencyChart data={paretoPoints} />
            <AccuracyVsMemoryChart data={paretoPoints} />
            <AccuracyVsSizeChart data={paretoPoints} />
            <AccuracyVsCO2Chart data={paretoPoints} />
          </div>
        </div>
      ) : (
        <div className="bg-gray-50 rounded-lg border border-gray-200 p-8 text-center text-gray-500">
          No Pareto data available for visualization
        </div>
      )}

      {/* All solutions table */}
      {solutions.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="p-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">
              All Pareto Solutions ({solutions.length})
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">
                    Method
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">
                    Bits
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">
                    Accuracy
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">
                    Latency
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">
                    Memory
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">
                    Size
                  </th>
                  <th className="px-4 py-3 text-left font-medium text-gray-600">
                    CO2
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {solutions.map((sol: ParetoSolution) => (
                  <tr
                    key={sol.strategy.strategy_id}
                    className="hover:bg-gray-50"
                  >
                    <td className="px-4 py-3 font-medium">
                      {sol.strategy.methods[0]?.toUpperCase() || 'N/A'}
                    </td>
                    <td className="px-4 py-3">
                      {sol.strategy.quantization_bits || '-'}
                    </td>
                    <td className="px-4 py-3">
                      {(sol.result.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="px-4 py-3">
                      {sol.result.latency_ms.toFixed(1)} ms
                    </td>
                    <td className="px-4 py-3">
                      {sol.result.memory_gb.toFixed(2)} GB
                    </td>
                    <td className="px-4 py-3">
                      {sol.result.model_size_gb.toFixed(2)} GB
                    </td>
                    <td className="px-4 py-3">
                      {sol.result.co2_grams?.toFixed(2) || '-'} g
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Model Spec Info */}
      {experiment.model_spec && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Model Specification
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Model Family</span>
              <p className="font-medium">{experiment.model_spec.model_family}</p>
            </div>
            <div>
              <span className="text-gray-500">Model Size</span>
              <p className="font-medium">
                {experiment.model_spec.model_size_gb?.toFixed(2)} GB
              </p>
            </div>
            <div>
              <span className="text-gray-500">Min VRAM</span>
              <p className="font-medium">
                {experiment.model_spec.min_vram_gb?.toFixed(1)} GB
              </p>
            </div>
            <div>
              <span className="text-gray-500">Preferred Methods</span>
              <p className="font-medium">
                {experiment.model_spec.preferred_methods?.join(', ') || 'N/A'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
