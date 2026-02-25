import { useParams, Link } from 'react-router-dom';
import { useEffect, useRef } from 'react';
import {
  ArrowLeft,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  RefreshCw,
  Terminal,
} from 'lucide-react';
import { useJob, usePareto, useLogs, useEpisodes } from '../hooks/useJobs';
import { formatRelativeTime } from '../hooks/usePolling';
import { MetricsCard, CompressionRatioMetric } from '../components/MetricsCard';
import {
  AccuracyVsLatencyChart,
  AccuracyVsMemoryChart,
  AccuracyVsCO2Chart,
  AccuracyVsSizeChart,
} from '../components/ParetoChart';
import { ParetoChart3D } from '../components/ParetoChart3D';
import { EpisodeTimeline } from '../components/EpisodeTimeline';
import { solutionsToPoints } from '../utils/pareto';
import type { ParetoSolution } from '../types';

export function ExperimentDetail() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data: job, isLoading: jobLoading, refetch } = useJob(jobId);
  const { data: pareto, isLoading: paretoLoading } = usePareto(
    job?.status === 'completed' ? jobId : undefined
  );
  const isRunningOrPending = job?.status === 'running' || job?.status === 'pending';
  const { data: logsData } = useLogs(jobId, isRunningOrPending);
  const { data: episodesData, isLoading: episodesLoading } = useEpisodes(
    jobId,
    isRunningOrPending
  );

  // Auto-scroll logs to bottom
  const logsEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (logsEndRef.current && isRunningOrPending) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logsData?.logs, isRunningOrPending]);

  if (jobLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  if (!job) {
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

  const statusIcons = {
    pending: <Clock className="w-5 h-5 text-yellow-500" />,
    running: <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />,
    completed: <CheckCircle2 className="w-5 h-5 text-green-500" />,
    failed: <XCircle className="w-5 h-5 text-red-500" />,
  };

  const paretoPoints = Array.isArray(pareto?.solutions) ? solutionsToPoints(pareto.solutions) : [];

  // Find best solution (highest accuracy)
  const bestSolution = pareto?.solutions?.reduce(
    (best, curr) =>
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
            Experiment
            <span className="font-mono text-lg text-gray-500">
              {job.job_id.slice(0, 8)}...
            </span>
            {statusIcons[job.status]}
          </h1>
          <p className="text-gray-500 mt-1">
            Created {formatRelativeTime(job.created_at)}
          </p>
        </div>

        <button
          onClick={() => refetch()}
          className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Progress */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Progress</span>
          <span className="text-sm text-gray-500">
            Episode {job.progress.current_episode} / {job.progress.max_episodes}
          </span>
        </div>
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              job.status === 'failed'
                ? 'bg-red-500'
                : job.status === 'completed'
                  ? 'bg-green-500'
                  : 'bg-primary-500'
            }`}
            style={{
              width: `${(job.progress.current_episode / job.progress.max_episodes) * 100}%`,
            }}
          />
        </div>
        <div className="mt-2 flex items-center gap-4 text-sm text-gray-600">
          <span>Pareto Solutions: {job.progress.pareto_solutions}</span>
        </div>
      </div>

      {/* Logs Panel */}
      {logsData && Array.isArray(logsData.logs) && logsData.logs.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
            <div className="flex items-center gap-2 text-gray-300">
              <Terminal className="w-4 h-4" />
              <span className="text-sm font-medium">Logs</span>
            </div>
            <span className="text-xs text-gray-500">
              {logsData.total} {logsData.total === 1 ? 'line' : 'lines'}
            </span>
          </div>
          <div className="h-64 overflow-y-auto p-4 font-mono text-sm">
            {logsData.logs.map((log, index) => (
              <div key={index} className="text-green-400 whitespace-pre-wrap break-all">
                {log}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}

      {/* Episode History Timeline */}
      {(isRunningOrPending || job.status === 'completed') && (
        <EpisodeTimeline
          episodes={episodesData?.episodes || []}
          isLoading={episodesLoading}
        />
      )}

      {/* Error message */}
      {job.error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-600">
          <p className="font-medium mb-1">Error</p>
          <p className="text-sm">{job.error}</p>
        </div>
      )}

      {/* Results section - only show for completed jobs */}
      {job.status === 'completed' && job.result && (
        <>
          {/* Summary metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <CompressionRatioMetric value={job.result.compression_achieved} />
            <MetricsCard
              title="Pareto Solutions"
              value={job.result.pareto_frontier_size}
              description="optimal trade-offs"
            />
            <MetricsCard
              title="Strategies Tried"
              value={job.result.total_strategies_tried}
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

            </div>
          )}

          {/* Pareto Charts */}
          {paretoLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
            </div>
          ) : paretoPoints.length > 0 ? (
            <div className="space-y-6">
              <h2 className="text-lg font-semibold text-gray-900">
                Pareto Frontier Visualization
              </h2>
              <ParetoChart3D data={paretoPoints} />
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
          {Array.isArray(pareto?.solutions) && pareto.solutions.length > 0 && (
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">
                  All Pareto Solutions ({pareto.solutions.length})
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
                    {pareto.solutions.map((sol) => (
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
        </>
      )}

      {/* Running state info */}
      {job.status === 'running' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 text-center">
          <Loader2 className="w-12 h-12 animate-spin text-blue-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Experiment in Progress
          </h3>
          <p className="text-gray-600">
            This page will automatically update as new results come in.
          </p>
        </div>
      )}

      {/* Pending state info */}
      {job.status === 'pending' && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <Clock className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Waiting to Start
          </h3>
          <p className="text-gray-600">
            This experiment is queued and will start soon.
          </p>
        </div>
      )}
    </div>
  );
}
