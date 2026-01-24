import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Loader2,
  Filter,
  PlusCircle,
  Search,
  CheckCircle2,
  Clock,
  XCircle,
  Zap,
  HardDrive,
  Leaf,
} from 'lucide-react';
import { useJobs, useExperiments } from '../hooks/useJobs';
import { JobCard } from '../components/JobCard';
import { formatRelativeTime } from '../hooks/usePolling';
import type { JobStatus, Experiment } from '../types';

const statusFilters: { value: JobStatus | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'running', label: 'Running' },
  { value: 'pending', label: 'Pending' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
];

// Card component for filesystem experiments
function ExperimentCard({ experiment }: { experiment: Experiment }) {
  const statusIcons = {
    completed: <CheckCircle2 className="w-5 h-5 text-green-500" />,
    running: <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />,
    failed: <XCircle className="w-5 h-5 text-red-500" />,
  };

  return (
    <Link
      to={`/experiment/${experiment.experiment_id}`}
      className="block bg-white rounded-lg border border-gray-200 p-4 hover:border-primary-300 hover:shadow-md transition-all"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {statusIcons[experiment.status]}
            <h3 className="font-medium text-gray-900 truncate">
              {experiment.model_name}
            </h3>
            <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
              {experiment.dataset}
            </span>
          </div>

          <p className="text-sm text-gray-500 mt-1">
            {formatRelativeTime(experiment.created_at)}
          </p>
        </div>

        <div className="flex items-center gap-4 text-sm">
          {/* Episodes */}
          <div className="text-center">
            <div className="font-semibold text-gray-900">
              {experiment.episodes_completed}
            </div>
            <div className="text-xs text-gray-500">episodes</div>
          </div>

          {/* Pareto Solutions */}
          <div className="text-center">
            <div className="font-semibold text-primary-600">
              {experiment.pareto_solutions}
            </div>
            <div className="text-xs text-gray-500">pareto</div>
          </div>
        </div>
      </div>

      {/* Metrics row */}
      {experiment.status === 'completed' && (
        <div className="mt-3 pt-3 border-t border-gray-100 flex items-center gap-6 text-sm">
          {experiment.best_accuracy !== null && (
            <div className="flex items-center gap-1 text-gray-600">
              <Zap className="w-4 h-4 text-yellow-500" />
              <span className="font-medium">
                {(experiment.best_accuracy * 100).toFixed(1)}%
              </span>
              <span className="text-gray-400">accuracy</span>
            </div>
          )}

          {experiment.best_compression !== null && (
            <div className="flex items-center gap-1 text-gray-600">
              <HardDrive className="w-4 h-4 text-blue-500" />
              <span className="font-medium">
                {experiment.best_compression.toFixed(1)}x
              </span>
              <span className="text-gray-400">compression</span>
            </div>
          )}

          {experiment.best_co2_grams !== null && (
            <div className="flex items-center gap-1 text-gray-600">
              <Leaf className="w-4 h-4 text-green-500" />
              <span className="font-medium">
                {experiment.best_co2_grams.toFixed(2)}g
              </span>
              <span className="text-gray-400">CO2</span>
            </div>
          )}
        </div>
      )}
    </Link>
  );
}

export function Experiments() {
  const [statusFilter, setStatusFilter] = useState<JobStatus | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'all' | 'api' | 'filesystem'>('all');

  // Fetch both API jobs and filesystem experiments
  const {
    data: jobs,
    isLoading: jobsLoading,
    error: jobsError,
  } = useJobs(statusFilter === 'all' ? undefined : statusFilter);

  const {
    data: experiments,
    isLoading: experimentsLoading,
    error: experimentsError,
  } = useExperiments();

  const isLoading = jobsLoading || experimentsLoading;
  const error = jobsError || experimentsError;

  // Ensure arrays
  const jobsList = Array.isArray(jobs) ? jobs : [];
  const experimentsList = Array.isArray(experiments) ? experiments : [];

  // Filter by search query
  const filteredJobs = jobsList.filter((job) =>
    searchQuery ? job.job_id.toLowerCase().includes(searchQuery.toLowerCase()) : true
  );

  const filteredExperiments = experimentsList.filter((exp) =>
    searchQuery
      ? exp.experiment_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        exp.model_name.toLowerCase().includes(searchQuery.toLowerCase())
      : true
  );

  // Apply status filter to experiments
  const statusFilteredExperiments =
    statusFilter === 'all'
      ? filteredExperiments
      : filteredExperiments.filter((exp) => exp.status === statusFilter);

  // Combine counts
  const totalApiJobs = jobsList.length;
  const totalFilesystemExperiments = experimentsList.length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Experiments</h1>
          <p className="text-gray-500 mt-1">
            View and manage your compression experiments
          </p>
        </div>
        <Link
          to="/new"
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          <PlusCircle className="w-5 h-5" />
          New Experiment
        </Link>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by ID or model name..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
          />
        </div>

        {/* View mode toggle */}
        <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setViewMode('all')}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              viewMode === 'all'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            All ({totalApiJobs + totalFilesystemExperiments})
          </button>
          <button
            onClick={() => setViewMode('filesystem')}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              viewMode === 'filesystem'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            CLI ({totalFilesystemExperiments})
          </button>
          <button
            onClick={() => setViewMode('api')}
            className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
              viewMode === 'api'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            API ({totalApiJobs})
          </button>
        </div>

        {/* Status filter */}
        <div className="flex items-center gap-2">
          <Filter className="w-5 h-5 text-gray-400" />
          <div className="flex gap-1">
            {statusFilters.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => setStatusFilter(value)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  statusFilter === value
                    ? 'bg-primary-100 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Experiment list */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-600">
          Failed to load experiments: {error.message}
        </div>
      ) : filteredJobs.length === 0 && statusFilteredExperiments.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
          <p className="text-gray-500 mb-4">
            {searchQuery
              ? 'No experiments match your search'
              : statusFilter !== 'all'
                ? `No ${statusFilter} experiments`
                : 'No experiments yet'}
          </p>
          {!searchQuery && statusFilter === 'all' && (
            <Link
              to="/new"
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              <PlusCircle className="w-5 h-5" />
              Start your first experiment
            </Link>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {/* Filesystem experiments (CLI) */}
          {(viewMode === 'all' || viewMode === 'filesystem') &&
            statusFilteredExperiments.map((exp) => (
              <ExperimentCard key={exp.experiment_id} experiment={exp} />
            ))}

          {/* API jobs */}
          {(viewMode === 'all' || viewMode === 'api') &&
            filteredJobs.map((job) => <JobCard key={job.job_id} job={job} />)}
        </div>
      )}

      {/* Stats summary */}
      {(jobsList.length > 0 || experimentsList.length > 0) && (
        <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 flex items-center justify-between text-sm text-gray-600">
          <span>
            Showing {filteredJobs.length + statusFilteredExperiments.length} experiments
          </span>
          <div className="flex gap-4">
            <span>
              <span className="font-medium text-green-600">
                {jobsList.filter((j) => j.status === 'completed').length +
                  experimentsList.filter((e) => e.status === 'completed').length}
              </span>{' '}
              completed
            </span>
            <span>
              <span className="font-medium text-blue-600">
                {jobsList.filter((j) => j.status === 'running').length}
              </span>{' '}
              running
            </span>
            <span>
              <span className="font-medium text-red-600">
                {jobsList.filter((j) => j.status === 'failed').length}
              </span>{' '}
              failed
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
