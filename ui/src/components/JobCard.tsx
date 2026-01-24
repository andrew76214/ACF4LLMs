import { Link } from 'react-router-dom';
import {
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronRight,
} from 'lucide-react';
import type { Job } from '../types';
import { formatRelativeTime } from '../hooks/usePolling';

interface JobCardProps {
  job: Job;
}

const statusConfig = {
  pending: {
    icon: Clock,
    color: 'text-yellow-600',
    bg: 'bg-yellow-50',
    label: 'Pending',
    animate: false,
  },
  running: {
    icon: Loader2,
    color: 'text-blue-600',
    bg: 'bg-blue-50',
    label: 'Running',
    animate: true,
  },
  completed: {
    icon: CheckCircle2,
    color: 'text-green-600',
    bg: 'bg-green-50',
    label: 'Completed',
    animate: false,
  },
  failed: {
    icon: XCircle,
    color: 'text-red-600',
    bg: 'bg-red-50',
    label: 'Failed',
    animate: false,
  },
};

export function JobCard({ job }: JobCardProps) {
  const config = statusConfig[job.status];
  const StatusIcon = config.icon;

  return (
    <Link
      to={`/experiments/${job.job_id}`}
      className="block bg-white rounded-lg border border-gray-200 p-4 hover:border-primary-300 hover:shadow-sm transition-all"
    >
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          {/* Job ID */}
          <p className="text-sm font-mono text-gray-500 truncate">
            {job.job_id.slice(0, 8)}...
          </p>

          {/* Progress */}
          <div className="mt-2">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">
                Episode {job.progress.current_episode}/{job.progress.max_episodes}
              </span>
              {job.progress.pareto_solutions > 0 && (
                <span className="text-xs bg-primary-100 text-primary-700 px-2 py-0.5 rounded-full">
                  {job.progress.pareto_solutions} Pareto
                </span>
              )}
            </div>

            {/* Progress bar */}
            <div className="mt-2 h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 rounded-full transition-all duration-300"
                style={{
                  width: `${(job.progress.current_episode / job.progress.max_episodes) * 100}%`,
                }}
              />
            </div>
          </div>

          {/* Result summary */}
          {job.result && (
            <div className="mt-3 text-sm text-gray-600">
              <span className="font-medium text-gray-900">
                {job.result.compression_achieved.toFixed(2)}x
              </span>{' '}
              compression achieved
            </div>
          )}

          {/* Error message */}
          {job.error && (
            <p className="mt-2 text-sm text-red-600 truncate">{job.error}</p>
          )}

          {/* Timestamp */}
          <p className="mt-2 text-xs text-gray-400">
            {formatRelativeTime(job.updated_at)}
          </p>
        </div>

        {/* Status badge */}
        <div className="flex items-center gap-2">
          <div className={`${config.bg} ${config.color} px-2 py-1 rounded-full flex items-center gap-1`}>
            <StatusIcon
              className={`w-4 h-4 ${config.animate ? 'animate-spin' : ''}`}
            />
            <span className="text-xs font-medium">{config.label}</span>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-400" />
        </div>
      </div>
    </Link>
  );
}
