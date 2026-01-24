import { Link } from 'react-router-dom';
import {
  FlaskConical,
  CheckCircle2,
  Loader2,
  AlertTriangle,
  ArrowRight,
  TrendingUp,
} from 'lucide-react';
import { useJobs, useHealth } from '../hooks/useJobs';
import { MetricsCard } from '../components/MetricsCard';
import { JobCard } from '../components/JobCard';
import { ArchitectureDiagram } from '../components/ArchitectureDiagram';

export function Dashboard() {
  const { data: jobs, isLoading } = useJobs();
  const { data: health } = useHealth();

  // Calculate stats
  const stats = {
    total: jobs?.length || 0,
    running: jobs?.filter((j) => j.status === 'running').length || 0,
    completed: jobs?.filter((j) => j.status === 'completed').length || 0,
    failed: jobs?.filter((j) => j.status === 'failed').length || 0,
  };

  // Get recent jobs
  const recentJobs = jobs?.slice(0, 5) || [];

  // Get running jobs
  const runningJobs = jobs?.filter((j) => j.status === 'running') || [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">
          Overview of your model compression experiments
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricsCard
          title="Total Experiments"
          value={stats.total}
          icon={<FlaskConical className="w-5 h-5" />}
        />
        <MetricsCard
          title="Running"
          value={stats.running}
          icon={<Loader2 className="w-5 h-5 animate-spin" />}
          variant={stats.running > 0 ? 'warning' : 'default'}
        />
        <MetricsCard
          title="Completed"
          value={stats.completed}
          icon={<CheckCircle2 className="w-5 h-5" />}
          variant="success"
        />
        <MetricsCard
          title="Failed"
          value={stats.failed}
          icon={<AlertTriangle className="w-5 h-5" />}
          variant={stats.failed > 0 ? 'error' : 'default'}
        />
      </div>

      {/* System Status */}
      {health && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">
            System Status
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <span className="text-sm text-gray-500">Backend</span>
              <p className="text-green-600 font-medium">Online</p>
            </div>
            <div>
              <span className="text-sm text-gray-500">CUDA</span>
              <p
                className={`font-medium ${
                  health.cuda_available === 'True'
                    ? 'text-green-600'
                    : 'text-yellow-600'
                }`}
              >
                {health.cuda_available === 'True' ? 'Available' : 'Not Available'}
              </p>
            </div>
            <div>
              <span className="text-sm text-gray-500">GPUs</span>
              <p className="font-medium text-gray-900">{health.gpu_count}</p>
            </div>
            <div>
              <span className="text-sm text-gray-500">Active Jobs</span>
              <p className="font-medium text-gray-900">{stats.running}</p>
            </div>
          </div>
        </div>
      )}

      {/* Agent Architecture Diagram */}
      <ArchitectureDiagram />

      {/* Running Jobs */}
      {runningJobs.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
              Running Experiments
            </h2>
          </div>
          <div className="space-y-3">
            {runningJobs.map((job) => (
              <JobCard key={job.job_id} job={job} />
            ))}
          </div>
        </div>
      )}

      {/* Recent Experiments */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-gray-500" />
            Recent Experiments
          </h2>
          <Link
            to="/experiments"
            className="text-primary-600 hover:text-primary-700 text-sm font-medium flex items-center gap-1"
          >
            View all
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
          </div>
        ) : recentJobs.length === 0 ? (
          <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
            <FlaskConical className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No experiments yet
            </h3>
            <p className="text-gray-500 mb-4">
              Start your first model compression experiment
            </p>
            <Link
              to="/new"
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              New Experiment
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {recentJobs.map((job) => (
              <JobCard key={job.job_id} job={job} />
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="bg-gradient-to-r from-primary-50 to-primary-100 rounded-lg border border-primary-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">
          Quick Start
        </h2>
        <p className="text-gray-600 mb-4">
          Ready to compress a model? Start a new experiment with just a model
          name and dataset.
        </p>
        <Link
          to="/new"
          className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          Start New Experiment
          <ArrowRight className="w-4 h-4" />
        </Link>
      </div>
    </div>
  );
}
