import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Loader2, Filter, PlusCircle, Search } from 'lucide-react';
import { useJobs } from '../hooks/useJobs';
import { JobCard } from '../components/JobCard';
import type { JobStatus } from '../types';

const statusFilters: { value: JobStatus | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'running', label: 'Running' },
  { value: 'pending', label: 'Pending' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
];

export function Experiments() {
  const [statusFilter, setStatusFilter] = useState<JobStatus | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const { data: jobs, isLoading, error } = useJobs(
    statusFilter === 'all' ? undefined : statusFilter
  );

  // Filter jobs by search query
  const filteredJobs = jobs?.filter((job) =>
    searchQuery
      ? job.job_id.toLowerCase().includes(searchQuery.toLowerCase())
      : true
  );

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
            placeholder="Search by job ID..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
          />
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

      {/* Job list */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-600">
          Failed to load experiments: {error.message}
        </div>
      ) : !filteredJobs || filteredJobs.length === 0 ? (
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
          {filteredJobs.map((job) => (
            <JobCard key={job.job_id} job={job} />
          ))}
        </div>
      )}

      {/* Stats summary */}
      {jobs && jobs.length > 0 && (
        <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 flex items-center justify-between text-sm text-gray-600">
          <span>
            Showing {filteredJobs?.length || 0} of {jobs.length} experiments
          </span>
          <div className="flex gap-4">
            <span>
              <span className="font-medium text-green-600">
                {jobs.filter((j) => j.status === 'completed').length}
              </span>{' '}
              completed
            </span>
            <span>
              <span className="font-medium text-blue-600">
                {jobs.filter((j) => j.status === 'running').length}
              </span>{' '}
              running
            </span>
            <span>
              <span className="font-medium text-red-600">
                {jobs.filter((j) => j.status === 'failed').length}
              </span>{' '}
              failed
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
