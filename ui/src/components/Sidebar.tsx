import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  FlaskConical,
  PlusCircle,
  Settings,
  Activity,
  Cpu,
  X,
} from 'lucide-react';
import { useHealth, useGpu } from '../hooks/useJobs';

interface SidebarProps {
  onNavigate?: () => void;
}

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/experiments', icon: FlaskConical, label: 'Experiments' },
  { to: '/new', icon: PlusCircle, label: 'New Experiment' },
];

export function Sidebar({ onNavigate }: SidebarProps) {
  const { data: health, isError } = useHealth();
  const { data: gpuStatus } = useGpu();
  const [showSettings, setShowSettings] = useState(false);

  return (
    <aside className="h-screen w-64 bg-white border-r border-gray-200 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <span className="text-2xl">🌿</span>
          Green AI
        </h1>
        <p className="text-sm text-gray-500 mt-1">Compression Dashboard</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        <ul className="space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <li key={to}>
              <NavLink
                to={to}
                onClick={onNavigate}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-primary-50 text-primary-700 font-medium'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`
                }
              >
                <Icon className="w-5 h-5" />
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* System Status */}
      <div className="p-4 border-t border-gray-200">
        <div className="flex items-center gap-2 text-sm">
          <Activity className="w-4 h-4" />
          <span className="text-gray-600">Backend:</span>
          {isError ? (
            <span className="text-red-600 font-medium">Offline</span>
          ) : health ? (
            <span className="text-green-600 font-medium">Online</span>
          ) : (
            <span className="text-yellow-600 font-medium">Checking...</span>
          )}
        </div>

        {/* GPU Status */}
        {gpuStatus?.available && gpuStatus.gpus.length > 0 && (
          <div className="mt-3 space-y-2">
            {gpuStatus.gpus.map((gpu) => (
              <div key={gpu.index} className="text-xs">
                <div className="flex items-center gap-1.5 text-gray-700">
                  <Cpu className="w-3 h-3" />
                  <span className="font-medium truncate" title={gpu.name}>
                    {gpu.name}
                  </span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        gpu.memory_utilization > 80
                          ? 'bg-red-500'
                          : gpu.memory_utilization > 50
                            ? 'bg-yellow-500'
                            : 'bg-green-500'
                      }`}
                      style={{ width: `${gpu.memory_utilization}%` }}
                    />
                  </div>
                  <span className="text-gray-500 w-12 text-right">
                    {gpu.memory_utilization}%
                  </span>
                </div>
                <div className="text-gray-400 mt-0.5">
                  {gpu.memory_used_gb} / {gpu.memory_total_gb} GB
                </div>
              </div>
            ))}
          </div>
        )}

        {health && health.cuda_available !== 'True' && (
          <div className="mt-2 text-xs text-gray-400">No GPU available</div>
        )}
      </div>

      {/* Settings */}
      <div className="p-4 border-t border-gray-200">
        <button
          onClick={() => setShowSettings(true)}
          className="flex items-center gap-3 px-4 py-2 w-full text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <Settings className="w-5 h-5" />
          Settings
        </button>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Settings</h2>
              <button
                onClick={() => setShowSettings(false)}
                className="p-1 text-gray-400 hover:text-gray-600 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  API Endpoint
                </label>
                <input
                  type="text"
                  readOnly
                  value={import.meta.env.VITE_API_URL || '/api'}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-gray-600 text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Configured at build time via VITE_API_URL
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  System Info
                </label>
                <div className="bg-gray-50 rounded-lg p-3 text-sm text-gray-600 space-y-1">
                  <div>Status: {health?.status || 'Unknown'}</div>
                  <div>CUDA: {health?.cuda_available === 'True' ? 'Available' : 'Not available'}</div>
                  <div>GPUs: {health?.gpu_count || '0'}</div>
                </div>
              </div>
            </div>
            <div className="p-4 border-t border-gray-200 flex justify-end">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
