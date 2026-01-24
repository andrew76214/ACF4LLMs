import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  FlaskConical,
  PlusCircle,
  Settings,
  Activity,
} from 'lucide-react';
import { useHealth } from '../hooks/useJobs';

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
        {health && health.cuda_available === 'True' && (
          <div className="mt-1 text-xs text-gray-500">
            {health.gpu_count} GPU(s) available
          </div>
        )}
      </div>

      {/* Settings */}
      <div className="p-4 border-t border-gray-200">
        <button className="flex items-center gap-3 px-4 py-2 w-full text-gray-600 hover:bg-gray-100 rounded-lg transition-colors">
          <Settings className="w-5 h-5" />
          Settings
        </button>
      </div>
    </aside>
  );
}
