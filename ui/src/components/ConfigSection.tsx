import { ChevronDown, ChevronRight } from 'lucide-react';
import type { ReactNode } from 'react';

interface ConfigSectionProps {
  title: string;
  subtitle?: string;
  isOpen: boolean;
  onToggle: () => void;
  children: ReactNode;
}

export function ConfigSection({
  title,
  subtitle,
  isOpen,
  onToggle,
  children,
}: ConfigSectionProps) {
  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
      >
        <div>
          <span className="text-sm font-medium text-gray-900">{title}</span>
          {subtitle && (
            <span className="ml-2 text-xs text-gray-500">{subtitle}</span>
          )}
        </div>
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-500" />
        )}
      </button>
      {isOpen && (
        <div className="px-4 py-4 space-y-4 bg-white">{children}</div>
      )}
    </div>
  );
}
