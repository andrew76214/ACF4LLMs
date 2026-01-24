import { useState } from 'react';
import {
  Brain,
  Cpu,
  Database,
  Gauge,
  GitBranch,
  Layers,
  Play,
  Square,
  Zap,
  RefreshCw,
  Target,
  BarChart3,
  Leaf,
  Timer,
  HardDrive,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  ArrowDown,
} from 'lucide-react';

interface NodeProps {
  title: string;
  subtitle?: string;
  icon: React.ReactNode;
  color: string;
  children?: React.ReactNode;
  isActive?: boolean;
  onClick?: () => void;
}

function Node({ title, subtitle, icon, color, children, isActive, onClick }: NodeProps) {
  return (
    <div
      className={`
        relative bg-white rounded-lg border-2 shadow-sm transition-all duration-200
        ${isActive ? `${color} shadow-md scale-105` : 'border-gray-200 hover:border-gray-300'}
        ${onClick ? 'cursor-pointer' : ''}
      `}
      onClick={onClick}
    >
      <div className="p-3">
        <div className="flex items-center gap-2 mb-1">
          <div className={`p-1.5 rounded-md ${color.replace('border-', 'bg-').replace('-500', '-100')} ${color.replace('border-', 'text-')}`}>
            {icon}
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 text-sm">{title}</h4>
            {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
          </div>
        </div>
        {children && <div className="mt-2 text-xs text-gray-600">{children}</div>}
      </div>
    </div>
  );
}

function Arrow({ direction = 'down', label }: { direction?: 'down' | 'right' | 'left'; label?: string }) {
  const Icon = direction === 'down' ? ArrowDown : ArrowRight;
  return (
    <div className={`flex ${direction === 'down' ? 'flex-col' : 'flex-row'} items-center justify-center ${direction === 'down' ? 'py-1' : 'px-1'}`}>
      <Icon className="w-4 h-4 text-gray-400" />
      {label && <span className="text-[10px] text-gray-400 ml-1">{label}</span>}
    </div>
  );
}


export function ArchitectureDiagram() {
  const [expanded, setExpanded] = useState(true);
  const [activeNode, setActiveNode] = useState<string | null>(null);

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <button
        className="w-full px-4 py-3 flex items-center justify-between bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-gray-200 hover:from-indigo-100 hover:to-purple-100 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-indigo-600" />
          <h2 className="text-lg font-semibold text-gray-900">Agent Architecture</h2>
          <span className="text-xs text-gray-500 bg-white px-2 py-0.5 rounded-full border">LangGraph + GPT-5.2</span>
        </div>
        {expanded ? (
          <ChevronUp className="w-5 h-5 text-gray-500" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-500" />
        )}
      </button>

      {/* Architecture Diagram */}
      {expanded && (
        <div className="p-4 overflow-x-auto">
          <div className="min-w-[700px]">
            {/* User Input Layer */}
            <div className="mb-4">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">User Input</div>
              <div className="flex gap-3 flex-wrap">
                <div className="bg-gray-50 rounded-md px-3 py-2 text-xs border border-gray-200">
                  <span className="text-gray-500">model_name:</span> <span className="font-mono text-indigo-600">gpt2, llama...</span>
                </div>
                <div className="bg-gray-50 rounded-md px-3 py-2 text-xs border border-gray-200">
                  <span className="text-gray-500">dataset:</span> <span className="font-mono text-indigo-600">gsm8k, mmlu...</span>
                </div>
                <div className="bg-gray-50 rounded-md px-3 py-2 text-xs border border-gray-200">
                  <span className="text-gray-500">max_episodes:</span> <span className="font-mono text-indigo-600">10</span>
                </div>
                <div className="bg-gray-50 rounded-md px-3 py-2 text-xs border border-gray-200">
                  <span className="text-gray-500">budget_hours:</span> <span className="font-mono text-indigo-600">4.0</span>
                </div>
              </div>
            </div>

            <Arrow />

            {/* Spec Inference */}
            <div className="flex justify-center mb-4">
              <Node
                title="Spec Inference"
                subtitle="src/coordinator/spec_inference.py"
                icon={<Database className="w-4 h-4" />}
                color="border-cyan-500"
                isActive={activeNode === 'spec'}
                onClick={() => setActiveNode(activeNode === 'spec' ? null : 'spec')}
              >
                <div className="flex gap-2 flex-wrap">
                  <span className="bg-cyan-50 text-cyan-700 px-1.5 py-0.5 rounded text-[10px]">HuggingFace Hub</span>
                  <span className="bg-cyan-50 text-cyan-700 px-1.5 py-0.5 rounded text-[10px]">MODEL_SIZE_DB</span>
                  <span className="bg-cyan-50 text-cyan-700 px-1.5 py-0.5 rounded text-[10px]">ModelSpec</span>
                </div>
              </Node>
            </div>

            <Arrow />

            {/* LangGraph State Machine */}
            <div className="border-2 border-indigo-200 rounded-xl p-4 bg-gradient-to-br from-indigo-50/50 to-purple-50/50">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-1.5 bg-indigo-100 rounded-md">
                  <Layers className="w-4 h-4 text-indigo-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-indigo-900">LangGraph State Machine</h3>
                  <p className="text-xs text-indigo-600">src/coordinator/langgraph_coordinator.py</p>
                </div>
              </div>

              {/* State Schema */}
              <div className="bg-white/80 rounded-lg p-3 mb-4 border border-indigo-100">
                <div className="text-xs font-medium text-gray-700 mb-2">CompressionState</div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px]">
                  <div className="bg-indigo-50 rounded px-2 py-1">
                    <span className="text-indigo-600 font-medium">model_name</span>
                  </div>
                  <div className="bg-indigo-50 rounded px-2 py-1">
                    <span className="text-indigo-600 font-medium">history[]</span>
                  </div>
                  <div className="bg-indigo-50 rounded px-2 py-1">
                    <span className="text-indigo-600 font-medium">pareto_frontier[]</span>
                  </div>
                  <div className="bg-indigo-50 rounded px-2 py-1">
                    <span className="text-indigo-600 font-medium">current_episode</span>
                  </div>
                  <div className="bg-indigo-50 rounded px-2 py-1">
                    <span className="text-indigo-600 font-medium">next_action</span>
                  </div>
                  <div className="bg-indigo-50 rounded px-2 py-1">
                    <span className="text-indigo-600 font-medium">should_terminate</span>
                  </div>
                  <div className="bg-indigo-50 rounded px-2 py-1">
                    <span className="text-indigo-600 font-medium">skill_recommendations</span>
                  </div>
                  <div className="bg-indigo-50 rounded px-2 py-1">
                    <span className="text-indigo-600 font-medium">messages[]</span>
                  </div>
                </div>
              </div>

              {/* START Node */}
              <div className="flex justify-center mb-3">
                <div className="bg-green-500 text-white px-4 py-1.5 rounded-full text-sm font-medium flex items-center gap-1.5">
                  <Play className="w-3.5 h-3.5" />
                  START
                </div>
              </div>

              <Arrow />

              {/* Coordinator Node */}
              <div className="flex justify-center mb-4">
                <div className="w-full max-w-lg">
                  <Node
                    title="Coordinator Node"
                    subtitle="GPT-5.2 Decision Engine"
                    icon={<Brain className="w-4 h-4" />}
                    color="border-purple-500"
                    isActive={activeNode === 'coordinator'}
                    onClick={() => setActiveNode(activeNode === 'coordinator' ? null : 'coordinator')}
                  >
                    <div className="space-y-2">
                      <div className="flex items-start gap-2">
                        <span className="text-purple-600 font-medium">1.</span>
                        <span>Check termination: episodes, budget, convergence</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <span className="text-purple-600 font-medium">2.</span>
                        <span>Analyze Pareto frontier + history</span>
                      </div>
                      <div className="flex items-start gap-2">
                        <span className="text-purple-600 font-medium">3.</span>
                        <span>Call GPT-5.2 API for decision</span>
                      </div>
                      <div className="mt-2 bg-gray-900 rounded-md p-2 text-[10px] font-mono text-gray-300">
                        {`{ "action": "quantization", "method": "gptq", "bits": 4 }`}
                      </div>
                    </div>
                  </Node>
                </div>
              </div>

              {/* Routing */}
              <div className="flex justify-center mb-3">
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <span className="bg-gray-100 px-2 py-1 rounded">next_action routing</span>
                </div>
              </div>

              {/* Worker Nodes */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
                <Node
                  title="Quantization"
                  icon={<Cpu className="w-4 h-4" />}
                  color="border-blue-500"
                  isActive={activeNode === 'quant'}
                  onClick={() => setActiveNode(activeNode === 'quant' ? null : 'quant')}
                >
                  <div className="space-y-1">
                    <div className="flex flex-wrap gap-1">
                      <span className="bg-blue-50 text-blue-700 px-1 rounded text-[10px]">AutoRound</span>
                      <span className="bg-blue-50 text-blue-700 px-1 rounded text-[10px]">GPTQ</span>
                      <span className="bg-blue-50 text-blue-700 px-1 rounded text-[10px]">AWQ</span>
                      <span className="bg-blue-50 text-blue-700 px-1 rounded text-[10px]">INT8</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      <span className="bg-green-50 text-green-700 px-1 rounded text-[10px]">LoRA</span>
                      <span className="bg-green-50 text-green-700 px-1 rounded text-[10px]">QLoRA</span>
                      <span className="bg-orange-50 text-orange-700 px-1 rounded text-[10px]">ASVD</span>
                    </div>
                  </div>
                </Node>

                <Node
                  title="Pruning"
                  icon={<Zap className="w-4 h-4" />}
                  color="border-amber-500"
                  isActive={activeNode === 'prune'}
                  onClick={() => setActiveNode(activeNode === 'prune' ? null : 'prune')}
                >
                  <div className="flex flex-wrap gap-1">
                    <span className="bg-amber-50 text-amber-700 px-1 rounded text-[10px]">Magnitude</span>
                    <span className="bg-amber-50 text-amber-700 px-1 rounded text-[10px]">Structured</span>
                  </div>
                </Node>

                <Node
                  title="Pipeline"
                  icon={<GitBranch className="w-4 h-4" />}
                  color="border-teal-500"
                  isActive={activeNode === 'pipeline'}
                  onClick={() => setActiveNode(activeNode === 'pipeline' ? null : 'pipeline')}
                >
                  <div className="flex flex-wrap gap-1">
                    <span className="bg-teal-50 text-teal-700 px-1 rounded text-[10px]">aggressive</span>
                    <span className="bg-teal-50 text-teal-700 px-1 rounded text-[10px]">balanced</span>
                    <span className="bg-teal-50 text-teal-700 px-1 rounded text-[10px]">quick_eval</span>
                  </div>
                </Node>

                <Node
                  title="Search"
                  icon={<Target className="w-4 h-4" />}
                  color="border-rose-500"
                  isActive={activeNode === 'search'}
                  onClick={() => setActiveNode(activeNode === 'search' ? null : 'search')}
                >
                  <div className="flex flex-wrap gap-1">
                    <span className="bg-rose-50 text-rose-700 px-1 rounded text-[10px]">Bayesian</span>
                    <span className="bg-rose-50 text-rose-700 px-1 rounded text-[10px]">Evolutionary</span>
                    <span className="bg-rose-50 text-rose-700 px-1 rounded text-[10px]">MAB</span>
                  </div>
                </Node>
              </div>

              <Arrow />

              {/* Evaluation Node */}
              <div className="flex justify-center mb-4">
                <div className="w-full max-w-xl">
                  <Node
                    title="Evaluation Node"
                    subtitle="src/agents/evaluation_agent.py"
                    icon={<Gauge className="w-4 h-4" />}
                    color="border-emerald-500"
                    isActive={activeNode === 'eval'}
                    onClick={() => setActiveNode(activeNode === 'eval' ? null : 'eval')}
                  >
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      <div className="flex items-center gap-1">
                        <BarChart3 className="w-3 h-3 text-emerald-600" />
                        <span>Benchmark</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Timer className="w-3 h-3 text-emerald-600" />
                        <span>Latency</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <HardDrive className="w-3 h-3 text-emerald-600" />
                        <span>Memory</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Leaf className="w-3 h-3 text-emerald-600" />
                        <span>CO2</span>
                      </div>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      <span className="bg-emerald-50 text-emerald-700 px-1 rounded text-[10px]">GSM8K</span>
                      <span className="bg-emerald-50 text-emerald-700 px-1 rounded text-[10px]">MMLU</span>
                      <span className="bg-emerald-50 text-emerald-700 px-1 rounded text-[10px]">HumanEval</span>
                      <span className="bg-emerald-50 text-emerald-700 px-1 rounded text-[10px]">TruthfulQA</span>
                      <span className="bg-emerald-50 text-emerald-700 px-1 rounded text-[10px]">HellaSwag</span>
                    </div>
                  </Node>
                </div>
              </div>

              <Arrow />

              {/* Update State Node */}
              <div className="flex justify-center mb-4">
                <div className="w-full max-w-md">
                  <Node
                    title="Update State"
                    subtitle="Pareto Frontier + Skill Memory"
                    icon={<RefreshCw className="w-4 h-4" />}
                    color="border-orange-500"
                    isActive={activeNode === 'update'}
                    onClick={() => setActiveNode(activeNode === 'update' ? null : 'update')}
                  >
                    <div className="space-y-1">
                      <div className="flex items-center gap-1">
                        <span className="text-orange-600">1.</span>
                        <span>Update 5D Pareto Frontier</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-orange-600">2.</span>
                        <span>Record to Skill Memory</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="text-orange-600">3.</span>
                        <span>Check convergence</span>
                      </div>
                    </div>
                  </Node>
                </div>
              </div>

              {/* Loop indicator */}
              <div className="flex justify-center items-center gap-2 mb-3">
                <div className="flex-1 h-px bg-gray-300"></div>
                <div className="flex items-center gap-1 text-xs text-gray-500 bg-white px-3 py-1 rounded-full border border-gray-200">
                  <RefreshCw className="w-3 h-3" />
                  Loop back to Coordinator
                </div>
                <div className="flex-1 h-px bg-gray-300"></div>
              </div>

              {/* END Node */}
              <div className="flex justify-center">
                <div className="bg-red-500 text-white px-4 py-1.5 rounded-full text-sm font-medium flex items-center gap-1.5">
                  <Square className="w-3.5 h-3.5" />
                  END
                </div>
              </div>

              {/* Termination conditions */}
              <div className="mt-3 flex justify-center">
                <div className="text-[10px] text-gray-500 bg-gray-100 rounded-md px-3 py-1.5 flex flex-wrap gap-2 justify-center">
                  <span>max_episodes reached</span>
                  <span className="text-gray-300">|</span>
                  <span>time_budget exceeded</span>
                  <span className="text-gray-300">|</span>
                  <span>convergence (5x no improvement)</span>
                </div>
              </div>
            </div>

            <Arrow />

            {/* Output Layer */}
            <div className="mt-4">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Output</div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <div className="bg-gray-50 rounded-md px-3 py-2 text-xs border border-gray-200">
                  <span className="font-medium text-gray-700">model_spec.json</span>
                </div>
                <div className="bg-gray-50 rounded-md px-3 py-2 text-xs border border-gray-200">
                  <span className="font-medium text-gray-700">pareto_frontier.json</span>
                </div>
                <div className="bg-gray-50 rounded-md px-3 py-2 text-xs border border-gray-200">
                  <span className="font-medium text-gray-700">final_results.json</span>
                </div>
                <div className="bg-gray-50 rounded-md px-3 py-2 text-xs border border-gray-200">
                  <span className="font-medium text-gray-700">visualization.html</span>
                </div>
              </div>
            </div>

            {/* 5D Optimization Space */}
            <div className="mt-4 p-3 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
              <div className="text-xs font-medium text-green-800 mb-2">5D Multi-Objective Optimization</div>
              <div className="flex flex-wrap gap-2">
                <div className="flex items-center gap-1 bg-white rounded px-2 py-1 text-xs border border-green-200">
                  <span className="text-green-600">Maximize:</span>
                  <span className="font-medium">Accuracy</span>
                </div>
                <div className="flex items-center gap-1 bg-white rounded px-2 py-1 text-xs border border-green-200">
                  <span className="text-red-600">Minimize:</span>
                  <span className="font-medium">Latency</span>
                </div>
                <div className="flex items-center gap-1 bg-white rounded px-2 py-1 text-xs border border-green-200">
                  <span className="text-red-600">Minimize:</span>
                  <span className="font-medium">Memory</span>
                </div>
                <div className="flex items-center gap-1 bg-white rounded px-2 py-1 text-xs border border-green-200">
                  <span className="text-red-600">Minimize:</span>
                  <span className="font-medium">Model Size</span>
                </div>
                <div className="flex items-center gap-1 bg-white rounded px-2 py-1 text-xs border border-green-200">
                  <span className="text-red-600">Minimize:</span>
                  <span className="font-medium">CO2 Emissions</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
