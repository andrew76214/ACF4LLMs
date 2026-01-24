import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Star,
  Cpu,
  Clock,
  Zap,
  Brain,
  Lightbulb,
} from 'lucide-react';
import type { Episode } from '../types';

interface EpisodeTimelineProps {
  episodes: Episode[];
  isLoading?: boolean;
}

// Method display names and colors
const METHOD_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  gptq: { bg: 'bg-blue-100', text: 'text-blue-800', label: 'GPTQ' },
  autoround: { bg: 'bg-purple-100', text: 'text-purple-800', label: 'AutoRound' },
  awq: { bg: 'bg-green-100', text: 'text-green-800', label: 'AWQ' },
  int8: { bg: 'bg-yellow-100', text: 'text-yellow-800', label: 'INT8' },
  lora: { bg: 'bg-pink-100', text: 'text-pink-800', label: 'LoRA' },
  qlora: { bg: 'bg-indigo-100', text: 'text-indigo-800', label: 'QLoRA' },
  asvd: { bg: 'bg-cyan-100', text: 'text-cyan-800', label: 'ASVD' },
  pruning: { bg: 'bg-orange-100', text: 'text-orange-800', label: 'Pruning' },
  pipeline: { bg: 'bg-gray-100', text: 'text-gray-800', label: 'Pipeline' },
};

function getMethodStyle(method: string | null | undefined) {
  if (!method) return { bg: 'bg-gray-100', text: 'text-gray-600', label: 'Unknown' };

  // Handle pipeline:name format
  if (method.startsWith('pipeline:')) {
    const pipelineName = method.replace('pipeline:', '');
    return {
      bg: 'bg-gradient-to-r from-purple-100 to-blue-100',
      text: 'text-purple-800',
      label: pipelineName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    };
  }

  const style = METHOD_STYLES[method.toLowerCase()];
  return style || { bg: 'bg-gray-100', text: 'text-gray-600', label: method.toUpperCase() };
}

function EpisodeCard({ episode, isExpanded, onToggle }: {
  episode: Episode;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const methodStyle = getMethodStyle(episode.decision.method);
  const result = episode.result;

  // Format parameters for display
  const formatParams = () => {
    const params = episode.decision.params;
    const parts: string[] = [];

    if (params.bits) parts.push(`${params.bits}-bit`);
    if (params.lora_rank) parts.push(`rank=${params.lora_rank}`);
    if (params.pruning_ratio) parts.push(`sparsity=${(Number(params.pruning_ratio) * 100).toFixed(0)}%`);
    if (params.asvd_rank_ratio) parts.push(`ratio=${params.asvd_rank_ratio}`);
    if (params.pipeline_name) parts.push(String(params.pipeline_name));

    return parts.join(', ') || 'default params';
  };

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden bg-white">
      {/* Collapsed Header */}
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}

          <span className="font-medium text-gray-700">
            Episode {episode.episode_id + 1}
          </span>

          {/* Method Badge */}
          <span className={`px-2 py-0.5 rounded text-xs font-medium ${methodStyle.bg} ${methodStyle.text}`}>
            {methodStyle.label}
          </span>

          {/* Pareto Star */}
          {episode.is_pareto && (
            <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" title="Pareto Optimal" />
          )}
        </div>

        <div className="flex items-center gap-4 text-sm text-gray-500">
          {result && (
            <>
              <span className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                {(result.accuracy * 100).toFixed(1)}%
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {result.latency_ms?.toFixed(0) || '-'} ms
              </span>
              {result.co2_grams !== null && (
                <span className="flex items-center gap-1 text-green-600">
                  {result.co2_grams?.toFixed(2) || '-'} g CO2
                </span>
              )}
            </>
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-4 py-4 border-t border-gray-100 bg-gray-50 space-y-4">
          {/* Coordinator Reasoning */}
          {episode.decision.reasoning && (
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="flex items-center gap-2 text-gray-700 font-medium mb-2">
                <Brain className="w-4 h-4 text-purple-500" />
                Coordinator Reasoning
              </div>
              <p className="text-gray-600 text-sm whitespace-pre-wrap leading-relaxed">
                {episode.decision.reasoning}
              </p>
            </div>
          )}

          {/* Skill Recommendations */}
          {episode.decision.skill_recommendations && episode.decision.skill_recommendations.length > 0 && (
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="flex items-center gap-2 text-gray-700 font-medium mb-2">
                <Lightbulb className="w-4 h-4 text-yellow-500" />
                Skill Recommendations
              </div>
              <div className="space-y-2">
                {episode.decision.skill_recommendations.map((rec, idx) => (
                  <div key={idx} className="text-sm flex items-start gap-2">
                    <span className="font-medium text-gray-700">{rec.skill_name}:</span>
                    <span className="text-gray-600">{rec.reasoning}</span>
                    <span className="text-xs text-gray-400">
                      ({(rec.confidence * 100).toFixed(0)}% confidence)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Parameters & Results Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Parameters */}
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="flex items-center gap-2 text-gray-700 font-medium mb-3">
                <Cpu className="w-4 h-4 text-blue-500" />
                Strategy Parameters
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="text-gray-500">Method</div>
                <div className="font-medium">{methodStyle.label}</div>

                {episode.decision.params.bits && (
                  <>
                    <div className="text-gray-500">Bit Width</div>
                    <div className="font-medium">{String(episode.decision.params.bits)}</div>
                  </>
                )}

                {episode.decision.params.lora_rank && (
                  <>
                    <div className="text-gray-500">LoRA Rank</div>
                    <div className="font-medium">{String(episode.decision.params.lora_rank)}</div>
                  </>
                )}

                {episode.decision.params.pruning_ratio && (
                  <>
                    <div className="text-gray-500">Sparsity</div>
                    <div className="font-medium">
                      {(Number(episode.decision.params.pruning_ratio) * 100).toFixed(0)}%
                    </div>
                  </>
                )}

                {episode.decision.params.pruning_method && (
                  <>
                    <div className="text-gray-500">Pruning Method</div>
                    <div className="font-medium">{String(episode.decision.params.pruning_method)}</div>
                  </>
                )}

                {episode.decision.params.asvd_rank_ratio && (
                  <>
                    <div className="text-gray-500">ASVD Rank Ratio</div>
                    <div className="font-medium">{String(episode.decision.params.asvd_rank_ratio)}</div>
                  </>
                )}

                {episode.decision.params.pipeline_name && (
                  <>
                    <div className="text-gray-500">Pipeline</div>
                    <div className="font-medium">{String(episode.decision.params.pipeline_name)}</div>
                  </>
                )}

                {episode.decision.timestamp && (
                  <>
                    <div className="text-gray-500">Timestamp</div>
                    <div className="font-medium text-xs">
                      {new Date(episode.decision.timestamp).toLocaleString()}
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Results */}
            {result && (
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="flex items-center gap-2 text-gray-700 font-medium mb-3">
                  <Zap className="w-4 h-4 text-green-500" />
                  Evaluation Results
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="text-gray-500">Accuracy</div>
                  <div className="font-medium text-green-600">
                    {(result.accuracy * 100).toFixed(2)}%
                  </div>

                  <div className="text-gray-500">Latency</div>
                  <div className="font-medium">{result.latency_ms?.toFixed(1) || '-'} ms</div>

                  <div className="text-gray-500">Compression</div>
                  <div className="font-medium">{result.compression_ratio?.toFixed(2) || '-'}x</div>

                  <div className="text-gray-500">Model Size</div>
                  <div className="font-medium">{result.model_size_gb?.toFixed(2) || '-'} GB</div>

                  <div className="text-gray-500">Memory</div>
                  <div className="font-medium">{result.memory_gb?.toFixed(2) || '-'} GB</div>

                  {result.co2_grams !== null && (
                    <>
                      <div className="text-gray-500">CO2 Emissions</div>
                      <div className="font-medium text-green-600">
                        {result.co2_grams?.toFixed(2) || '-'} g
                      </div>
                    </>
                  )}

                  {result.energy_joules !== null && (
                    <>
                      <div className="text-gray-500">Energy/Inference</div>
                      <div className="font-medium">
                        {result.energy_joules?.toFixed(4) || '-'} J
                      </div>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function EpisodeTimeline({ episodes, isLoading }: EpisodeTimelineProps) {
  const [expandedEpisodes, setExpandedEpisodes] = useState<Set<number>>(new Set());

  const toggleEpisode = (episodeId: number) => {
    setExpandedEpisodes(prev => {
      const next = new Set(prev);
      if (next.has(episodeId)) {
        next.delete(episodeId);
      } else {
        next.add(episodeId);
      }
      return next;
    });
  };

  const expandAll = () => {
    setExpandedEpisodes(new Set(episodes.map(e => e.episode_id)));
  };

  const collapseAll = () => {
    setExpandedEpisodes(new Set());
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-1/3"></div>
          <div className="h-16 bg-gray-100 rounded"></div>
          <div className="h-16 bg-gray-100 rounded"></div>
          <div className="h-16 bg-gray-100 rounded"></div>
        </div>
      </div>
    );
  }

  if (episodes.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg border border-gray-200 p-8 text-center text-gray-500">
        No episodes recorded yet
      </div>
    );
  }

  // Count Pareto optimal episodes
  const paretoCount = episodes.filter(e => e.is_pareto).length;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            Episode History
          </h2>
          <p className="text-sm text-gray-500">
            {episodes.length} episodes, {paretoCount} Pareto optimal
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={expandAll}
            className="px-3 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
          >
            Expand All
          </button>
          <button
            onClick={collapseAll}
            className="px-3 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Episode Cards */}
      <div className="space-y-2">
        {episodes.map((episode) => (
          <EpisodeCard
            key={episode.episode_id}
            episode={episode}
            isExpanded={expandedEpisodes.has(episode.episode_id)}
            onToggle={() => toggleEpisode(episode.episode_id)}
          />
        ))}
      </div>
    </div>
  );
}
