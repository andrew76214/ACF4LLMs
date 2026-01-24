import { NewJobForm } from '../components/NewJobForm';

export function NewExperiment() {
  return (
    <div className="max-w-2xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">New Experiment</h1>
        <p className="text-gray-500 mt-1">
          Configure and start a new model compression experiment
        </p>
      </div>

      {/* Form */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <NewJobForm />
      </div>

      {/* Tips */}
      <div className="mt-6 bg-gray-50 rounded-lg border border-gray-200 p-4">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Tips</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>
            • For small models (like GPT-2), start with 5-10 episodes
          </li>
          <li>
            • Larger models (8B+) benefit from more episodes (10-20)
          </li>
          <li>
            • Use mock mode to test the workflow without GPU requirements
          </li>
          <li>
            • The AI will automatically explore different compression methods
          </li>
        </ul>
      </div>
    </div>
  );
}
