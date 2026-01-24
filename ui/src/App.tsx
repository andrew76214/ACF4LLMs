import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Experiments } from './pages/Experiments';
import { ExperimentDetail } from './pages/ExperimentDetail';
import { FilesystemExperimentDetail } from './pages/FilesystemExperimentDetail';
import { NewExperiment } from './pages/NewExperiment';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="experiments" element={<Experiments />} />
            <Route path="experiments/:jobId" element={<ExperimentDetail />} />
            <Route path="experiment/:experimentId" element={<FilesystemExperimentDetail />} />
            <Route path="new" element={<NewExperiment />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
