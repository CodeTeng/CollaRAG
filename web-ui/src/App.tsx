import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { IngestPage } from "./pages/IngestPage";
import { QueryPage } from "./pages/QueryPage";
import { AgentsPage } from "./pages/AgentsPage";
import { GraphPage } from "./pages/GraphPage";
import { EvaluatePage } from "./pages/EvaluatePage";
import { PluginsPage } from "./pages/PluginsPage";
import { ConfigPage } from "./pages/ConfigPage";
import { TracePage } from "./pages/TracePage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/query" replace />} />
          <Route path="/ingest" element={<IngestPage />} />
          <Route path="/query" element={<QueryPage />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/evaluate" element={<EvaluatePage />} />
          <Route path="/plugins" element={<PluginsPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/trace" element={<TracePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
