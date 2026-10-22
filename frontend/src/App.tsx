import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import { useBackend } from "./hooks/useBackend";
import LandingPage from "./pages/LandingPage";

// Heavier pages (charts, viewers) are code-split so the landing page loads fast.
const AnalyzePage = lazy(() => import("./pages/AnalyzePage"));
const HistoryPage = lazy(() => import("./pages/HistoryPage"));
const MethodologyPage = lazy(() => import("./pages/MethodologyPage"));

export default function App() {
  const { state } = useBackend();
  const page = (el: React.ReactNode) => (
    <Suspense fallback={<div className="flex justify-center py-24"><Spinner className="h-6 w-6 text-cyan" /></div>}>{el}</Suspense>
  );
  return (
    <Routes>
      <Route element={<Layout backend={state} />}>
        <Route index element={<LandingPage />} />
        <Route path="analyze" element={page(<AnalyzePage backend={state} />)} />
        <Route path="analysis/:id" element={page(<AnalyzePage backend={state} />)} />
        <Route path="history" element={page(<HistoryPage />)} />
        <Route path="methodology" element={page(<MethodologyPage backend={state} />)} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
