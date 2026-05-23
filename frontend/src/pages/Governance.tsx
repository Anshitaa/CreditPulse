import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { api } from "../api";

function MetricRow({ label, value, threshold, passed }: { label: string; value: number; threshold: number; passed: boolean }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-800">
      <span className="text-sm text-gray-300">{label}</span>
      <div className="flex items-center gap-4">
        <div className="w-40 bg-gray-800 rounded-full h-2">
          <div className="h-2 rounded-full transition-all" style={{ width: `${Math.min(value / 0.15 * 100, 100)}%`, backgroundColor: passed ? "#22c55e" : "#ef4444" }} />
        </div>
        <span className={`text-sm font-mono font-bold w-14 text-right ${passed ? "text-green-400" : "text-red-400"}`}>
          {value.toFixed(3)}
        </span>
        <span className="text-xs text-gray-600 w-20">threshold {threshold}</span>
        <span className={`text-xs px-2 py-0.5 rounded ${passed ? "bg-green-900/40 text-green-400" : "bg-red-900/40 text-red-400"}`}>
          {passed ? "PASS" : "FAIL"}
        </span>
      </div>
    </div>
  );
}

export default function Governance() {
  const qc = useQueryClient();

  const { data: fairness } = useQuery({
    queryKey: ["fairness"],
    queryFn: () => api.get("/governance/fairness").then((r) => r.data),
    refetchInterval: 60_000,
  });

  const { data: drift } = useQuery({
    queryKey: ["drift"],
    queryFn: () => api.get("/governance/drift").then((r) => r.data),
    refetchInterval: 60_000,
  });

  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: () => api.get("/governance/models").then((r) => r.data),
  });

  const triggerFairness = useMutation({
    mutationFn: () => api.post("/governance/fairness/run").then((r) => r.data),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["fairness"] });
      toast[data.gate_passed ? "success" : "error"](
        data.gate_passed ? "Fairness gate PASSED" : "Fairness gate FAILED — review metrics"
      );
    },
    onError: () => toast.error("Fairness check failed to run"),
  });

  const triggerDrift = useMutation({
    mutationFn: () => api.post("/governance/drift/run").then((r) => r.data),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["drift"] });
      toast[data.drift_detected ? "error" : "success"](
        data.drift_detected ? "Drift detected — consider retraining" : "No significant drift detected"
      );
    },
    onError: () => toast.error("Drift check failed to run"),
  });

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Model Governance</h2>
          <p className="text-sm text-gray-500 mt-0.5">Fairness gates + PSI drift monitoring · Spec: CREDIT-001 NFR-004, CREDIT-002 FR-004</p>
        </div>
        <div className="flex gap-3">
          <button onClick={() => triggerFairness.mutate()} disabled={triggerFairness.isPending}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-sm text-white rounded-lg transition-colors">
            {triggerFairness.isPending ? "Running…" : "Run Fairness Gate"}
          </button>
          <button onClick={() => triggerDrift.mutate()} disabled={triggerDrift.isPending}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-sm text-white rounded-lg transition-colors">
            {triggerDrift.isPending ? "Running…" : "Run PSI Check"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fairness metrics */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">Fairness Gate Results</h3>
            {fairness && (
              <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${fairness.gate_passed ? "bg-green-900/40 text-green-400" : "bg-red-900/40 text-red-400"}`}>
                {fairness.gate_passed ? "Gate PASSED" : "Gate FAILED"}
              </span>
            )}
          </div>
          {fairness?.report?.group_results ? (
            Object.entries(fairness.report.group_results).map(([group, metrics]: [string, any]) => (
              <div key={group} className="mb-4">
                <p className="text-xs text-gray-500 mb-1 uppercase tracking-wider">{group.replace(/_/g, " ")}</p>
                <MetricRow
                  label="Demographic Parity Δ"
                  value={metrics.demographic_parity_difference ?? 0}
                  threshold={0.05}
                  passed={(metrics.demographic_parity_difference ?? 0) <= 0.05}
                />
                <MetricRow
                  label="Equal Opportunity Δ"
                  value={metrics.equal_opportunity_difference ?? 0}
                  threshold={0.05}
                  passed={(metrics.equal_opportunity_difference ?? 0) <= 0.05}
                />
              </div>
            ))
          ) : (
            <p className="text-sm text-gray-600 py-4">No fairness report yet. Click "Run Fairness Gate".</p>
          )}
          <p className="text-xs text-gray-700 mt-2">Kiro hook: .kiro/hooks/fairness-gate.sh (runs on model file save)</p>
        </div>

        {/* PSI drift */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">PSI Feature Drift</h3>
            {drift && (
              <span className={`text-xs px-2.5 py-1 rounded-full font-medium ${drift.drift_detected ? "bg-red-900/40 text-red-400" : "bg-green-900/40 text-green-400"}`}>
                {drift.recommendation}
              </span>
            )}
          </div>
          {drift?.features?.length > 0 ? (
            <div className="space-y-2">
              {drift.features.map((f: any) => (
                <div key={f.feature_name} className="flex items-center justify-between py-2 border-b border-gray-800">
                  <span className="text-sm text-gray-300">{f.feature_name}</span>
                  <div className="flex items-center gap-3">
                    <span className={`text-sm font-mono font-bold ${f.psi_score > 0.2 ? "text-red-400" : f.psi_score > 0.1 ? "text-amber-400" : "text-green-400"}`}>
                      PSI {f.psi_score?.toFixed(3)}
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      f.psi_score > 0.2 ? "bg-red-900/40 text-red-400"
                      : f.psi_score > 0.1 ? "bg-amber-900/40 text-amber-400"
                      : "bg-green-900/40 text-green-400"
                    }`}>
                      {f.psi_score > 0.2 ? "RETRAIN" : f.psi_score > 0.1 ? "MONITOR" : "STABLE"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-600 py-4">No drift report yet. Click "Run PSI Check".</p>
          )}
          <p className="text-xs text-gray-700 mt-2">Kiro hook: .kiro/hooks/psi-check.sh (runs pre-commit)</p>
        </div>
      </div>

      {/* Registered models */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-300 mb-4">MLflow Model Registry</h3>
        {models?.models?.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {models.models.map((m: any) => (
              <div key={m.name} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <p className="text-sm font-medium text-white">{m.name}</p>
                <div className="mt-2 space-y-1">
                  {m.versions.map((v: any) => (
                    <div key={v.version} className="flex items-center justify-between text-xs">
                      <span className="text-gray-400">v{v.version}</span>
                      <span className={`px-1.5 py-0.5 rounded ${v.stage === "Production" ? "bg-green-900/40 text-green-400" : v.stage === "Staging" ? "bg-blue-900/40 text-blue-400" : "bg-gray-700 text-gray-400"}`}>
                        {v.stage}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-600">No registered models yet. Run models/fraud_detector.py --train first.</p>
        )}
      </div>
    </div>
  );
}
