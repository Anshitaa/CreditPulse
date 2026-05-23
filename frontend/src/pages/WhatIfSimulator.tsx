import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { api } from "../api";

interface WhatIfResult {
  fraud_risk_score: number;
  decision: string;
  top_features: Array<{ feature: string; shap_value: number; description: string; direction: string }>;
  fraud_probability: number;
}

interface CounterfactualResult {
  original_risk_score: number;
  counterfactuals: Array<{
    target_risk_score: number;
    changes: Array<{ feature: string; original: any; counterfactual: any }>;
    feasibility: string;
  }>;
  explanation: string;
}

const MERCHANT_CATEGORIES = [
  "grocery", "restaurant", "gas_station", "retail", "online_retail",
  "travel", "entertainment", "atm_withdrawal", "peer_transfer",
  "wire_transfer", "gambling", "cryptocurrency", "money_service",
];

export default function WhatIfSimulator() {
  const [form, setForm] = useState({
    account_id: "acct-demo",
    amount: 1000,
    merchant_category: "retail",
    is_foreign_merchant: false,
    hour_of_day: 14,
    day_of_week: 2,
    txn_velocity_1h: 1,
    amount_vs_avg_ratio: 1.5,
  });

  const scoreMutation = useMutation<WhatIfResult, Error, typeof form>({
    mutationFn: (data) => api.post("/explain/whatif", data).then((r) => r.data),
  });

  const cfMutation = useMutation<CounterfactualResult, Error, typeof form>({
    mutationFn: (data) => api.post("/explain/counterfactual", data).then((r) => r.data),
  });

  const score = scoreMutation.data;
  const cf = cfMutation.data;

  const scoreColor = score
    ? score.fraud_risk_score > 75 ? "#ef4444" : score.fraud_risk_score > 40 ? "#f59e0b" : "#22c55e"
    : "#6b7280";

  const shapData = (score?.top_features ?? []).map((f) => ({
    name: f.feature.replace(/_/g, " "),
    value: Math.abs(f.shap_value),
    direction: f.direction,
  }));

  const update = (k: string, v: any) => setForm((p) => ({ ...p, [k]: v }));

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white">What-If Simulator</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Adjust transaction attributes and see how the fraud risk score changes in real time.
          Spec: CREDIT-003 FR-003
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Controls */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-medium text-gray-300">Transaction Attributes</h3>

          <label className="block">
            <span className="text-xs text-gray-500">Amount ($)</span>
            <input
              type="number" min={1} max={50000}
              value={form.amount}
              onChange={(e) => update("amount", Number(e.target.value))}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:ring-1 focus:ring-indigo-500 outline-none"
            />
          </label>

          <label className="block">
            <span className="text-xs text-gray-500">Merchant Category</span>
            <select
              value={form.merchant_category}
              onChange={(e) => update("merchant_category", e.target.value)}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:ring-1 focus:ring-indigo-500 outline-none"
            >
              {MERCHANT_CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-xs text-gray-500">Hour of Day (0–23)</span>
              <input type="number" min={0} max={23} value={form.hour_of_day} onChange={(e) => update("hour_of_day", Number(e.target.value))}
                className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </label>
            <label className="block">
              <span className="text-xs text-gray-500">Velocity (txns/hour)</span>
              <input type="number" min={0} max={50} value={form.txn_velocity_1h} onChange={(e) => update("txn_velocity_1h", Number(e.target.value))}
                className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </label>
          </div>

          <label className="block">
            <span className="text-xs text-gray-500">Amount vs Account Average (ratio)</span>
            <input type="number" min={0.1} max={50} step={0.1} value={form.amount_vs_avg_ratio} onChange={(e) => update("amount_vs_avg_ratio", Number(e.target.value))}
              className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={form.is_foreign_merchant} onChange={(e) => update("is_foreign_merchant", e.target.checked)}
              className="w-4 h-4 accent-indigo-500"
            />
            <span className="text-sm text-gray-300">Foreign merchant</span>
          </label>

          <div className="flex gap-3 pt-2">
            <button onClick={() => scoreMutation.mutate(form)} disabled={scoreMutation.isPending}
              className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-medium py-2 rounded-lg transition-colors">
              {scoreMutation.isPending ? "Scoring…" : "Score Transaction"}
            </button>
            <button onClick={() => cfMutation.mutate(form)} disabled={cfMutation.isPending || !score}
              className="flex-1 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 text-white text-sm font-medium py-2 rounded-lg transition-colors">
              {cfMutation.isPending ? "Generating…" : "Get Counterfactual"}
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="space-y-4">
          {/* Score gauge */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Risk Score</h3>
            {score ? (
              <div className="flex items-center gap-6">
                <div className="text-center">
                  <p className="text-5xl font-black" style={{ color: scoreColor }}>
                    {score.fraud_risk_score.toFixed(0)}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">out of 100</p>
                </div>
                <div>
                  <span className={`px-3 py-1 rounded-full text-sm font-bold border ${
                    score.decision === "FRAUD" ? "text-red-400 bg-red-900/30 border-red-700"
                    : score.decision === "REVIEW" ? "text-amber-400 bg-amber-900/30 border-amber-700"
                    : "text-green-400 bg-green-900/30 border-green-700"
                  }`}>
                    {score.decision}
                  </span>
                  <p className="text-xs text-gray-500 mt-2">Fraud probability: {(score.fraud_probability * 100).toFixed(1)}%</p>
                </div>
              </div>
            ) : (
              <p className="text-sm text-gray-600">Click "Score Transaction" to see the result</p>
            )}
          </div>

          {/* SHAP chart */}
          {shapData.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Top Risk Drivers (SHAP)</h3>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={shapData} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" tick={{ fill: "#6b7280", fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} width={130} />
                  <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 12 }} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {shapData.map((entry) => (
                      <Cell key={entry.name} fill={entry.direction === "increases_risk" ? "#ef4444" : "#22c55e"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Counterfactuals */}
          {cf && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <h3 className="text-sm font-medium text-gray-300 mb-2">Counterfactual Explanation</h3>
              <p className="text-xs text-gray-400 mb-3">{cf.explanation}</p>
              {cf.counterfactuals.map((c, i) => (
                <div key={i} className="mb-3 p-3 bg-gray-800 rounded-lg border border-gray-700">
                  <p className="text-xs text-green-400 font-medium mb-2">Target score: {c.target_risk_score?.toFixed(0)} · {c.feasibility}</p>
                  {c.changes.map((ch) => (
                    <p key={ch.feature} className="text-xs text-gray-400">
                      <span className="text-gray-500">Change</span>{" "}
                      <span className="text-white font-medium">{ch.feature}</span>{" "}
                      <span className="text-gray-500">from</span>{" "}
                      <span className="text-red-400">{String(ch.original)}</span>{" "}
                      <span className="text-gray-500">to</span>{" "}
                      <span className="text-green-400">{String(ch.counterfactual)}</span>
                    </p>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
