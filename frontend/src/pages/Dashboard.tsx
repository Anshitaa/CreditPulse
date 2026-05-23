import { useQuery } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { api } from "../api";

const DECISION_COLORS: Record<string, string> = {
  FRAUD: "#ef4444",
  REVIEW: "#f59e0b",
  CLEAR: "#22c55e",
};

function StatCard({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
      <p className={`text-3xl font-bold mt-1 ${color ?? "text-white"}`}>{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
    </div>
  );
}

export default function Dashboard() {
  const { data: summary, isLoading } = useQuery({
    queryKey: ["risk-summary"],
    queryFn: () => api.get("/risk/summary").then((r) => r.data),
    refetchInterval: 15_000,
  });

  const { data: topRisk } = useQuery({
    queryKey: ["top-risk"],
    queryFn: () => api.get("/risk/scores?min_score=75&limit=10").then((r) => r.data),
    refetchInterval: 15_000,
  });

  const pieData = summary
    ? [
        { name: "FRAUD", value: summary.fraud_count ?? 0 },
        { name: "REVIEW", value: summary.review_count ?? 0 },
        { name: "CLEAR", value: summary.clear_count ?? 0 },
      ]
    : [];

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white">Risk Overview</h2>
        <p className="text-sm text-gray-500 mt-0.5">Last 24 hours · auto-refreshes every 15s</p>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Transactions Scored" value={isLoading ? "—" : (summary?.total_scored_today ?? 0).toLocaleString()} />
        <StatCard label="Fraud Detected" value={isLoading ? "—" : summary?.fraud_count ?? 0} color="text-red-400" sub={`${summary?.fraud_rate_pct ?? 0}% fraud rate`} />
        <StatCard label="Pending Review" value={isLoading ? "—" : summary?.review_count ?? 0} color="text-amber-400" />
        <StatCard label="Avg Risk Score" value={isLoading ? "—" : summary?.avg_risk_score ?? 0} sub="0–100 scale" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Decision distribution pie */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Decision Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={DECISION_COLORS[entry.name]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Top risky transactions */}
        <div className="lg:col-span-2 bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Highest Risk Transactions (last 24h)</h3>
          <div className="overflow-auto max-h-64">
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="pb-2 pr-4">Transaction ID</th>
                  <th className="pb-2 pr-4">Score</th>
                  <th className="pb-2 pr-4">Decision</th>
                  <th className="pb-2">Scored At</th>
                </tr>
              </thead>
              <tbody>
                {(topRisk?.scores ?? []).map((row: any) => (
                  <tr key={row.txn_id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-1.5 pr-4 font-mono text-gray-400">{row.txn_id?.slice(0, 12)}…</td>
                    <td className="py-1.5 pr-4">
                      <span className={`font-bold ${row.composite_risk_score > 75 ? "text-red-400" : row.composite_risk_score > 40 ? "text-amber-400" : "text-green-400"}`}>
                        {row.composite_risk_score?.toFixed(1)}
                      </span>
                    </td>
                    <td className="py-1.5 pr-4">
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${row.decision === "FRAUD" ? "bg-red-900/60 text-red-300" : row.decision === "REVIEW" ? "bg-amber-900/60 text-amber-300" : "bg-green-900/60 text-green-300"}`}>
                        {row.decision}
                      </span>
                    </td>
                    <td className="py-1.5 text-gray-500">{new Date(row.scored_at).toLocaleTimeString()}</td>
                  </tr>
                ))}
                {(!topRisk?.scores || topRisk.scores.length === 0) && (
                  <tr><td colSpan={4} className="py-8 text-center text-gray-600">No high-risk transactions in the last 24h</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Spec reference */}
      <p className="text-xs text-gray-700">Spec: CREDIT-001 FR-001, CREDIT-002 FR-005</p>
    </div>
  );
}
