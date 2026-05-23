import { useEffect, useRef, useState } from "react";

interface ScoredTransaction {
  txn_id: string;
  fraud_risk_score: number;
  decision: string;
  inference_latency_ms: number;
  scored_at: string;
  top_features?: Array<{ feature: string; shap_value: number; direction: string }>;
}

const DECISION_CLASS: Record<string, string> = {
  FRAUD: "text-red-400 bg-red-900/30 border-red-800",
  REVIEW: "text-amber-400 bg-amber-900/30 border-amber-800",
  CLEAR: "text-green-400 bg-green-900/30 border-green-800",
};

const WS_URL = "ws://127.0.0.1:8000/ws/scores";

export default function LiveFeed() {
  const [transactions, setTransactions] = useState<ScoredTransaction[]>([]);
  const [connected, setConnected] = useState(false);
  const [latencies, setLatencies] = useState<number[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000); // reconnect
      };
      ws.onmessage = (evt) => {
        try {
          const txn: ScoredTransaction = JSON.parse(evt.data);
          setTransactions((prev) => [txn, ...prev].slice(0, 100)); // keep last 100
          setLatencies((prev) => [...prev, txn.inference_latency_ms].slice(-50));
        } catch {}
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  const p99 = latencies.length > 0 ? Math.round(latencies.sort((a, b) => a - b)[Math.floor(latencies.length * 0.99)] ?? 0) : 0;
  const avgLatency = latencies.length > 0 ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : 0;
  const fraudCount = transactions.filter((t) => t.decision === "FRAUD").length;

  return (
    <div className="p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Live Transaction Feed</h2>
          <p className="text-sm text-gray-500 mt-0.5">Real-time scored transactions via WebSocket · Spec: CREDIT-001 FR-001</p>
        </div>
        <span className={`flex items-center gap-2 text-sm px-3 py-1.5 rounded-full border ${connected ? "text-green-400 bg-green-900/20 border-green-800" : "text-gray-500 bg-gray-800 border-gray-700"}`}>
          <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400 animate-pulse" : "bg-gray-600"}`} />
          {connected ? "Connected" : "Connecting…"}
        </span>
      </div>

      {/* Live stats */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "Total Received", value: transactions.length },
          { label: "Fraud Flagged", value: fraudCount, color: "text-red-400" },
          { label: "Avg Latency", value: `${avgLatency}ms` },
          { label: "p99 Latency", value: `${p99}ms`, color: p99 > 100 ? "text-red-400" : "text-green-400" },
        ].map((s) => (
          <div key={s.label} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wider">{s.label}</p>
            <p className={`text-2xl font-bold mt-1 ${s.color ?? "text-white"}`}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* Transaction rows */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800 text-xs text-gray-500 grid grid-cols-5 gap-4 font-medium uppercase tracking-wider">
          <span>Transaction ID</span>
          <span>Risk Score</span>
          <span>Decision</span>
          <span>Top Feature</span>
          <span>Latency / Time</span>
        </div>
        <div className="divide-y divide-gray-800/60 max-h-[60vh] overflow-auto">
          {transactions.map((txn) => (
            <div key={txn.txn_id} className="px-4 py-2.5 grid grid-cols-5 gap-4 text-sm items-center hover:bg-gray-800/30 transition-colors">
              <span className="font-mono text-xs text-gray-400">{txn.txn_id.slice(0, 14)}…</span>
              <span>
                <span className={`font-bold text-lg ${txn.fraud_risk_score > 75 ? "text-red-400" : txn.fraud_risk_score > 40 ? "text-amber-400" : "text-green-400"}`}>
                  {txn.fraud_risk_score?.toFixed(1)}
                </span>
                <span className="text-gray-600 text-xs ml-1">/100</span>
              </span>
              <span className={`inline-flex items-center px-2 py-0.5 rounded border text-xs font-medium w-fit ${DECISION_CLASS[txn.decision]}`}>
                {txn.decision}
              </span>
              <span className="text-xs text-gray-500 truncate">
                {txn.top_features?.[0]?.feature ?? "—"}
              </span>
              <span className="text-xs text-gray-500">
                {txn.inference_latency_ms?.toFixed(1)}ms · {new Date(txn.scored_at).toLocaleTimeString()}
              </span>
            </div>
          ))}
          {transactions.length === 0 && (
            <div className="py-16 text-center text-gray-600 text-sm">
              Waiting for scored transactions…
              <br />
              <span className="text-xs">POST to /score or run the Kafka producer to see live data</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
