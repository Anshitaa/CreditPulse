import { Routes, Route, NavLink } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import Dashboard from "./pages/Dashboard";
import LiveFeed from "./pages/LiveFeed";
import WhatIfSimulator from "./pages/WhatIfSimulator";
import Governance from "./pages/Governance";
import AgentChat from "./pages/AgentChat";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "⬛" },
  { to: "/feed", label: "Live Feed", icon: "⚡" },
  { to: "/whatif", label: "What-If Simulator", icon: "🔬" },
  { to: "/governance", label: "Governance", icon: "⚖" },
  { to: "/agent", label: "AI Agent", icon: "💬" },
];

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="px-4 py-5 border-b border-gray-800">
          <h1 className="text-lg font-bold text-white tracking-tight">CreditPulse</h1>
          <p className="text-xs text-gray-500 mt-0.5">Fraud Intelligence Platform</p>
        </div>
        <nav className="flex-1 px-2 py-4 space-y-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-indigo-600 text-white"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-100"
                }`
              }
            >
              <span>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-3 border-t border-gray-800">
          <p className="text-xs text-gray-600">Built spec-first with Kiro</p>
          <p className="text-xs text-gray-700">Spec: CREDIT-001/002/003</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/feed" element={<LiveFeed />} />
          <Route path="/whatif" element={<WhatIfSimulator />} />
          <Route path="/governance" element={<Governance />} />
          <Route path="/agent" element={<AgentChat />} />
        </Routes>
      </main>

      <Toaster position="bottom-right" toastOptions={{ style: { background: "#1f2937", color: "#f9fafb", border: "1px solid #374151" } }} />
    </div>
  );
}
