import { useState, useRef, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "../api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  ts: Date;
}

const EXAMPLE_QUERIES = [
  "Are there any high-risk transactions right now?",
  "Is the fraud model showing signs of drift?",
  "What fairness metrics does the current model have?",
  "What does FCRA require when we deny a credit request?",
];

export default function AgentChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hello! I'm the CreditPulse Fraud Intelligence Agent. I can answer questions about transactions, risk scores, model health, and financial regulations.\n\nTry asking me something like: *\"Why was transaction X flagged?\"* or *\"Should we retrain the model?\"*",
      ts: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  const chatMutation = useMutation({
    mutationFn: (query: string) =>
      api.post("/agent/chat", { query }).then((r) => r.data),
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        { id: String(Date.now()), role: "assistant", content: data.answer, ts: new Date() },
      ]);
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        { id: String(Date.now()), role: "assistant", content: "Sorry, the agent encountered an error. Make sure the API is running and an LLM API key is configured.", ts: new Date() },
      ]);
    },
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = (query: string) => {
    if (!query.trim()) return;
    setMessages((prev) => [
      ...prev,
      { id: String(Date.now()), role: "user", content: query, ts: new Date() },
    ]);
    setInput("");
    chatMutation.mutate(query);
  };

  return (
    <div className="flex flex-col h-screen p-6">
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-white">AI Agent Chat</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          LangChain ReAct agent · 5 tools (SHAP, counterfactual, risk scores, drift, RAG) · Spec: CREDIT-003 FR-004
        </p>
      </div>

      {/* Example queries */}
      <div className="flex flex-wrap gap-2 mb-4">
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            onClick={() => sendMessage(q)}
            className="text-xs px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 rounded-full transition-colors"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Message list */}
      <div className="flex-1 overflow-auto bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-4 min-h-0">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${msg.role === "user" ? "bg-indigo-600 text-white" : "bg-gray-700 text-gray-300"}`}>
              {msg.role === "user" ? "U" : "AI"}
            </div>
            <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${msg.role === "user" ? "bg-indigo-600 text-white rounded-tr-sm" : "bg-gray-800 text-gray-100 rounded-tl-sm"}`}>
              <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
              <p className="text-xs mt-1.5 opacity-50">{msg.ts.toLocaleTimeString()}</p>
            </div>
          </div>
        ))}
        {chatMutation.isPending && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-xs text-gray-300">AI</div>
            <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1.5 items-center h-5">
                {[0, 1, 2].map((i) => (
                  <span key={i} className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={(e) => { e.preventDefault(); sendMessage(input); }}
        className="mt-4 flex gap-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about transactions, risk scores, model health, or regulations…"
          disabled={chatMutation.isPending}
          className="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:ring-1 focus:ring-indigo-500 outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || chatMutation.isPending}
          className="px-5 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-medium rounded-xl transition-colors"
        >
          Send
        </button>
      </form>
    </div>
  );
}
