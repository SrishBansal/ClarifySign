// ClarifySign - DialogueHistory Component
import React from "react";
import type { ConversationTurn } from "../types/api";

interface DialogueHistoryProps {
  history: ConversationTurn[];
  onClear?: () => void;
}

export const DialogueHistory: React.FC<DialogueHistoryProps> = ({
  history,
  onClear,
}) => {
  if (history.length === 0) {
    return (
      <div className="card history-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "var(--space-3)" }}>
          <h3 style={{ fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-primary)" }}>
            Session History
          </h3>
          <span className="badge badge-muted">Empty</span>
        </div>
        <div className="placeholder-card" style={{ padding: "var(--space-6)" }}>
          <div className="placeholder-icon">📜</div>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
            Predictions and clarification interactions will appear here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card history-card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "var(--space-4)",
        }}
      >
        <h3
          style={{
            fontSize: "0.95rem",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
            color: "var(--text-primary)",
            margin: 0,
          }}
        >
          Session History ({history.length})
        </h3>
        {onClear && (
          <button
            className="btn btn-secondary"
            style={{ padding: "2px 8px", fontSize: "0.75rem" }}
            onClick={onClear}
          >
            Clear
          </button>
        )}
      </div>

      <div className="history-list">
        {history.map((turn) => {
          const dateStr = new Date(turn.timestamp).toLocaleTimeString();
          return (
            <div key={turn.id} className="history-turn">
              <div className="history-turn-header">
                <div style={{ display: "flex", alignItems: "center", gap: "var(--space-2)" }}>
                  <span
                    className={`badge ${
                      turn.type === "resolved"
                        ? "badge-success"
                        : turn.type === "clarification"
                        ? "badge-warning"
                        : "badge-primary"
                    }`}
                  >
                    {turn.type}
                  </span>
                  {turn.label && (
                    <span className="history-turn-label">{turn.label}</span>
                  )}
                </div>
                <span className="history-turn-time">{dateStr}</span>
              </div>

              {turn.confidence !== undefined && (
                <div
                  style={{
                    fontSize: "0.75rem",
                    color: "var(--text-muted)",
                    marginTop: "var(--space-1)",
                  }}
                >
                  Confidence: {(turn.confidence * 100).toFixed(1)}%
                  {turn.wasAmbiguous && " • Ambiguity resolved via EIG"}
                </div>
              )}

              {turn.question && (
                <div
                  style={{
                    fontSize: "0.8rem",
                    color: "var(--text-secondary)",
                    marginTop: "var(--space-1)",
                    fontStyle: "italic",
                  }}
                >
                  Q: {turn.question}
                </div>
              )}

              {turn.answer && (
                <div
                  style={{
                    fontSize: "0.8rem",
                    color: "var(--accent-primary)",
                    marginTop: "var(--space-1)",
                    fontWeight: 600,
                  }}
                >
                  A: {turn.answer}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
