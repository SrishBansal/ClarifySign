// ClarifySign - StatusBar Component
import React from "react";
import { useUIStore } from "../store/uiStore";
import { useInferenceStore } from "../store/inferenceStore";
import { useDialogueStore } from "../store/dialogueStore";

export const StatusBar: React.FC = () => {
  const { backendOnline, language, isCapturing } = useUIStore();
  const { frameCount, lastPrediction } = useInferenceStore();
  const { currentAction, dialogueId } = useDialogueStore();

  return (
    <footer className="status-bar">
      <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)" }}>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: backendOnline
                ? "var(--badge-success-text)"
                : "var(--badge-danger-text)",
              display: "inline-block",
            }}
          />
          {backendOnline ? "API Connected" : "API Offline"}
        </span>

        <span>
          Capture: {isCapturing ? "Streaming (10 FPS)" : "Standby"}
        </span>

        <span>Processed: {frameCount} frames</span>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "var(--space-4)" }}>
        {lastPrediction && (
          <span>
            Latency:{" "}
            <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-primary)" }}>
              {lastPrediction.inference_ms.toFixed(1)} ms
            </span>
          </span>
        )}

        <span>
          Dialogue State:{" "}
          <span style={{ color: "var(--accent-primary)", fontWeight: 600 }}>
            {currentAction.toUpperCase()}
          </span>
          {dialogueId && ` (${dialogueId.slice(0, 8)})`}
        </span>

        <span>Target: {language}</span>
      </div>
    </footer>
  );
};
