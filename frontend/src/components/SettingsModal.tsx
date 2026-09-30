// ClarifySign - SettingsModal Component
import React from "react";
import { useUIStore } from "../store/uiStore";
import { SUPPORTED_LANGUAGES, type AppLanguage } from "../types/api";

export const SettingsModal: React.FC = () => {
  const { showSettings, setShowSettings, language, setLanguage, backendOnline } =
    useUIStore();

  if (!showSettings) return null;

  return (
    <div className="settings-overlay" onClick={() => setShowSettings(false)}>
      <div
        className="settings-panel"
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "var(--space-6)",
          }}
        >
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, margin: 0 }}>
            System Settings
          </h2>
          <button
            className="btn btn-secondary"
            style={{ padding: "4px 10px", fontSize: "0.85rem" }}
            onClick={() => setShowSettings(false)}
          >
            ✕ Close
          </button>
        </div>

        <div className="settings-row">
          <div>
            <div className="settings-label">Target Translation Language</div>
            <div className="settings-desc">
              Preferred spoken Indian language for TTS and dialogue responses
            </div>
          </div>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value as AppLanguage)}
            style={{
              padding: "6px 10px",
              borderRadius: "var(--radius-sm)",
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              color: "var(--text-primary)",
              fontSize: "0.85rem",
            }}
          >
            {SUPPORTED_LANGUAGES.map((lang) => (
              <option key={lang} value={lang}>
                {lang}
              </option>
            ))}
          </select>
        </div>

        <div className="settings-row">
          <div>
            <div className="settings-label">Backend API URL</div>
            <div className="settings-desc">
              Target FastAPI server endpoint
            </div>
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.8rem",
              color: "var(--accent-primary)",
              background: "rgba(99, 102, 241, 0.1)",
              padding: "4px 8px",
              borderRadius: "var(--radius-sm)",
            }}
          >
            {import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}
          </div>
        </div>

        <div className="settings-row">
          <div>
            <div className="settings-label">Backend Connection Status</div>
            <div className="settings-desc">
              Real-time health check response
            </div>
          </div>
          <span
            className={`badge ${
              backendOnline === true
                ? "badge-success"
                : backendOnline === false
                ? "badge-danger"
                : "badge-warning"
            }`}
          >
            {backendOnline === true
              ? "ONLINE (FastAPI)"
              : backendOnline === false
              ? "DISCONNECTED"
              : "CONNECTING..."}
          </span>
        </div>

        <div className="settings-row">
          <div>
            <div className="settings-label">Recognition Engine</div>
            <div className="settings-desc">
              MediaPipe holistic + BiLSTM architecture
            </div>
          </div>
          <span className="badge badge-accent">ISLBiLSTM (126-dim)</span>
        </div>

        <div className="settings-row">
          <div>
            <div className="settings-label">Dialogue Decision Engine</div>
            <div className="settings-desc">
              Expected Information Gain (EIG) with Shannon Entropy
            </div>
          </div>
          <span className="badge badge-primary">EIG Active</span>
        </div>

        <div style={{ marginTop: "var(--space-6)", textAlign: "right" }}>
          <button
            className="btn btn-primary"
            onClick={() => setShowSettings(false)}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
