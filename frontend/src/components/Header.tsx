// ClarifySign - Header Component
import React from "react";
import { SUPPORTED_LANGUAGES, type AppLanguage } from "../types/api";
import { useUIStore } from "../store/uiStore";

interface HeaderProps {
  backendOnline: boolean | null;
}

export const Header: React.FC<HeaderProps> = ({ backendOnline }) => {
  const { language, setLanguage, setShowSettings } = useUIStore();

  return (
    <header className="header" role="banner">
      <div className="header-brand">
        <span className="header-logo" aria-hidden="true">🤟</span>
        <span className="header-title">ClarifySign</span>
        <span
          className={`badge ${
            backendOnline === true
              ? "badge-success"
              : backendOnline === false
              ? "badge-danger"
              : "badge-muted"
          }`}
          role="status"
          aria-live="polite"
        >
          <span className="dot-live" />
          {backendOnline === true
            ? "Online"
            : backendOnline === false
            ? "Offline"
            : "Connecting…"}
        </span>
      </div>

      <nav className="header-nav" role="navigation" aria-label="App controls">
        <label htmlFor="language-select" className="text-muted" style={{ fontSize: "0.8rem" }}>
          Lang:
        </label>
        <select
          id="language-select"
          value={language}
          onChange={(e) => setLanguage(e.target.value as AppLanguage)}
          aria-label="Select output language"
        >
          {SUPPORTED_LANGUAGES.map((lang) => (
            <option key={lang} value={lang}>
              {lang}
            </option>
          ))}
        </select>

        <button
          id="settings-btn"
          className="btn btn-ghost btn-sm"
          onClick={() => setShowSettings(true)}
          aria-label="Open settings"
        >
          ⚙️
        </button>
      </nav>
    </header>
  );
};
