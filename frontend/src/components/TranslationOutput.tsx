// ClarifySign - TranslationOutput Component
import React, { useState } from "react";
import { speakWithBrowserAPI } from "../services/speech";
import type { AppLanguage } from "../types/api";

interface TranslationOutputProps {
  intent: string | null;
  translations: Record<string, string> | null;
  selectedLanguage: AppLanguage;
  onLanguageSelect?: (lang: AppLanguage) => void;
}

export const TranslationOutput: React.FC<TranslationOutputProps> = ({
  intent,
  translations,
  selectedLanguage,
  onLanguageSelect,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);

  if (!intent || !translations) {
    return null;
  }

  const activeTranslation =
    translations[selectedLanguage] ||
    translations["Hindi"] ||
    translations["English"] ||
    Object.values(translations)[0] ||
    "";

  const handleSpeak = async (text: string, lang: string) => {
    if (!text || isPlaying) return;
    try {
      setIsPlaying(true);
      await speakWithBrowserAPI(text, lang);
    } catch (err) {
      console.error("Speech playback error:", err);
    } finally {
      setIsPlaying(false);
    }
  };

  return (
    <div className="card translation-card">
      <div className="translation-header">
        <div>
          <span className="badge badge-accent" style={{ marginBottom: "6px" }}>
            TRANSLATED INTENT
          </span>
          <div className="translation-intent">
            <span>🎯</span>
            <span>{intent}</span>
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={() => handleSpeak(activeTranslation, selectedLanguage)}
          disabled={isPlaying || !activeTranslation}
          title={`Speak in ${selectedLanguage}`}
        >
          {isPlaying ? "🔊 Speaking..." : `🔊 Speak (${selectedLanguage})`}
        </button>
      </div>

      <div className="translation-grid">
        {Object.entries(translations).map(([lang, text]) => {
          const isActive = lang.toLowerCase() === selectedLanguage.toLowerCase();
          return (
            <div
              key={lang}
              className={`translation-item ${isActive ? "active-lang" : ""}`}
              onClick={() => {
                if (onLanguageSelect) {
                  onLanguageSelect(lang as AppLanguage);
                }
                handleSpeak(text, lang);
              }}
              title="Click to select language and speak"
            >
              <div className="translation-lang">
                {lang} {isActive && "• active"}
              </div>
              <div className="translation-text">{text}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
