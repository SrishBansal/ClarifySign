// ClarifySign - PredictionDisplay Component
import React from "react";
import type { PredictResponse } from "../types/api";

interface PredictionDisplayProps {
  prediction: PredictResponse | null;
  isLoading: boolean;
}

function confidenceColor(p: number): string {
  if (p >= 0.78) return "var(--accent-success)";
  if (p >= 0.50) return "var(--accent-warning)";
  return "var(--accent-danger)";
}

export const PredictionDisplay: React.FC<PredictionDisplayProps> = ({
  prediction,
  isLoading,
}) => {
  if (!prediction && !isLoading) {
    return (
      <div className="card predictions-card placeholder-card">
        <span className="placeholder-icon" aria-hidden="true">🤟</span>
        <h3 style={{ color: "var(--text-secondary)", fontWeight: 500 }}>
          Predictions will appear here
        </h3>
        <p className="text-muted" style={{ fontSize: "0.8rem" }}>
          Start camera and begin capturing to see real-time predictions
        </p>
      </div>
    );
  }

  return (
    <div className="card predictions-card" role="region" aria-label="Prediction results" aria-live="polite">
      <h3>
        <span aria-hidden="true">🎯</span> Predictions
        {isLoading && <span className="spinner" style={{ marginLeft: 8 }} aria-label="Processing" />}
      </h3>

      {prediction ? (
        <>
          {prediction.predictions.slice(0, 3).map((item, idx) => (
            <div key={item.label} className="prediction-item">
              <div className="prediction-item-header">
                <span
                  className={`prediction-label ${idx === 0 ? "top" : ""}`}
                  style={idx === 0 ? { color: confidenceColor(item.probability) } : {}}
                >
                  {idx === 0 && "★ "}{item.label}
                </span>
                <span className="prediction-prob">
                  {(item.probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="confidence-bar-track" role="progressbar"
                aria-valuenow={Math.round(item.probability * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="confidence-bar-fill"
                  style={{
                    width: `${item.probability * 100}%`,
                    background:
                      idx === 0
                        ? `linear-gradient(90deg, ${confidenceColor(item.probability)}, ${confidenceColor(item.probability)}99)`
                        : "rgba(255,255,255,0.2)",
                  }}
                />
              </div>
            </div>
          ))}

          {/* Uncertainty metrics */}
          <UncertaintyDisplay uncertainty={prediction.uncertainty} />

          <p className="text-muted" style={{ fontSize: "0.7rem", marginTop: 8 }}>
            ⚡ {prediction.inference_ms.toFixed(0)}ms
          </p>
        </>
      ) : (
        <div>
          {[1, 2, 3].map((i) => (
            <div key={i} className="prediction-item">
              <div className="skeleton" style={{ height: 16, marginBottom: 6, width: "60%" }} />
              <div className="skeleton" style={{ height: 6 }} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ── UncertaintyDisplay sub-component ─────────────────────────────────────────
interface UncertaintyDisplayProps {
  uncertainty: PredictResponse["uncertainty"];
}

export const UncertaintyDisplay: React.FC<UncertaintyDisplayProps> = ({
  uncertainty,
}) => {
  return (
    <div className="uncertainty-card" aria-label="Uncertainty metrics">
      <div className="flex items-center justify-between">
        <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)" }}>
          Uncertainty
        </span>
        <span
          className={`badge ${uncertainty.is_ambiguous ? "badge-warning" : "badge-success"}`}
        >
          {uncertainty.decision === "clarify" ? "⚠ Clarify" : "✓ Commit"}
        </span>
      </div>

      <div className="uncertainty-grid">
        <div className="uncertainty-metric">
          <span
            className="uncertainty-value"
            style={{ color: confidenceColor(uncertainty.top1_confidence) }}
          >
            {(uncertainty.top1_confidence * 100).toFixed(0)}%
          </span>
          <span className="uncertainty-label">Confidence</span>
        </div>

        <div className="uncertainty-metric">
          <span
            className="uncertainty-value"
            style={{ color: uncertainty.margin < 0.20 ? "var(--accent-warning)" : "var(--text-primary)" }}
          >
            {uncertainty.margin.toFixed(2)}
          </span>
          <span className="uncertainty-label">Margin</span>
        </div>

        <div className="uncertainty-metric">
          <span
            className="uncertainty-value"
            style={{ color: uncertainty.entropy_bits > 1.25 ? "var(--accent-warning)" : "var(--text-primary)" }}
          >
            {uncertainty.entropy_bits.toFixed(2)}
          </span>
          <span className="uncertainty-label">Entropy (bits)</span>
        </div>
      </div>
    </div>
  );
};
