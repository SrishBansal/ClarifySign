// ClarifySign - ClarificationCard Component
import React from "react";
import type { ClarificationQuestion } from "../types/api";

interface ClarificationCardProps {
  question: ClarificationQuestion;
  onSelect: (option: string, index: number) => void;
  isLoading?: boolean;
}

export const ClarificationCard: React.FC<ClarificationCardProps> = ({
  question,
  onSelect,
  isLoading = false,
}) => {
  return (
    <div
      className="clarification-card"
      role="dialog"
      aria-label="Clarification needed"
      aria-modal="false"
    >
      <div className="clarification-header">
        <div className="clarification-icon" aria-hidden="true">❓</div>
        <div>
          <span
            className="badge badge-warning"
            style={{ marginBottom: 6 }}
          >
            Clarification Needed
          </span>
          <p className="clarification-question">{question.text}</p>
        </div>
      </div>

      <div className="clarification-options" role="group" aria-label="Clarification options">
        {question.options.map((option, idx) => (
          <button
            key={`clarification-option-${idx}`}
            id={`clarification-option-${idx}`}
            className="clarification-option"
            onClick={() => onSelect(option, idx + 1)}
            disabled={isLoading}
            aria-label={`Option ${idx + 1}: ${option}`}
          >
            <span className="clarification-option-num" aria-hidden="true">
              {idx + 1}
            </span>
            {option}
          </button>
        ))}
      </div>

      <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
        <span className="badge badge-muted" style={{ fontSize: "0.65rem" }}>
          EIG: {question.information_gain.toFixed(3)}
        </span>
        <span className="badge badge-muted" style={{ fontSize: "0.65rem" }}>
          U: {question.utility.toFixed(3)}
        </span>
        <span className="badge badge-muted" style={{ fontSize: "0.65rem" }}>
          {question.question_type}
        </span>
      </div>
    </div>
  );
};
