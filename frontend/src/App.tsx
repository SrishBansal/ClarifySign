// ClarifySign - Main Application Component
import React, { useEffect, useCallback, useState, useRef } from "react";
import { Header } from "./components/Header";
import { VideoCapture } from "./components/VideoCapture";
import { PredictionDisplay } from "./components/PredictionDisplay";
import { ClarificationCard } from "./components/ClarificationCard";
import { TranslationOutput } from "./components/TranslationOutput";
import { DialogueHistory } from "./components/DialogueHistory";
import { SettingsModal } from "./components/SettingsModal";
import { StatusBar } from "./components/StatusBar";

import { useUIStore } from "./store/uiStore";
import { useInferenceStore } from "./store/inferenceStore";
import { useDialogueStore } from "./store/dialogueStore";

import {
  getHealth,
  predictFrame,
  dialogueStart,
  dialogueClarify,
} from "./services/api";
import type { PredictResponse, ConversationTurn } from "./types/api";
import "./App.css";

export const App: React.FC = () => {
  const {
    language,
    backendOnline,
    setBackendOnline,
    isCapturing,
    setIsCapturing,
    setLanguage,
  } = useUIStore();

  const {
    lastPrediction,
    isLoading: isPredicting,
    setLastPrediction,
    setIsLoading: setIsPredicting,
    incrementFrameCount,
    reset: resetInference,
  } = useInferenceStore();

  const {
    dialogueId,
    pendingClarification,
    translations,
    confirmedClass,
    conversationHistory,
    setDialogueStart,
    setDialogueClarified,
    addTurn,
    reset: resetDialogue,
  } = useDialogueStore();

  const [isClarifying, setIsClarifying] = useState(false);
  const [inFlight, setInFlight] = useState(false);
  const lastEvaluationTime = useRef<number>(0);

  // ── Poll Backend Health ───────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      try {
        const res = await getHealth();
        if (mounted) {
          setBackendOnline(res.status === "ok");
        }
      } catch {
        if (mounted) {
          setBackendOnline(false);
        }
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [setBackendOnline]);

  // ── Handle Frame from VideoCapture ────────────────────────────────────────
  const handleFrame = useCallback(
    async (frameB64: string) => {
      if (inFlight || pendingClarification) {
        return; // Don't overwhelm network or interrupt pending clarification
      }

      setInFlight(true);
      setIsPredicting(true);
      try {
        const pred = await predictFrame(frameB64);
        setLastPrediction(pred);
        incrementFrameCount();

        const now = Date.now();
        // Evaluate dialogue if prediction has sufficient confidence or is ambiguous
        // Throttle dialogue evaluation to at most once every 1200ms
        const shouldEvaluate =
          now - lastEvaluationTime.current > 1200 &&
          (pred.uncertainty.decision === "clarify" ||
            pred.top_prediction.probability >= 0.40);

        if (shouldEvaluate && !pendingClarification) {
          lastEvaluationTime.current = now;
          const diag = await dialogueStart(pred, language);
          setDialogueStart(diag);

          if (diag.action === "clarify" && diag.clarification) {
            const turn: ConversationTurn = {
              id: `clarify-${now}`,
              type: "clarification",
              timestamp: now,
              label: pred.top_prediction.label,
              confidence: pred.top_prediction.probability,
              question: diag.clarification.text,
              wasAmbiguous: true,
            };
            addTurn(turn);
          } else if (diag.action === "commit" && diag.confirmed_class) {
            const turn: ConversationTurn = {
              id: `commit-${now}`,
              type: "prediction",
              timestamp: now,
              label: diag.confirmed_class,
              confidence: pred.top_prediction.probability,
              translations: diag.translation ?? undefined,
              wasAmbiguous: false,
            };
            addTurn(turn);
          }
        }
      } catch (err) {
        console.error("Frame prediction error:", err);
      } finally {
        setIsPredicting(false);
        setInFlight(false);
      }
    },
    [
      inFlight,
      pendingClarification,
      language,
      setLastPrediction,
      incrementFrameCount,
      setIsPredicting,
      setDialogueStart,
      addTurn,
    ]
  );

  // ── Handle Clarification Response Selection ──────────────────────────────
  const handleClarificationSelect = async (option: string) => {
    if (!dialogueId) return;

    setIsClarifying(true);
    try {
      const resp = await dialogueClarify(dialogueId, option);
      setDialogueClarified(resp);

      const turn: ConversationTurn = {
        id: `resolved-${Date.now()}`,
        type: "resolved",
        timestamp: Date.now(),
        label: resp.confirmed_class,
        answer: option,
        translations: resp.translation,
        wasAmbiguous: true,
      };
      addTurn(turn);
    } catch (err) {
      console.error("Clarification error:", err);
    } finally {
      setIsClarifying(false);
    }
  };

  // ── Manual Demo / Test Action ─────────────────────────────────────────────
  const handleSimulateDemo = async (type: "clear" | "ambiguous") => {
    // Generate synthetic realistic prediction for demonstration
    const isAmb = type === "ambiguous";
    const demoPred: PredictResponse = {
      top_prediction: {
        label: isAmb ? "help" : "bill",
        probability: isAmb ? 0.42 : 0.88,
      },
      predictions: isAmb
        ? [
            { label: "help", probability: 0.42 },
            { label: "hello", probability: 0.38 },
            { label: "welcome", probability: 0.12 },
          ]
        : [
            { label: "bill", probability: 0.88 },
            { label: "price", probability: 0.08 },
            { label: "total", probability: 0.04 },
          ],
      uncertainty: {
        top1_confidence: isAmb ? 0.42 : 0.88,
        margin: isAmb ? 0.04 : 0.80,
        entropy_bits: isAmb ? 1.62 : 0.45,
        normalized_entropy: isAmb ? 0.54 : 0.15,
        is_ambiguous: isAmb,
        primary_reason: isAmb ? "margin_below_threshold" : "none",
        triggered_criteria: isAmb ? ["margin < 0.15", "confidence < 0.78"] : [],
        decision: isAmb ? "clarify" : "commit",
      },
      inference_ms: 18.4,
    };

    setLastPrediction(demoPred);
    incrementFrameCount();

    try {
      const diag = await dialogueStart(demoPred, language);
      setDialogueStart(diag);

      const now = Date.now();
      if (diag.action === "clarify" && diag.clarification) {
        addTurn({
          id: `demo-clarify-${now}`,
          type: "clarification",
          timestamp: now,
          label: demoPred.top_prediction.label,
          confidence: demoPred.top_prediction.probability,
          question: diag.clarification.text,
          wasAmbiguous: true,
        });
      } else if (diag.action === "commit" && diag.confirmed_class) {
        addTurn({
          id: `demo-commit-${now}`,
          type: "prediction",
          timestamp: now,
          label: diag.confirmed_class,
          confidence: demoPred.top_prediction.probability,
          translations: diag.translation ?? undefined,
          wasAmbiguous: false,
        });
      }
    } catch (err) {
      console.error("Demo evaluation error:", err);
    }
  };

  const handleResetSession = () => {
    resetInference();
    resetDialogue();
  };

  return (
    <div className="layout">
      <Header backendOnline={backendOnline} />

      <main className="main-content">
        <div className="app-grid">
          {/* ── Left Column: Video Capture & Predictions ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
            <VideoCapture
              onFrame={handleFrame}
              isCapturing={isCapturing}
              onCameraReady={(ready) => {
                if (ready) setIsCapturing(true);
              }}
            />

            {/* Quick Demo Controls */}
            <div
              className="card"
              style={{
                padding: "var(--space-3) var(--space-4)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "var(--space-2)",
                flexWrap: "wrap",
              }}
            >
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Quick Test / Simulation:
              </span>
              <div style={{ display: "flex", gap: "var(--space-2)" }}>
                <button
                  className="btn btn-secondary"
                  style={{ fontSize: "0.75rem", padding: "4px 8px" }}
                  onClick={() => handleSimulateDemo("clear")}
                  title="Simulate a high-confidence sign (e.g., 'bill')"
                >
                  ⚡ Clear Sign (Commit)
                </button>
                <button
                  className="btn btn-secondary"
                  style={{ fontSize: "0.75rem", padding: "4px 8px" }}
                  onClick={() => handleSimulateDemo("ambiguous")}
                  title="Simulate an ambiguous sign (e.g., 'help' vs 'hello')"
                >
                  ⚡ Ambiguous (EIG Clarify)
                </button>
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: "0.75rem", padding: "4px 8px" }}
                  onClick={handleResetSession}
                >
                  🔄 Reset
                </button>
              </div>
            </div>

            <PredictionDisplay
              prediction={lastPrediction}
              isLoading={isPredicting}
            />
          </div>

          {/* ── Right Column: Dialogue, Translation & History ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-6)" }}>
            {/* Pending Clarification Card */}
            {pendingClarification && (
              <ClarificationCard
                question={pendingClarification}
                onSelect={handleClarificationSelect}
                isLoading={isClarifying}
              />
            )}

            {/* Translation Output */}
            {translations && (
              <TranslationOutput
                intent={confirmedClass}
                translations={translations}
                selectedLanguage={language}
                onLanguageSelect={setLanguage}
              />
            )}

            {/* If neither clarification nor translation yet, show helpful onboarding */}
            {!pendingClarification && !translations && (
              <div className="card placeholder-card">
                <span className="placeholder-icon">💬</span>
                <h3 style={{ color: "var(--text-primary)", fontWeight: 600 }}>
                  Interactive Dialogue Ready
                </h3>
                <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", maxWidth: 380 }}>
                  When Indian Sign Language gestures are recognized, ClarifySign uses
                  Expected Information Gain (EIG) to resolve uncertainty and translate
                  retail intents into 10 Indian spoken languages.
                </p>
              </div>
            )}

            {/* Conversation / Session History */}
            <DialogueHistory
              history={conversationHistory}
              onClear={handleResetSession}
            />
          </div>
        </div>
      </main>

      <SettingsModal />
      <StatusBar />
    </div>
  );
};

export default App;
