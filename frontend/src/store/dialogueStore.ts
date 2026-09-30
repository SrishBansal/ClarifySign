// ClarifySign Frontend - Zustand Dialogue Store

import { create } from "zustand";
import type {
  DialogueStartResponse,
  DialogueClarifyResponse,
  ConversationTurn,
} from "../types/api";

interface DialogueState {
  dialogueId: string | null;
  currentAction: "idle" | "commit" | "clarify" | "resolved";
  pendingClarification: DialogueStartResponse["clarification"] | null;
  confirmedClass: string | null;
  translations: Record<string, string> | null;
  conversationHistory: ConversationTurn[];
  isProcessing: boolean;
  error: string | null;

  setDialogueStart: (resp: DialogueStartResponse) => void;
  setDialogueClarified: (resp: DialogueClarifyResponse) => void;
  addTurn: (turn: ConversationTurn) => void;
  setIsProcessing: (v: boolean) => void;
  setError: (msg: string | null) => void;
  reset: () => void;
}

export const useDialogueStore = create<DialogueState>()((set) => ({
  dialogueId: null,
  currentAction: "idle",
  pendingClarification: null,
  confirmedClass: null,
  translations: null,
  conversationHistory: [],
  isProcessing: false,
  error: null,

  setDialogueStart: (resp) =>
    set({
      dialogueId: resp.dialogue_id,
      currentAction: resp.action,
      pendingClarification: resp.clarification,
      confirmedClass: resp.confirmed_class,
      translations: resp.translation,
    }),

  setDialogueClarified: (resp) =>
    set({
      currentAction: "resolved",
      confirmedClass: resp.confirmed_class,
      translations: resp.translation,
      pendingClarification: null,
    }),

  addTurn: (turn) =>
    set((s) => ({
      conversationHistory: [turn, ...s.conversationHistory].slice(0, 50),
    })),

  setIsProcessing: (v) => set({ isProcessing: v }),
  setError: (msg) => set({ error: msg }),
  reset: () =>
    set({
      dialogueId: null,
      currentAction: "idle",
      pendingClarification: null,
      confirmedClass: null,
      translations: null,
      isProcessing: false,
      error: null,
    }),
}));
