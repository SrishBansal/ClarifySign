// ClarifySign Frontend - Zustand Inference Store

import { create } from "zustand";
import type { PredictResponse } from "../types/api";

interface InferenceState {
  lastPrediction: PredictResponse | null;
  isLoading: boolean;
  error: string | null;
  frameCount: number;
  setLastPrediction: (p: PredictResponse | null) => void;
  setIsLoading: (v: boolean) => void;
  setError: (msg: string | null) => void;
  incrementFrameCount: () => void;
  reset: () => void;
}

export const useInferenceStore = create<InferenceState>()((set) => ({
  lastPrediction: null,
  isLoading: false,
  error: null,
  frameCount: 0,
  setLastPrediction: (p) => set({ lastPrediction: p }),
  setIsLoading: (v) => set({ isLoading: v }),
  setError: (msg) => set({ error: msg }),
  incrementFrameCount: () => set((s) => ({ frameCount: s.frameCount + 1 })),
  reset: () =>
    set({ lastPrediction: null, isLoading: false, error: null, frameCount: 0 }),
}));
