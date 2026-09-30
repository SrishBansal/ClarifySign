// ClarifySign Frontend - Zustand UI Store

import { create } from "zustand";
import type { AppLanguage } from "../types/api";

interface UIState {
  language: AppLanguage;
  showSettings: boolean;
  isCameraActive: boolean;
  isCapturing: boolean;
  backendOnline: boolean | null; // null = unknown
  setLanguage: (lang: AppLanguage) => void;
  setShowSettings: (v: boolean) => void;
  setIsCameraActive: (v: boolean) => void;
  setIsCapturing: (v: boolean) => void;
  setBackendOnline: (v: boolean | null) => void;
}

export const useUIStore = create<UIState>()((set) => ({
  language: "Hindi",
  showSettings: false,
  isCameraActive: false,
  isCapturing: false,
  backendOnline: null,
  setLanguage: (lang) => set({ language: lang }),
  setShowSettings: (v) => set({ showSettings: v }),
  setIsCameraActive: (v) => set({ isCameraActive: v }),
  setIsCapturing: (v) => set({ isCapturing: v }),
  setBackendOnline: (v) => set({ backendOnline: v }),
}));
