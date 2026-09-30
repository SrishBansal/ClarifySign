// ClarifySign Frontend - Axios API Client

import axios, { type AxiosInstance, AxiosError } from "axios";
import type {
  PredictResponse,
  DialogueStartResponse,
  DialogueClarifyResponse,
  TranslateResponse,
  HealthResponse,
} from "../types/api";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// ── Response interceptor for structured error handling ────────────────────────
apiClient.interceptors.response.use(
  (res) => res,
  (error: AxiosError<{ detail: string; error_code: string }>) => {
    const detail = error.response?.data?.detail ?? error.message;
    const code = error.response?.data?.error_code ?? "NETWORK_ERROR";
    return Promise.reject(new ApiError(detail, code, error.response?.status));
  }
);

export class ApiError extends Error {
  readonly errorCode: string;
  readonly statusCode?: number;

  constructor(
    message: string,
    errorCode: string,
    statusCode?: number
  ) {
    super(message);
    this.name = "ApiError";
    this.errorCode = errorCode;
    this.statusCode = statusCode;
  }
}

// ── Inference ─────────────────────────────────────────────────────────────────
export async function predictFrame(
  frameB64: string,
  topK = 5
): Promise<PredictResponse> {
  const { data } = await apiClient.post<PredictResponse>("/api/predict", {
    frame_b64: frameB64,
    top_k: topK,
  });
  return data;
}

// ── Dialogue ──────────────────────────────────────────────────────────────────
export async function dialogueStart(
  prediction: PredictResponse,
  language: string
): Promise<DialogueStartResponse> {
  const { data } = await apiClient.post<DialogueStartResponse>("/api/dialogue/start", {
    prediction,
    language,
  });
  return data;
}

export async function dialogueClarify(
  dialogueId: string,
  customerResponse: string
): Promise<DialogueClarifyResponse> {
  const { data } = await apiClient.post<DialogueClarifyResponse>("/api/dialogue/clarify", {
    dialogue_id: dialogueId,
    customer_response: customerResponse,
  });
  return data;
}

// ── Translation ───────────────────────────────────────────────────────────────
export async function translateIntent(
  intent: string,
  language?: string
): Promise<TranslateResponse> {
  const { data } = await apiClient.post<TranslateResponse>("/api/translate", {
    intent,
    ...(language ? { language } : {}),
  });
  return data;
}

// ── Speech ────────────────────────────────────────────────────────────────────
export async function speakText(text: string, language: string): Promise<Blob> {
  const { data } = await apiClient.post(
    "/api/speak",
    { text, language },
    { responseType: "blob" }
  );
  return data;
}

// ── Health ────────────────────────────────────────────────────────────────────
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>("/health");
  return data;
}
