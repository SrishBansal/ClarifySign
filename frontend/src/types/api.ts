// ClarifySign Frontend - TypeScript API Types

export interface PredictionItem {
  label: string;
  probability: number;
}

export interface UncertaintyMetrics {
  top1_confidence: number;
  margin: number;
  entropy_bits: number;
  normalized_entropy: number;
  is_ambiguous: boolean;
  primary_reason: string;
  triggered_criteria: string[];
  decision: "commit" | "clarify";
}

export interface PredictResponse {
  top_prediction: PredictionItem;
  predictions: PredictionItem[];
  uncertainty: UncertaintyMetrics;
  inference_ms: number;
}

export interface ClarificationQuestion {
  text: string;
  options: string[];
  candidate_keys: string[];
  question_type: string;
  information_gain: number;
  utility: number;
}

export interface DialogueStartResponse {
  dialogue_id: string;
  action: "commit" | "clarify";
  confirmed_class: string | null;
  translation: Record<string, string> | null;
  clarification: ClarificationQuestion | null;
  message: string;
}

export interface DialogueClarifyResponse {
  dialogue_id: string;
  confirmed_class: string;
  translation: Record<string, string>;
  was_clarified: boolean;
  message: string;
}

export interface TranslateResponse {
  intent: string;
  translations: Record<string, string>;
  sources: Record<string, string>;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  timestamp: string;
}

export type AppLanguage =
  | "Hindi"
  | "Marathi"
  | "Bengali"
  | "Gujarati"
  | "Tamil"
  | "Telugu"
  | "Kannada"
  | "Malayalam"
  | "Punjabi"
  | "Odia";

export const SUPPORTED_LANGUAGES: AppLanguage[] = [
  "Hindi",
  "Marathi",
  "Bengali",
  "Gujarati",
  "Tamil",
  "Telugu",
  "Kannada",
  "Malayalam",
  "Punjabi",
  "Odia",
];

export interface ConversationTurn {
  id: string;
  type: "prediction" | "clarification" | "resolved";
  timestamp: number;
  label?: string;
  confidence?: number;
  question?: string;
  answer?: string;
  translations?: Record<string, string>;
  wasAmbiguous?: boolean;
}
