// ClarifySign Frontend - Web Speech API wrapper

const VOICE_TAGS: Record<string, string> = {
  Hindi: "hi-IN",
  Marathi: "mr-IN",
  Bengali: "bn-IN",
  Gujarati: "gu-IN",
  Tamil: "ta-IN",
  Telugu: "te-IN",
  Kannada: "kn-IN",
  Malayalam: "ml-IN",
  Punjabi: "pa-IN",
  Odia: "or-IN",
  English: "en-IN",
};

function loadVoices(): Promise<SpeechSynthesisVoice[]> {
  return new Promise((resolve) => {
    if (!("speechSynthesis" in window)) {
      resolve([]);
      return;
    }
    const voices = window.speechSynthesis.getVoices();
    if (voices.length > 0) {
      resolve(voices);
      return;
    }
    window.speechSynthesis.onvoiceschanged = () => {
      resolve(window.speechSynthesis.getVoices());
    };
  });
}

export async function speakWithBrowserAPI(
  text: string,
  language: string
): Promise<void> {
  if (!("speechSynthesis" in window)) {
    console.warn("Web Speech API not supported");
    return;
  }

  window.speechSynthesis.cancel();

  const voices = await loadVoices();
  const tag = VOICE_TAGS[language] ?? "en-IN";
  const lang2 = tag.split("-")[0];

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = tag;
  utter.rate = 0.95;
  utter.pitch = 1.0;

  const matched =
    voices.find((v) => v.lang === tag) ??
    voices.find((v) => v.lang.startsWith(lang2));
  if (matched) utter.voice = matched;

  return new Promise((resolve) => {
    utter.onend = () => resolve();
    utter.onerror = (e) => {
      console.warn("TTS error:", e.error);
      resolve(); // don't reject—best effort
    };
    window.speechSynthesis.speak(utter);
  });
}

export function cancelSpeech(): void {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

export function isSpeechSupported(): boolean {
  return "speechSynthesis" in window;
}
