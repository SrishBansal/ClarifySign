"""
ClarifySign - Multilingual Speech Synthesis Module
Provides both client-side browser speech synthesis (Web Speech API)
and server-side audio generation (gTTS) for the 10 Indian languages.
"""

import io
from typing import Optional

# Language BCP-47 tags for Web Speech API and gTTS
SPEECH_LANG_CODES = {
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Odia": "or",  # Note: gTTS may fall back to Hindi or English for Odia
    "English": "en"
}

WEB_SPEECH_VOICE_TAGS = {
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Gujarati": "gu-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Punjabi": "pa-IN",
    "Odia": "or-IN",
    "English": "en-IN"
}


def get_browser_speech_html(text: str, language: str = "English", button_label: str = "🔊 Speak Output") -> str:
    """
    Generates a self-contained HTML/JS button utilizing the browser's native
    window.speechSynthesis Web Speech API. Completely client-side, zero server latency.
    """
    clean_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    voice_tag = WEB_SPEECH_VOICE_TAGS.get(language, "en-IN")

    html = f"""
    <div style="margin-top: 10px; margin-bottom: 10px;">
        <button id="tts_speak_btn" onclick="speakText()" style="
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: white;
            border: none;
            padding: 8px 18px;
            font-size: 14px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        ">
            {button_label}
        </button>
        <span id="tts_status" style="font-size: 12px; color: #64748B; margin-left: 10px;"></span>
        <script>
            function speakText() {{
                if (!('speechSynthesis' in window)) {{
                    document.getElementById('tts_status').innerText = 'Browser TTS not supported';
                    return;
                }}
                window.speechSynthesis.cancel();
                const utter = new SpeechSynthesisUtterance("{clean_text}");
                utter.lang = "{voice_tag}";
                utter.rate = 0.95;
                utter.pitch = 1.0;

                const voices = window.speechSynthesis.getVoices();
                const matchedVoice = voices.find(v => v.lang === "{voice_tag}" || v.lang.startsWith("{voice_tag.split('-')[0]}"));
                if (matchedVoice) utter.voice = matchedVoice;

                utter.onstart = function() {{
                    document.getElementById('tts_status').innerText = 'Speaking ({voice_tag})...';
                }};
                utter.onend = function() {{
                    document.getElementById('tts_status').innerText = '';
                }};
                utter.onerror = function(e) {{
                    document.getElementById('tts_status').innerText = 'Audio playback error';
                }};
                window.speechSynthesis.speak(utter);
            }}
        </script>
    </div>
    """
    return html


def synthesize_audio_bytes(text: str, language: str = "English") -> Optional[bytes]:
    """
    Synthesizes MP3 audio bytes using gTTS.
    Returns bytes or None if synthesis fails.
    """
    try:
        from gtts import gTTS
        lang_code = SPEECH_LANG_CODES.get(language, "en")
        try:
            tts = gTTS(text=text, lang=lang_code, slow=False)
        except Exception:
            # Fallback for languages without gTTS voice (e.g. Odia)
            tts = gTTS(text=text, lang="hi" if language == "Odia" else "en", slow=False)

        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        print(f"Warning: Audio synthesis failed ({e})")
        return None
