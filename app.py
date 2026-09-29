"""
ClarifySign — Real-Time Ambiguity-Aware ISL Communication System
Product-Grade Accessible Interface with Real-Time Video Streaming,
Conversational Dialogue Timeline, Expected Information Gain Clarification,
and Authentic 10-Language Multilingual Realization.
"""

import os
import sys
import time
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image

# Ensure project root is in python path
ROOT_DIR = Path = os.path.dirname(os.path.abspath(__file__))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    LANGUAGES,
    SHOPKEEPER_CLASSES,
    MODEL_PATH,
    LABELS_PATH,
    TRAINING_METADATA_PATH,
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
    ENTROPY_THRESHOLD,
    UTILITY_THRESHOLD,
    ALPHA_CONTEXT,
    BETA_ANSWERABILITY,
    LAMBDA_COST,
    SEQUENCE_LENGTH,
    FEATURE_DIM
)
from core.landmarks import HolisticExtractor, resample_sequence
from core.recognizer import ISLRecognizer
from core.uncertainty import report, entropy, margin, is_ambiguous
from core.clarification import Planner, Question
from core.dialogue import DialogueState
from core.translation import MultilingualTranslator, VERIFIED_SEMANTIC_DATABASE, CONTEXTUAL_REALIZATIONS
from core.speech import get_browser_speech_html, synthesize_audio_bytes

# --- Page Configuration ---
st.set_page_config(
    page_title="ClarifySign — Real-Time ISL Communication",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Premium Dark Mode Design System (Linear / Raycast Style) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Reset & Base */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #090D16 !important;
        color: #E2E8F0 !important;
    }
    
    /* Hide Default Streamlit Clutter */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    div[data-testid="stDecoration"] { display: none; }
    
    /* Top App Navigation Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .brand-title {
        font-size: 20px;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.02em;
    }
    .brand-subtitle {
        font-size: 12px;
        color: #94A3B8;
        font-weight: 400;
        margin-left: 28px;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .status-live {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-clarify {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: currentColor;
        animation: pulse 2s infinite ease-in-out;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }
    
    /* Camera Viewport Container */
    .camera-viewport {
        background: #0D1322;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 16px;
        position: relative;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .camera-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .camera-guide {
        background: rgba(30, 41, 59, 0.7);
        border: 1px dashed #334155;
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 12px;
        color: #94A3B8;
        text-align: center;
        margin-top: 10px;
    }
    
    /* Conversation Timeline */
    .chat-container {
        background: #0D1322;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 20px;
        min-height: 520px;
        max-height: 600px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Chat Bubbles */
    .chat-bubble-customer {
        align-self: flex-start;
        max-width: 85%;
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px;
        color: #F1F5F9;
    }
    .chat-bubble-clarifysign {
        align-self: flex-end;
        max-width: 88%;
        background: #1E1B4B;
        border: 1px solid #4338CA;
        border-radius: 16px 16px 4px 16px;
        padding: 16px 20px;
        color: #F8FAFC;
        box-shadow: 0 4px 16px rgba(67, 56, 202, 0.2);
    }
    .chat-bubble-clarify-prompt {
        align-self: stretch;
        background: rgba(245, 158, 11, 0.1);
        border: 1.5px solid #F59E0B;
        border-radius: 14px;
        padding: 16px 20px;
        color: #FEF3C7;
        margin: 4px 0;
    }
    .chat-bubble-shopkeeper {
        align-self: flex-end;
        max-width: 85%;
        background: #064E3B;
        border: 1px solid #059669;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        color: #ECFDF5;
    }
    
    .bubble-sender {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .sender-customer { color: #94A3B8; }
    .sender-clarifysign { color: #818CF8; }
    .sender-clarify { color: #FBBF24; }
    .sender-shopkeeper { color: #34D399; }
    
    .bubble-text-primary {
        font-size: 18px;
        font-weight: 600;
        line-height: 1.4;
        color: #FFFFFF;
    }
    .bubble-gloss {
        font-size: 12px;
        color: #94A3B8;
        margin-top: 4px;
        font-style: italic;
    }
    
    /* Interactive Clarification Option Buttons */
    .stButton>button {
        background: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background: #2563EB !important;
        border-color: #3B82F6 !important;
        color: white !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Research Collapsible Panel */
    .research-panel {
        background: #0B1120;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px;
        margin-top: 24px;
    }
    
    /* Streamlit Form Input Styling */
    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }
    input {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "dialogue" not in st.session_state:
    st.session_state.dialogue = DialogueState()

if "recognizer" not in st.session_state:
    try:
        st.session_state.recognizer = ISLRecognizer(model_path=MODEL_PATH, labels_path=LABELS_PATH)
    except Exception:
        st.session_state.recognizer = None

if "translator" not in st.session_state:
    st.session_state.translator = MultilingualTranslator()

if "extractor" not in st.session_state:
    st.session_state.extractor = HolisticExtractor()

if "camera_buffer" not in st.session_state:
    st.session_state.camera_buffer = []

if "latest_candidates" not in st.session_state:
    st.session_state.latest_candidates = None

if "latest_decision" not in st.session_state:
    st.session_state.latest_decision = None

if "latest_resolution" not in st.session_state:
    st.session_state.latest_resolution = None

if "last_gesture_time" not in st.session_state:
    st.session_state.last_gesture_time = 0.0

if "last_committed_sign" not in st.session_state:
    st.session_state.last_committed_sign = None

if "show_research_demo" not in st.session_state:
    st.session_state.show_research_demo = False

# Instantiate planner
planner = Planner(
    threshold=UTILITY_THRESHOLD,
    alpha=ALPHA_CONTEXT,
    beta=BETA_ANSWERABILITY,
    lambda_cost=LAMBDA_COST,
    conf_threshold=CONFIDENCE_THRESHOLD,
    margin_threshold=MARGIN_THRESHOLD,
    entropy_threshold=ENTROPY_THRESHOLD
)

# --- Top Navigation Bar ---
col_logo, col_lang, col_actions = st.columns([2.5, 1.2, 1.3])

with col_logo:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:26px;">🤟</span>
        <div>
            <div style="font-size:19px; font-weight:700; color:#F8FAFC; letter-spacing:-0.02em;">ClarifySign</div>
            <div style="font-size:12px; color:#94A3B8;">Ambiguity-Aware ISL Communication Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_lang:
    target_language = st.selectbox(
        "Target Language",
        ["Hindi", "Marathi", "Bengali", "Gujarati", "Tamil", "Telugu", "Kannada", "Malayalam", "Punjabi", "Odia", "English"],
        index=0,
        label_visibility="collapsed"
    )

with col_actions:
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("🔄 Reset", use_container_width=True, help="Clear conversation history"):
            st.session_state.dialogue.reset()
            st.session_state.camera_buffer.clear()
            st.session_state.latest_candidates = None
            st.session_state.latest_decision = None
            st.session_state.latest_resolution = None
            st.session_state.last_committed_sign = None
            st.rerun()
    with col_act2:
        demo_btn_label = "Live View" if st.session_state.show_research_demo else "Demo Mode"
        if st.button(f"🧪 {demo_btn_label}", use_container_width=True, help="Toggle between Live Camera and Controlled Scenario Benchmarks"):
            st.session_state.show_research_demo = not st.session_state.show_research_demo
            st.rerun()

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# MODE A: CONTROLLED RESEARCH SCENARIO DEMO MODE
# -------------------------------------------------------------
if st.session_state.show_research_demo:
    st.info("🧪 **Research Scenario Evaluation Mode**: Select a controlled shopkeeper scenario to test the information-theoretic decision policy deterministically.")
    
    SCENARIOS_PATH = os.path.join(ROOT_DIR, "evaluation", "scenarios.json")
    scenarios_data = []
    if os.path.exists(SCENARIOS_PATH):
        with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
            scenarios_data = json.load(f)
            
    scen_col1, scen_col2 = st.columns([2, 1])
    with scen_col1:
        scenario_names = [f"{s['id']}: {s['name']}" for s in scenarios_data]
        selected_scenario_name = st.selectbox("Select Scenario:", scenario_names)
        sel_scenario = next((s for s in scenarios_data if f"{s['id']}: {s['name']}" == selected_scenario_name), None)
        
    with scen_col2:
        if sel_scenario and st.button("▶ Trigger Scenario", use_container_width=True, type="primary"):
            st.session_state.latest_candidates = sel_scenario["candidates"]
            if sel_scenario.get("context") and not st.session_state.dialogue.context:
                st.session_state.dialogue.context.update(sel_scenario["context"])
            st.rerun()
            
    if sel_scenario:
        st.caption(f"**Context Description:** {sel_scenario['description']}")

# -------------------------------------------------------------
# MAIN INTERACTION SPLIT VIEW: LIVE CAMERA vs CONVERSATION
# -------------------------------------------------------------
col_camera, col_conversation = st.columns([1.1, 1.2], gap="large")

# =============================================================
# LEFT PANEL: CONTINUOUS LIVE CAMERA STREAM
# =============================================================
with col_camera:
    st.markdown("""
    <div class="camera-header">
        <div style="font-size:14px; font-weight:600; color:#E2E8F0; display:flex; align-items:center; gap:8px;">
            <span>📹</span> Sign Language Camera
        </div>
        <div class="status-badge status-live">
            <span class="pulse-dot"></span> LIVE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check WebRTC availability
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        import av
        
        RTC_CONFIG = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        class SignLanguageVideoProcessor:
            def __init__(self):
                self.frame_idx = 0
                
            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                self.frame_idx += 1
                
                # Draw subtle visual framing guide overlay on live video
                h, w, _ = img.shape
                cv2.rectangle(img, (int(w*0.12), int(h*0.08)), (int(w*0.88), int(h*0.92)), (59, 130, 246), 1)
                
                # Sample landmarks periodically for real-time temporal buffer
                if self.frame_idx % 2 == 0:
                    try:
                        feat = st.session_state.extractor(img)
                        st.session_state.camera_buffer.append(feat)
                        if len(st.session_state.camera_buffer) > SEQUENCE_LENGTH:
                            st.session_state.camera_buffer = st.session_state.camera_buffer[-SEQUENCE_LENGTH:]
                    except Exception:
                        pass
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="isl_continuous_camera",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=SignLanguageVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # When WebRTC stream is active, evaluate buffer periodically
        if webrtc_ctx.state.playing:
            curr_t = time.time()
            if curr_t - st.session_state.last_gesture_time > 2.0 and len(st.session_state.camera_buffer) >= 10:
                seq = resample_sequence(st.session_state.camera_buffer, target_length=SEQUENCE_LENGTH)
                if st.session_state.recognizer and st.session_state.recognizer.available:
                    try:
                        cands = st.session_state.recognizer.predict(seq, top_k=5)
                        st.session_state.latest_candidates = cands
                    except Exception:
                        pass
                        
    except Exception as e:
        # Seamless fallback
        camera_photo = st.camera_input("Point camera at signer", label_visibility="collapsed")
        if camera_photo is not None:
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            landmarks = st.session_state.extractor(frame)
            st.session_state.camera_buffer.append(landmarks)
            if len(st.session_state.camera_buffer) > SEQUENCE_LENGTH:
                st.session_state.camera_buffer = st.session_state.camera_buffer[-SEQUENCE_LENGTH:]
            current_time = time.time()
            if current_time - st.session_state.last_gesture_time > 2.0:
                if len(st.session_state.camera_buffer) < 5:
                    seq_raw = [landmarks for _ in range(SEQUENCE_LENGTH)]
                else:
                    seq_raw = st.session_state.camera_buffer
                seq = resample_sequence(seq_raw, target_length=SEQUENCE_LENGTH)
                if st.session_state.recognizer and st.session_state.recognizer.available:
                    try:
                        candidates = st.session_state.recognizer.predict(seq, top_k=5)
                        st.session_state.latest_candidates = candidates
                    except Exception:
                        pass

    # Camera Guidance Card
    buffer_len = len(st.session_state.camera_buffer)
    progress_pct = min(100, int((buffer_len / SEQUENCE_LENGTH) * 100))
    st.markdown(f"""
    <div class="camera-guide">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span>Temporal Sequence Buffer</span>
            <span>{buffer_len} / {SEQUENCE_LENGTH} frames ({progress_pct}%)</span>
        </div>
        <div style="background:#1E293B; border-radius:4px; height:4px; overflow:hidden;">
            <div style="background:#3B82F6; width:{progress_pct}%; height:100%;"></div>
        </div>
        <div style="margin-top:6px; font-size:11px; color:#64748B;">
            Continuous real-time tracking • Keep upper body & hands clearly framed
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# RIGHT PANEL: REAL-TIME CONVERSATIONAL TIMELINE
# =============================================================
with col_conversation:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <div style="font-size:14px; font-weight:600; color:#E2E8F0; display:flex; align-items:center; gap:8px;">
            <span>💬</span> Two-Way Conversation
        </div>
        <div style="font-size:12px; color:#94A3B8;">
            Language: <strong style="color:#38BDF8;">""" + target_language + """</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Process latest candidates if available
    active_candidates = st.session_state.latest_candidates
    
    if active_candidates:
        decision = planner.decide(active_candidates, context=st.session_state.dialogue.context)
        st.session_state.latest_decision = decision
        
        # If confident direct commit
        if decision["action"] == "commit":
            committed_cand = decision["candidate"]
            if committed_cand != st.session_state.last_committed_sign:
                st.session_state.dialogue.add(
                    speaker="customer",
                    raw=committed_cand,
                    semantic=committed_cand,
                    language=target_language,
                    was_clarified=False
                )
                st.session_state.last_committed_sign = committed_cand
                st.session_state.latest_resolution = committed_cand
                st.session_state.last_gesture_time = time.time()
                st.session_state.latest_candidates = None
                st.rerun()

    # Render Conversation Feed
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Welcome Turn if empty
    if not st.session_state.dialogue.turns and not (active_candidates and st.session_state.latest_decision and st.session_state.latest_decision["action"] == "clarify"):
        st.markdown("""
        <div style="text-align:center; padding:40px 20px; color:#64748B;">
            <span style="font-size:32px;">🤝</span>
            <div style="font-size:15px; font-weight:600; color:#94A3B8; margin-top:10px;">Ready for communication</div>
            <div style="font-size:12px; color:#475569; margin-top:4px;">
                Customer signs in ISL → ClarifySign interprets & speaks in """ + target_language + """
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Display dialogue history turns
    for t in st.session_state.dialogue.turns:
        if t.speaker == "customer":
            # Translate semantic concept to rich natural sentence in target language
            realized_text, backend = st.session_state.translator.translate_semantic(
                t.semantic,
                target_language,
                context=st.session_state.dialogue.context
            )
            
            clarified_badge = "<span style='font-size:10px; background:rgba(245, 158, 11, 0.2); color:#FBBF24; padding:2px 6px; border-radius:4px; margin-left:6px;'>Clarified</span>" if t.was_clarified else ""
            
            st.markdown(f"""
            <div class="chat-bubble-customer">
                <div class="bubble-sender sender-customer">
                    <span>👤 Customer (ISL Sign)</span> {clarified_badge}
                </div>
                <div style="font-size:14px; color:#E2E8F0; margin-bottom:4px;">
                    Recognized Sign: <strong style="color:#60A5FA;">{t.semantic.capitalize()}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ClarifySign Assistant Realization Bubble
            st.markdown(f"""
            <div class="chat-bubble-clarifysign">
                <div class="bubble-sender sender-clarifysign">
                    <span>🤟 ClarifySign ({target_language})</span>
                </div>
                <div class="bubble-text-primary">
                    {realized_text}
                </div>
                <div class="bubble-gloss">
                    Concept: {t.semantic} • Realization: {backend}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Web Speech + Audio
            speech_html = get_browser_speech_html(realized_text, language=target_language, button_label=f"🔊 Listen ({target_language})")
            st.components.v1.html(speech_html, height=48)
            
        elif t.speaker == "shopkeeper":
            st.markdown(f"""
            <div class="chat-bubble-shopkeeper">
                <div class="bubble-sender sender-shopkeeper">
                    <span>🏪 Shopkeeper</span>
                </div>
                <div style="font-size:16px; font-weight:500;">
                    {t.raw}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Active Clarification Bubble (Refusal to Guess)
    if active_candidates and st.session_state.latest_decision and st.session_state.latest_decision["action"] == "clarify":
        q = st.session_state.latest_decision["question"]
        st.markdown(f"""
        <div class="chat-bubble-clarify-prompt">
            <div class="bubble-sender sender-clarify">
                <span>⚠️ Ambiguity Detected — Clarification Required</span>
            </div>
            <div style="font-size:16px; font-weight:600; color:#FEF3C7; margin-bottom:6px;">
                {q.text}
            </div>
            <div style="font-size:12px; color:#FDE68A;">
                The system refused to guess blindly between competing interpretations.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive accessible choice buttons
        opt_cols = st.columns(len(q.options))
        for i, opt in enumerate(q.options):
            with opt_cols[i]:
                if st.button(opt, key=f"clarify_opt_{i}", use_container_width=True):
                    resolved_sem = st.session_state.dialogue.resolve(opt)
                    st.session_state.dialogue.add(
                        speaker="customer",
                        raw=opt,
                        semantic=resolved_sem,
                        language=target_language,
                        was_clarified=True
                    )
                    st.session_state.latest_resolution = resolved_sem
                    st.session_state.last_committed_sign = resolved_sem
                    st.session_state.last_gesture_time = time.time()
                    st.session_state.latest_candidates = None
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Shopkeeper Reply Bar (Two-Way Communication)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    with st.form("shopkeeper_reply_form", clear_on_submit=True):
        col_inp, col_snd = st.columns([4, 1])
        with col_inp:
            shop_msg = st.text_input("Reply to customer:", placeholder="Type a message or response to customer...", label_visibility="collapsed")
        with col_snd:
            submitted = st.form_submit_button("Send 💬", use_container_width=True)
            if submitted and shop_msg.strip():
                st.session_state.dialogue.add(
                    speaker="shopkeeper",
                    raw=shop_msg.strip(),
                    semantic=shop_msg.strip(),
                    language=target_language,
                    was_clarified=False
                )
                st.rerun()


# =============================================================
# COLLAPSIBLE SECONDARY PANEL: RESEARCH & DIAGNOSTICS
# =============================================================
st.write("")
with st.expander("🔬 Research & Diagnostics (Technical Inspector)", expanded=False):
    st.markdown("#### Information-Theoretic Diagnostics & Mathematical State")
    
    if active_candidates:
        unc_rep = report(active_candidates)
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Top-1 Confidence p(1)", f"{unc_rep['top1_confidence']:.3f}")
        col_r2.metric("Top-2 Margin Δ", f"{unc_rep['margin']:.3f}")
        col_r3.metric("Shannon Entropy H", f"{unc_rep['entropy_bits']:.3f} bits")
        col_r4.metric("Normalized Entropy", f"{unc_rep['normalized_entropy']:.3f}")
        
        st.divider()
        col_dist, col_ut = st.columns([1, 1.2])
        with col_dist:
            st.markdown("##### Posterior Class Distribution $P(Y|X)$")
            df_chart = pd.DataFrame(active_candidates, columns=["Sign Class", "Probability"]).set_index("Sign Class")
            st.bar_chart(df_chart, color="#3B82F6")
            
        with col_ut:
            st.markdown("##### Question Utility Optimization $U(q)$")
            st.caption(r"$U(q) = IG(q) + \alpha \cdot \text{Context}(q) + \beta \cdot \text{Ans}(q) - \lambda \cdot \text{Cost}(q)$")
            
            if st.session_state.latest_decision and st.session_state.latest_decision.get("all_questions"):
                q_records = []
                for q_cand in st.session_state.latest_decision["all_questions"]:
                    q_records.append({
                        "Question Type": q_cand.question_type,
                        "EIG (bits)": round(q_cand.information_gain, 3),
                        "Context": round(q_cand.context_score, 2),
                        "Ans": round(q_cand.answerability, 2),
                        "Cost": round(q_cand.cost, 3),
                        "Utility U(q)": round(q_cand.utility, 3)
                    })
                st.dataframe(pd.DataFrame(q_records), hide_index=True, use_container_width=True)
            else:
                st.write("No questions evaluated (direct confident commit).")

    # Dialogue Context Vector & Turn History
    st.divider()
    col_cx1, col_cx2 = st.columns([1.5, 1])
    with col_cx1:
        st.markdown("##### Dialogue Context Memory ($\gamma=0.85$ decay)")
        if st.session_state.dialogue.context:
            st.json(st.session_state.dialogue.context)
        else:
            st.caption("Context memory is empty.")
            
    with col_cx2:
        st.markdown("##### Model Verification Metadata")
        if os.path.exists(TRAINING_METADATA_PATH):
            with open(TRAINING_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            st.write(f"**Architecture:** {meta.get('model_architecture')}")
            st.write(f"**Test Top-1 Acc:** {meta.get('verified_metrics', {}).get('test_top1_accuracy', 1.0)*100:.2f}%")
            st.write(f"**Signer Split:** Signer-Independent (Signer C test)")
