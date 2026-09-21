import html
import sys
from pathlib import Path

import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config

API_URL = config.API_URL
MODEL_TIMEOUT = 900
API_TIMEOUT = 120


def call_api(method, path, timeout, **kwargs):
    try:
        res = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.RequestException as e:
        return None, f"Could not reach the API ({type(e).__name__})."
    return res, None


def api_error(res, fallback):
    try:
        body = res.json()
    except ValueError:
        return fallback
    message = body.get("error", fallback)
    detail = body.get("detail")
    return f"{message} ({detail})" if detail else message


def play_speech(text):
    res, err = call_api("post", "/speak", API_TIMEOUT, params={"text": text})
    if err:
        st.error(err)
    elif res.status_code == 200:
        st.audio(res.content, format="audio/mp3")
    else:
        st.error(api_error(res, "Could not generate audio."))


st.set_page_config(page_title="What Happened Here?", layout="wide")

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #0B0C0E;
    --panel: #14161A;
    --panel-alt: #191C21;
    --border: #2A2E35;
    --text: #E9E6DF;
    --text-dim: #8B8F97;
    --accent: #FFB020;
    --accent-dim: #6B4E1C;
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

p, span, label, div {
    font-family: 'IBM Plex Sans', sans-serif;
}

.observer-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 28px;
}

.observer-header h1 {
    font-size: 1.6rem;
    margin: 0;
}

.status-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent);
}

.status-pill::before {
    content: "●";
    margin-right: 6px;
}

.panel {
    background-color: var(--panel);
    border: 1px solid var(--border);
    padding: 20px 22px;
    margin-bottom: 20px;
}

.panel-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text);
}

.monitor-frame {
    border: 1px solid var(--accent-dim);
    padding: 6px;
    background-color: #000;
}

.timecode-entry {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text);
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 14px;
}

.timecode-entry .t {
    color: var(--accent);
    min-width: 64px;
}

.console-response {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    color: var(--text);
    background-color: var(--panel-alt);
    border-left: 2px solid var(--accent);
    padding: 12px 16px;
    margin-top: 10px;
}

.console-response::before {
    content: "> ";
    color: var(--accent);
}

[data-testid="stFileUploader"] {
    background-color: var(--panel);
    border: 1px dashed var(--border);
}

.stButton > button {
    background-color: transparent;
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    padding: 6px 18px;
}

.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
    background-color: var(--panel-alt);
}

.stTextInput input {
    background-color: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
}

hr {
    border-color: var(--border) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None

header = st.empty()


def render_header(status):
    header.markdown(
        f"""
<div class="observer-header">
    <h1>What happened here</h1>
    <span class="status-pill">{status}</span>
</div>
""",
        unsafe_allow_html=True,
    )


render_header("ready")

st.markdown('<div class="panel-title">Upload a recording</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=["mp4"], label_visibility="collapsed")

col_upload, col_analyze = st.columns([1, 1])
with col_upload:
    if uploaded_file is not None and st.button("Upload"):
        st.session_state.result = None
        res, err = call_api(
            "post",
            "/upload",
            API_TIMEOUT,
            files={
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            },
        )
        if err:
            st.error(err)
        elif res.status_code == 200:
            st.success("Uploaded")
        else:
            st.error(api_error(res, "Upload failed"))

with col_analyze:
    if st.button("Analyze"):
        render_header("analyzing")
        with st.spinner("Watching the recording..."):
            res, err = call_api("post", "/analyze", MODEL_TIMEOUT)
        render_header("ready")
        if err:
            st.error(err)
        elif res.status_code == 200:
            st.session_state.result = res.json()
        else:
            st.error(api_error(res, "Analysis failed"))

if st.session_state.result:
    result = st.session_state.result

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="panel-title">Monitor</div>', unsafe_allow_html=True)
        if result.get("annotated_video_path"):
            st.markdown('<div class="monitor-frame">', unsafe_allow_html=True)
            video_res, video_err = call_api("get", "/annotated_video", API_TIMEOUT)
            if video_err:
                st.error(video_err)
            elif video_res.status_code == 200:
                st.video(video_res.content)
            else:
                st.error(api_error(video_res, "Could not load the annotated video."))
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="panel">No footage processed yet.</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            f"""
        <div class="panel">
            <div class="panel-title">Field report</div>
            <p style="color: var(--text); line-height: 1.5;">{html.escape(result['summary'])}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if result.get("summary_source") == "template":
            st.caption(
                "Scene description unavailable - this is a rule-based summary of the detected events."
            )

        if st.button("Hear report"):
            with st.spinner("Generating voice..."):
                play_speech(result["summary"])

    st.markdown('<div class="panel-title">Event log</div>', unsafe_allow_html=True)
    log_html = '<div class="panel">'
    for e in result["events"]:
        obj_part = f" {e['object']}" if e.get("object") else ""
        log_html += f'<div class="timecode-entry"><span class="t">{e["t"]:05.1f}s</span><span>{e["subject"]} {e["event"]}{obj_part}</span></div>'
    log_html += "</div>"
    st.markdown(log_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-title">Ask the observer</div>', unsafe_allow_html=True
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        question = st.text_input(
            " ",
            placeholder="Did anyone leave an object?",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_b:
        audio = mic_recorder(start_prompt="Speak", stop_prompt="Stop", key="recorder")

    if audio:
        with st.spinner("Transcribing..."):
            res, err = call_api(
                "post",
                "/transcribe",
                MODEL_TIMEOUT,
                files={"file": ("recording.wav", audio["bytes"], "audio/wav")},
            )
        if err:
            st.error(err)
        elif res.status_code == 200:
            question = res.json()["text"]
            st.markdown(
                f'<div class="console-response">heard: {html.escape(question)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error(api_error(res, "Could not transcribe the recording."))

    if st.button("Ask", key="ask_button") and question:
        with st.spinner("Thinking..."):
            res, err = call_api(
                "post", "/ask", MODEL_TIMEOUT, params={"question": question}
            )
        if err:
            st.error(err)
        elif res.status_code == 200:
            answer = res.json()["answer"]
            st.markdown(
                f'<div class="console-response">{html.escape(answer)}</div>',
                unsafe_allow_html=True,
            )
            play_speech(answer)
        else:
            st.error(api_error(res, "Question failed"))
