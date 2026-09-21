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

# Colour encodes which track an event belongs to, so the same subject keeps one
# hue in the timeline and the log. Chosen at similar lightness so no track reads
# as more important than another.
TRACK_COLORS = ["#58A6E8", "#5FC98C", "#C08BE8", "#E8A0C8", "#7FD6C0", "#9DA8F0"]
OBJECT_COLOR = "#F5A524"


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


def track_colors(events):
    registry = {}
    for e in events:
        subject = e["subject"]
        if subject not in registry:
            registry[subject] = TRACK_COLORS[len(registry) % len(TRACK_COLORS)]
    return registry


def timecode(seconds):
    return f"{int(seconds) // 60:02d}:{seconds % 60:04.1f}"


st.set_page_config(page_title="What Happened Here", layout="wide")

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #0B0D10;
    --chrome: #0E1115;
    --panel: #14171B;
    --raised: #1A1E23;
    --line: #272C33;
    --line-2: #353B44;
    --text: #ECEAE5;
    --text-2: #A6ACB6;
    --text-3: #858C96;
    --text-4: #6E747E;
    --accent: #F5A524;
    --accent-edge: #5A4312;
    --danger: #E8746A;
    --ok: #5FC98C;
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px; }

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: 8px; }

h1, h2, h3, p, span, label, div { font-family: 'IBM Plex Sans', sans-serif; }

.mono { font-family: 'IBM Plex Mono', monospace; }

/* ---------- top bar ---------- */

.topbar {
    display: flex;
    align-items: center;
    gap: 18px;
    background: var(--chrome);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 0 18px;
    height: 56px;
    margin-bottom: 22px;
}

.mark {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.16em;
    color: var(--text);
}

.mark-dot {
    display: inline-block;
    width: 16px; height: 16px;
    border: 1.5px solid var(--accent);
    border-radius: 3px;
    margin-right: 10px;
    vertical-align: -3px;
    position: relative;
}
.mark-dot::after {
    content: "";
    position: absolute;
    inset: 4px;
    background: var(--accent);
    border-radius: 1px;
}

.rule { width: 1px; height: 20px; background: var(--line); }

.status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.04em;
    display: flex; align-items: center; gap: 8px;
}
.status .led { width: 6px; height: 6px; border-radius: 50%; }

.meta {
    flex-grow: 1;
    display: flex; align-items: center; gap: 12px; justify-content: flex-end;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-3);
}
.meta .sep { color: var(--line-2); }
.meta .name { color: var(--text); font-size: 12px; }

/* ---------- panels ---------- */

.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 18px 20px;
    margin-bottom: 18px;
}

.section {
    display: flex; align-items: baseline; gap: 10px;
    margin: 4px 0 10px 0;
}
.section h2 {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin: 0 !important;
    letter-spacing: 0 !important;
}
.section .count {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-4);
}

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--text-3);
}

.badge {
    height: 20px; padding: 0 8px;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    display: inline-flex; align-items: center;
}
.badge-vlm { background: rgba(88,166,232,0.12); border: 1px solid rgba(88,166,232,0.40); color: #58A6E8; }
.badge-tpl { background: rgba(232,116,106,0.12); border: 1px solid rgba(232,116,106,0.40); color: var(--danger); }

/* ---------- field report ---------- */

.report {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 16.5px;
    line-height: 1.62;
    color: #E4E1DB;
    margin: 12px 0 0 0;
}

/* ---------- timeline ---------- */

.timeline { position: relative; height: 30px; margin: 2px 0 6px 0; }
.timeline .track {
    position: absolute; left: 0; right: 0; top: 12px;
    height: 5px; background: var(--line); border-radius: 3px;
}
.timeline .tick {
    position: absolute; top: 6px;
    width: 2px; height: 18px; border-radius: 1px;
}
.timeline-scale {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: var(--text-4);
}

/* ---------- event log ---------- */

.log-head, .log-row {
    display: grid;
    grid-template-columns: 86px 1.1fr 0.9fr 1.3fr;
    align-items: center;
    gap: 12px;
    padding: 0 14px;
}
.log-head {
    height: 30px;
    background: var(--raised);
    border-bottom: 1px solid var(--line);
}
.log-head span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 0.1em; color: var(--text-3);
}
.log-row { height: 32px; border-bottom: 1px solid #1E2227; }
.log-row:last-child { border-bottom: none; }
.log-row .t { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--accent); }
.log-row .subject { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--text); }
.log-row .chip { width: 7px; height: 7px; border-radius: 2px; flex-shrink: 0; }
.log-row .action { font-size: 12.5px; color: var(--text-2); }
.log-row .object { font-size: 12.5px; color: var(--text); }
.log-wrap { border: 1px solid var(--line); border-radius: 4px; overflow: hidden; background: var(--panel); }

/* ---------- answer ---------- */

.answer {
    border-left: 2px solid var(--accent);
    background: var(--raised);
    border-radius: 0 4px 4px 0;
    padding: 11px 14px;
    font-size: 13.5px;
    line-height: 1.55;
    color: #E4E1DB;
    margin-top: 10px;
}
.heard {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--text-2);
    background: var(--raised);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 9px 12px;
    margin-top: 8px;
}

/* ---------- empty state ---------- */

.empty-title {
    font-size: 30px; font-weight: 600; letter-spacing: -0.02em;
    color: var(--text); margin: 10px 0 8px 0;
}
.empty-sub { font-size: 14.5px; line-height: 1.6; color: var(--text-2); max-width: 620px; margin: 0; }
.steps { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 20px; margin-top: 8px; }
.step { border-top: 1px solid var(--line); padding-top: 12px; }
.step .n { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--accent); }
.step .h { font-size: 13px; font-weight: 600; color: var(--text); margin: 7px 0 5px 0; }
.step .b { font-size: 12.5px; line-height: 1.5; color: var(--text-3); }

/* ---------- streamlit widgets ---------- */

[data-testid="stVideo"] video {
    border: 1px solid var(--line);
    border-radius: 4px;
    background: #000;
}

[data-testid="stFileUploader"] {
    background: var(--panel);
    border: 1px dashed var(--line-2);
    border-radius: 6px;
    padding: 10px 14px;
}
[data-testid="stFileUploader"] section { background: transparent; }

.stButton > button {
    height: 44px;
    border-radius: 4px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    background: transparent;
    color: var(--text-2);
    border: 1px solid var(--line-2);
}
.stButton > button:hover { border-color: var(--text-3); color: var(--text); background: var(--raised); }

.stButton > button[kind="primary"] {
    background: var(--accent);
    color: #1A1003;
    border: 1px solid var(--accent);
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover { background: #FFBB43; color: #1A1003; }

.stTextInput input {
    background: var(--chrome);
    color: var(--text);
    border: 1px solid var(--line-2);
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    height: 44px;
}
.stTextInput input::placeholder { color: var(--text-4); }

[data-testid="stCaptionContainer"] { color: var(--text-3) !important; }

hr { border-color: var(--line) !important; }
</style>
""",
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None

header = st.empty()


def render_header(status, result=None):
    led, tone = {
        "ready": ("#4A5058", "var(--text-3)"),
        "analyzing": ("var(--accent)", "var(--accent)"),
        "complete": ("var(--ok)", "var(--text-2)"),
    }[status]

    meta = ""
    if result:
        events = result.get("events", [])
        span = max((e["t"] for e in events), default=0.0)
        meta = (
            f'<span class="name">{html.escape(str(result.get("video_id", "")))}</span>'
            f'<span class="sep">/</span><span>{len(events)} events</span>'
            f'<span class="sep">/</span><span>{span:.1f}s span</span>'
        )

    header.markdown(
        f"""
<div class="topbar">
  <span class="mark"><span class="mark-dot"></span>WHAT HAPPENED HERE</span>
  <span class="rule"></span>
  <span class="status" style="color:{tone}"><span class="led" style="background:{led}"></span>{status}</span>
  <span class="meta">{meta}</span>
</div>
""",
        unsafe_allow_html=True,
    )


render_header("complete" if st.session_state.result else "ready", st.session_state.result)

uploaded_file = st.file_uploader(
    "Upload a recording", type=["mp4"], label_visibility="collapsed"
)

col_upload, col_analyze, _ = st.columns([1, 1, 5])
with col_upload:
    if st.button("Upload", disabled=uploaded_file is None):
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
    if st.button("Analyze", type="primary"):
        render_header("analyzing")
        with st.spinner("Watching the recording..."):
            res, err = call_api("post", "/analyze", MODEL_TIMEOUT)
        if err:
            st.error(err)
        elif res.status_code == 200:
            st.session_state.result = res.json()
        else:
            st.error(api_error(res, "Analysis failed"))
        render_header(
            "complete" if st.session_state.result else "ready", st.session_state.result
        )

if not st.session_state.result:
    st.markdown(
        """
<div class="panel" style="padding: 28px 30px;">
  <span class="eyebrow">INTAKE</span>
  <div class="empty-title">Upload footage to begin</div>
  <p class="empty-sub">A short clip works best. Everything runs on this machine &mdash; the video is not sent anywhere, and each new upload replaces the last one.</p>
  <div class="steps">
    <div class="step"><div class="n">01</div><div class="h">Detect &amp; track</div><div class="b">YOLOv8n finds people and objects; BoT-SORT keeps an identity on each one across frames.</div></div>
    <div class="step"><div class="n">02</div><div class="h">Extract events</div><div class="b">Track lifetimes and proximity become enter, exit, place, pick up and approach.</div></div>
    <div class="step"><div class="n">03</div><div class="h">Describe</div><div class="b">Keyframes go to Qwen2-VL, which writes the report in plain language.</div></div>
    <div class="step"><div class="n">04</div><div class="h">Ask</div><div class="b">Question the footage by text or voice and hear the answer read back.</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

else:
    result = st.session_state.result
    events = result.get("events", [])
    colors = track_colors(events)
    span = max((e["t"] for e in events), default=0.0)

    col1, col2 = st.columns([2, 1], gap="medium")

    with col1:
        st.markdown(
            '<div class="section"><h2>Monitor</h2>'
            '<span class="count">yolov8n + botsort</span></div>',
            unsafe_allow_html=True,
        )
        if result.get("annotated_video_path"):
            video_res, video_err = call_api("get", "/annotated_video", API_TIMEOUT)
            if video_err:
                st.error(video_err)
            elif video_res.status_code == 200:
                st.video(video_res.content)
            else:
                st.error(api_error(video_res, "Could not load the annotated video."))
        else:
            st.markdown(
                '<div class="panel" style="color: var(--text-3); font-size: 13px;">'
                "No annotated video was produced for this clip.</div>",
                unsafe_allow_html=True,
            )

        if span > 0:
            ticks = "".join(
                f'<div class="tick" style="left: calc({e["t"] / span * 100:.2f}% - 1px);'
                f' background: {colors[e["subject"]]};"></div>'
                for e in events
            )
            st.markdown(
                f"""
<div class="timeline"><div class="track"></div>{ticks}</div>
<div class="timeline-scale"><span>00:00.0</span><span>each mark is an event, coloured by subject</span><span>{timecode(span)}</span></div>
""",
                unsafe_allow_html=True,
            )

    with col2:
        is_template = result.get("summary_source") == "template"
        badge = (
            '<span class="badge badge-tpl">template fallback</span>'
            if is_template
            else '<span class="badge badge-vlm">Qwen2-VL</span>'
        )
        st.markdown(
            f"""
<div class="panel">
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="font-size:13px; font-weight:600; color:var(--text);">Field report</span>
    <span style="flex-grow:1;"></span>{badge}
  </div>
  <p class="report">{html.escape(result["summary"])}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        if is_template:
            st.caption(
                "The vision model did not answer, so this is a rule-based summary "
                "of the detected events rather than a description of the footage."
            )

        if st.button("Hear report"):
            with st.spinner("Generating voice..."):
                play_speech(result["summary"])

    st.markdown(
        f'<div class="section"><h2>Event log</h2>'
        f'<span class="count">{len(events)} events</span></div>',
        unsafe_allow_html=True,
    )

    if events:
        rows = "".join(
            f'<div class="log-row">'
            f'<span class="t">{timecode(e["t"])}</span>'
            f'<span class="subject"><span class="chip" style="background:{colors[e["subject"]]}"></span>'
            f'{html.escape(str(e["subject"]))}</span>'
            f'<span class="action">{html.escape(str(e["event"]))}</span>'
            f'<span class="object">{html.escape(str(e.get("object") or "—"))}</span>'
            f"</div>"
            for e in events
        )
        st.markdown(
            f"""
<div class="log-wrap">
  <div class="log-head"><span>TIME</span><span>SUBJECT</span><span>ACTION</span><span>OBJECT</span></div>
  {rows}
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="panel" style="color: var(--text-3); font-size: 13px;">'
            "No people or objects were tracked in this clip.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section" style="margin-top:22px;"><h2>Ask the observer</h2>'
        '<span class="count">text or voice</span></div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([4, 1], gap="small")
    with col_a:
        question = st.text_input(
            "Question",
            placeholder="Did anyone leave an object behind?",
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
                f'<div class="heard">heard: {html.escape(question)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error(api_error(res, "Could not transcribe the recording."))

    if st.button("Ask", type="primary", key="ask_button") and question:
        with st.spinner("Thinking..."):
            res, err = call_api(
                "post", "/ask", MODEL_TIMEOUT, params={"question": question}
            )
        if err:
            st.error(err)
        elif res.status_code == 200:
            answer = res.json()["answer"]
            st.markdown(
                f'<div class="answer">{html.escape(answer)}</div>',
                unsafe_allow_html=True,
            )
            play_speech(answer)
        else:
            st.error(api_error(res, "Question failed"))
