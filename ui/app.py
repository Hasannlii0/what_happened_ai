import streamlit as st
import requests

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="What Happened Here?", layout="wide")
st.title("What Happened Here? — AI Video Understanding")

if "result" not in st.session_state:
    st.session_state.result = None

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    if st.button("Upload"):
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(f"{API_URL}/upload", files={"file": uploaded_file})
        st.success("Video uploaded. Now run analysis.")

if st.button("Analyze"):
    with st.spinner("Analyzing video..."):
        res = requests.post(f"{API_URL}/analyze")
        if res.status_code == 200:
            st.session_state.result = res.json()
        else:
            st.error(res.json().get("error", "Analysis failed"))

if st.button("Reset"):
    st.session_state.result = None
    st.rerun()

if st.session_state.result:
    result = st.session_state.result

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Processed Video")
        if result.get("annotated_video_path"):
            with open(result["annotated_video_path"], "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
        else:
            st.info("No annotated video found.")

    with col2:
        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Event Timeline")
        for e in result["events"]:
            line = f"**{e['t']:.1f}s** — {e['subject']} {e['event']}"
            if e.get("object"):
                line += f" {e['object']}"
            st.write(line)

    st.divider()
    st.subheader("Ask a question")
    question = st.text_input("e.g. Did anyone leave an object?")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            res = requests.post(f"{API_URL}/ask", params={"question": question})
            if res.status_code == 200:
                st.write(res.json()["answer"])
            else:
                st.error(res.json().get("error", "Question failed"))