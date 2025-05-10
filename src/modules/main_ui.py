import streamlit as st
import zipfile
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import cv2

from modules.pose_estimators import get_pose_analyzer
from modules.video_estimation import run_video_estimation, generate_report, save_session
from modules.webcam_transformers import (
    WebcamPoseTransformer,
    WebcamPostureFeedbackTransformer,
    ExerciseAnalysisTransformer,
)
from modules.image_analysis import run_image_analysis
from modules.session_history import display_session_history
from modules.comparison_mode import compare_pose_images


# ─────────────────────────────  CSS  ────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
<style>
body                    {background:#f8f9fa;font-family:'Segoe UI',Tahoma;}
.sidebar .sidebar-content{
    background:linear-gradient(135deg,#2c3e50,#3498db);color:#fff;}
button[data-testid="tab"]{
    background:#3498db;color:#fff;border-radius:4px 4px 0 0;
    padding:0.45em 1.1em;border:none;margin-right:4px;}
button[data-testid="tab"][aria-selected="true"]{
    background:#1f4166;font-weight:bold;}
.stButton>button{
    background:#1f4166;color:#fff;border-radius:4px;padding:0.45em 1.1em;}
.stButton>button:hover{background:#2960a3;}
</style>
""",
        unsafe_allow_html=True,
    )


inject_css()

# ─────────────────────────────  MAIN  ───────────────────────────
def main():
    st.title("Testing")
    st.caption("")

    # ── sidebar settings ─────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox("Pose model", ["MediaPipe", "MoveNet", "OpenPose"])
        thr = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
        enable_alerts = st.checkbox("Enable alerts", True)
        alert_deg = st.slider("Alert sensitivity (°)", 0, 30, 10)

        st.divider()
        st.subheader("Session")
        if st.button("Clear session"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.success("Session cleared")

        if st.button("Download session ZIP"):
            collected, mapping = {}, {
                "original_image.png": "last_original",
                "pose_image.png": "last_pose",
                "skeleton_image.png": "last_skeleton",
                "recorded_video.mp4": "last_recorded_video",
                "skeleton_video.mp4": "last_skeleton_video",
                "metrics.csv": "last_metrics_csv",
            }
            for out, key in mapping.items():
                if key in st.session_state:
                    collected[out] = st.session_state[key]
            if collected:
                buf = BytesIO()
                with zipfile.ZipFile(buf, "w") as zf:
                    for n, d in collected.items():
                        zf.writestr(n, d)
                buf.seek(0)
                st.download_button("Save ZIP", buf, "poseji_session.zip")
            else:
                st.info("Nothing stored yet.")

    # ── load pose analyzer ───────────────────────────────────────
    MODEL_PATH = "E:\Holowellness\Posture Analysis\Posture Analysis\models\graph_opt.pb"
    if model_choice == "MoveNet":
        analyzer = get_pose_analyzer(
            model_choice, "E:\Holowellness\Posture Analysis\Posture Analysis\models\movenet_lightning_fp16.tflite"
        )
    else:
        analyzer = get_pose_analyzer(model_choice, MODEL_PATH)

    if analyzer is None:
        st.error("Pose model failed to load. Check paths.")
        return

    # ── tabs for every mode ──────────────────────────────────────
    tabs = st.tabs(
        [
            "Image • Basic",
            "Image • Biomech",
            "Image • Metrics",
            "Image • 3-D",
            "Compare Images",
            "Video Estimation",
            # "Live Webcam",
            # "Posture Feedback",
            # "Exercise Coach",
            # "History",
        ]
    )

    # 0 Basic
    with tabs[0]:
        img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img0")
        if img:
            im = Image.open(img).convert("RGB")
            run_image_analysis(analyzer, im, thr, model_choice, "Basic Pose Detection", "t0")

    # 1 Biomech
    with tabs[1]:
        img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img1")
        if img:
            im = Image.open(img)
            run_image_analysis(analyzer, im, thr, model_choice, "Biomechanical Analysis", "t1")

    # 2 Metrics
    with tabs[2]:
        img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img2")
        if img:
            im = Image.open(img)
            run_image_analysis(analyzer, im, thr, model_choice, "Detailed Metrics", "t2")

    # 3 3-D
    with tabs[3]:
        img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img3")
        if img:
            im = Image.open(img)
            run_image_analysis(analyzer, im, thr, model_choice, "3D Pose Visualization", "t3")

    # 4 Comparison
    with tabs[4]:
        compare_pose_images(analyzer, thr, model_choice)

    # 5 Video
    with tabs[5]:
        vid = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "gif"], key="vid")
        rec = st.checkbox("Record processed video", True, key="rec")
        skel = st.checkbox("Extract skeleton video", True, key="skel")
        if vid:
            rvid, svid, met = run_video_estimation(analyzer, vid, thr, rec, extract_skeleton=skel)
            if rvid: st.video(rvid)
            if svid: st.video(svid)
            st.write(met)
            if st.button("Save metrics", key="save_vid"):
                save_session(met, "Video Estimation")

    # # 6 Live webcam
    # with tabs[6]:
    #     st.subheader("Live Webcam Pose Detection")
    #     from streamlit_webrtc import webrtc_streamer

    #     webrtc_streamer(
    #         key="live",
    #         video_transformer_factory=lambda: WebcamPoseTransformer(analyzer, thr),
    #         audio_receiver_size=0,
    #     )

    # # 7 Posture feedback
    # with tabs[7]:
    #     st.subheader("Real-time Posture Feedback")
    #     from streamlit_webrtc import webrtc_streamer

    #     webrtc_streamer(
    #         key="feedback",
    #         video_transformer_factory=lambda: WebcamPostureFeedbackTransformer(
    #             analyzer, thr, enable_alerts, alert_deg
    #         ),
    #         audio_receiver_size=0,
    #     )

    # # 8 Exercise coach
    # with tabs[8]:
    #     st.subheader("Exercise Analysis & Coaching (squats demo)")
    #     from streamlit_webrtc import webrtc_streamer

    #     webrtc_streamer(
    #         key="coach",
    #         video_transformer_factory=lambda: ExerciseAnalysisTransformer(analyzer, thr),
    #         audio_receiver_size=0,
    #     )

    # # 9 History
    # with tabs[9]:
    #     display_session_history()


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
