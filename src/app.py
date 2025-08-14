import streamlit as st
import zipfile
from io import BytesIO
import httpx
import os
from dotenv import load_dotenv
from modules.pose_estimators import get_pose_analyzer
from modules.video_estimation import run_video_estimation, save_session
from modules.squat_analysis import analyze_squat_from_sequence

# Load environment variables
load_dotenv()
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://15.152.146.34/api/chat")
USER_ID = os.getenv("USER_ID", "60d5ec49e472e3a8e4e1d3b4")

def inject_css() -> None:
    st.markdown(
        """
<style>
body {background:#f8f9fa;font-family:'Segoe UI',Tahoma;}
.sidebar .sidebar-content{
    background:linear-gradient(135deg,#2c3e50,#3498db);color:#fff;}
.stButton>button{
    background:#1f4166;color:#fff;border-radius:4px;padding:0.45em 1.1em;}
.stButton>button:hover{background:#2960a3;}
</style>
""",
        unsafe_allow_html=True,
    )

inject_css()

# RAG Chatbot Functions
async def get_rag_chatbot_analysis(prompt: str) -> dict:
    """Get analysis from RAG chatbot."""
    
    payload = {
        "query": prompt,
        "user_id": USER_ID
    }
    
    try:
        # Timeout ditingkatkan untuk tugas LLM yang mungkin berjalan lama
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(RAG_ENDPOINT, json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("response", "")

            diagnosis = ""
            recommendations = []
            
            if "Diagnosis:" in content:
                parts = content.split("Diagnosis:", 1)
                # Periksa apakah Recommendations: ada di bagian kedua
                if len(parts) > 1 and "Recommendations:" in parts[1]:
                    diag_part, rec_part = parts[1].split("Recommendations:", 1)
                    diagnosis = diag_part.strip()
                    recommendations = [line.strip('- ').strip() for line in rec_part.strip().split('\n') if line.strip()]
                elif len(parts) > 1:
                    diagnosis = parts[1].strip()
            elif content:
                diagnosis = content  # Fallback jika tidak ada format spesifik
                
            return {
                "diagnosis_summary": diagnosis,
                "exercise_recommendation": recommendations
            }

    except httpx.RequestError as e:
        error_message = f"Gagal menghubungi chatbot di {RAG_ENDPOINT}. Pastikan endpoint berjalan dan dapat diakses. Detail: {e}"
        return {"diagnosis_summary": error_message, "exercise_recommendation": []}
    except Exception as e:
        return {"diagnosis_summary": f"Terjadi kesalahan tak terduga: {e}", "exercise_recommendation": []}

def build_squat_analysis_prompt(squat_metrics, squat_flags, front_video_path="", side_video_path="", back_video_path="") -> str:
    """Build prompt for RAG chatbot analysis of squat performance."""
    
    prompt = f"""
You are a biomechanics and physical therapy expert specializing in squat analysis. Based on the following squat metrics and flags, provide a comprehensive analysis and recommendations.

SQUAT ANALYSIS DATA:

Front View Metrics:
- Thorax side bend max: {getattr(squat_metrics, 'thorax_side_bend_max_deg', 'N/A')}°
- Pelvis drop at depth: {getattr(squat_metrics, 'pelvis_drop_deg_at_depth', 'N/A')}°
- Left foot external rotation: {getattr(squat_metrics, 'foot_ER_deg_L_at_depth', 'N/A')}°
- Right foot external rotation: {getattr(squat_metrics, 'foot_ER_deg_R_at_depth', 'N/A')}°
- Left knee valgus: {getattr(squat_metrics, 'knee_valgus_deg_L_at_depth', 'N/A')}°
- Right knee valgus: {getattr(squat_metrics, 'knee_valgus_deg_R_at_depth', 'N/A')}°
- Weight bearing ratio (right): {getattr(squat_metrics, 'com_shift_ratio_right', 'N/A')}

Side View Metrics:
- Trunk lean max: {getattr(squat_metrics, 'trunk_lean_max_deg', 'N/A')}°
- Left knee flexion max: {getattr(squat_metrics, 'knee_flex_max_deg_L', 'N/A')}°
- Right knee flexion max: {getattr(squat_metrics, 'knee_flex_max_deg_R', 'N/A')}°
- Left hip flexion max: {getattr(squat_metrics, 'hip_flex_max_deg_L', 'N/A')}°
- Right hip flexion max: {getattr(squat_metrics, 'hip_flex_max_deg_R', 'N/A')}°
- Left ankle dorsiflexion at depth: {getattr(squat_metrics, 'ankle_dorsi_deg_L_at_depth', 'N/A')}°
- Right ankle dorsiflexion at depth: {getattr(squat_metrics, 'ankle_dorsi_deg_R_at_depth', 'N/A')}°
- Squat depth (thigh angle): {getattr(squat_metrics, 'squat_depth_thigh_deg', 'N/A')}°

Back View Metrics:
- Thorax side bend max: {getattr(squat_metrics, 'thorax_side_bend_max_deg', 'N/A')}°
- Pelvis drop at depth: {getattr(squat_metrics, 'pelvis_drop_deg_at_depth', 'N/A')}°
- Left knee valgus: {getattr(squat_metrics, 'knee_valgus_deg_L_at_depth', 'N/A')}°
- Right knee valgus: {getattr(squat_metrics, 'knee_valgus_deg_R_at_depth', 'N/A')}°

FLAGS (Rule-based Findings):
"""
    
    # Add flags to prompt
    for flag_name in dir(squat_flags):
        if not flag_name.startswith('_'):
            flag_value = getattr(squat_flags, flag_name, False)
            prompt += f"- {flag_name}: {flag_value}\n"
    
    prompt += f"""
VIDEO FILES:
- Front view: {front_video_path}
- Side view: {side_video_path}
- Back view: {back_video_path}

Please provide a comprehensive analysis in the following format:

Diagnosis: [Detailed analysis of the person's squat form, highlighting key findings, imbalances, and potential issues based on the metrics and flags.]

Recommendations:
- [First specific exercise or corrective recommendation with reasoning]
- [Second recommendation with reasoning]
- [Third recommendation with reasoning]
- [Continue as needed...]

Focus on:
1. Movement quality and form issues
2. Muscle imbalances and compensations
3. Specific exercises to address identified problems
4. Progression recommendations
5. Safety considerations
"""
    
    return prompt

def main():
    st.title("HOLOWELLNESS EXPERT POSTURE ANALYSIS TOOL")
    st.caption("")

    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox("Pose model", ["MediaPipe", "MoveNet", "OpenPose"])
        thr = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)

    MODEL_PATH = "models/graph_opt.pb"
    if model_choice == "MoveNet":
        analyzer = get_pose_analyzer(
            model_choice, "models/movenet_lightning_fp16.tflite"
        )
    else:
        analyzer = get_pose_analyzer(model_choice, MODEL_PATH)

    if analyzer is None:
        st.error("Pose model failed to load. Check paths.")
        return

    st.header("Video Estimation - 3 Angle (Front, Side, Back)")
    col1, col2, col3 = st.columns(3)
    with col1:
        vid_front = st.file_uploader("Upload Front", type=["mp4", "avi", "mov", "gif"], key="vid_front")
    with col2:
        vid_side = st.file_uploader("Upload Side", type=["mp4", "avi", "mov", "gif"], key="vid_side")
    with col3:
        vid_back = st.file_uploader("Upload Back", type=["mp4", "avi", "mov", "gif"], key="vid_back")

    rec = st.checkbox("Record processed video per angle", True, key="rec")
    skel = st.checkbox("Extract skeleton video per angle", True, key="skel")

    def process_angle(label, file):
        if not file:
            return None
        st.subheader(f"{label} View")
        rvid, svid, met = run_video_estimation(
            analyzer, file, thr, rec, extract_skeleton=False, compute_builtin_metrics=False, ui_mode=True
        )
        if rvid: st.video(rvid)
        pose_frames = met.get("pose_frames", []) if isinstance(met, dict) else []
        reps = met.get("squat_reps", 0) if isinstance(met, dict) else 0
        st.info(f"Repetisi squat terdeteksi: {reps}")
        if reps < 5:
            st.warning("Repetisi < 5. Hasil analisis mungkin kurang stabil.")
        return pose_frames, reps

    all_results = {}
    if vid_front or vid_side or vid_back:
        pf_front, rep_front = process_angle("Front", vid_front) if vid_front else ([], 0)
        pf_side, rep_side = process_angle("Side", vid_side) if vid_side else ([], 0)
        pf_back, rep_back = process_angle("Back", vid_back) if vid_back else ([], 0)

        # Analisis khusus per angle (hanya metrik relevan)
        def show_metrics_flags(label, pose_frames):
            if not pose_frames:
                return
            metrics, flags = analyze_squat_from_sequence(pose_frames, score_thr=thr)
            st.markdown(f"**{label} Metrics**")
            if label == "Front":
                st.json({
                    "thorax_side_bend_max_deg": metrics.thorax_side_bend_max_deg,
                    "pelvis_drop_deg_at_depth": metrics.pelvis_drop_deg_at_depth,
                    "foot_ER_deg_L_at_depth": metrics.foot_ER_deg_L_at_depth,
                    "foot_ER_deg_R_at_depth": metrics.foot_ER_deg_R_at_depth,
                    "knee_valgus_deg_L_at_depth": metrics.knee_valgus_deg_L_at_depth,
                    "knee_valgus_deg_R_at_depth": metrics.knee_valgus_deg_R_at_depth,
                    "com_shift_ratio_right": metrics.com_shift_ratio_right
                })
            elif label == "Side":
                st.json({
                    "trunk_lean_max_deg": metrics.trunk_lean_max_deg,
                    "knee_flex_max_deg_L": metrics.knee_flex_max_deg_L,
                    "knee_flex_max_deg_R": metrics.knee_flex_max_deg_R,
                    "hip_flex_max_deg_L": metrics.hip_flex_max_deg_L,
                    "hip_flex_max_deg_R": metrics.hip_flex_max_deg_R,
                    "ankle_dorsi_deg_L_at_depth": metrics.ankle_dorsi_deg_L_at_depth,
                    "ankle_dorsi_deg_R_at_depth": metrics.ankle_dorsi_deg_R_at_depth,
                    "squat_depth_thigh_deg": metrics.squat_depth_thigh_deg
                })
            elif label == "Back":
                st.json({
                    "thorax_side_bend_max_deg": metrics.thorax_side_bend_max_deg,
                    "pelvis_drop_deg_at_depth": metrics.pelvis_drop_deg_at_depth,
                    "knee_valgus_deg_L_at_depth": metrics.knee_valgus_deg_L_at_depth,
                    "knee_valgus_deg_R_at_depth": metrics.knee_valgus_deg_R_at_depth
                })
            st.markdown(f"**{label} Flags**")
            if label == "Front":
                st.json({
                    "thorax_side_bend_right": flags.thorax_side_bend_right,
                    "thorax_side_bend_left": flags.thorax_side_bend_left,
                    "right_foot_ER": flags.right_foot_ER,
                    "left_foot_ER": flags.left_foot_ER,
                    "left_knee_valgus": flags.left_knee_valgus,
                    "right_knee_valgus": flags.right_knee_valgus,
                    "weight_bearing_right": flags.weight_bearing_right,
                    "weight_bearing_left": flags.weight_bearing_left
                })
            elif label == "Side":
                st.json({
                    "knee_dominant": flags.knee_dominant,
                    "hip_dominant": flags.hip_dominant,
                    "trunk_leans_anterior": flags.trunk_leans_anterior,
                    "insufficient_right_ankle_dorsi": flags.insufficient_right_ankle_dorsi,
                    "insufficient_left_ankle_dorsi": flags.insufficient_left_ankle_dorsi,
                    "thoracolumbar_hyperextension": flags.thoracolumbar_hyperextension,
                    "insufficient_depth_parallel": flags.insufficient_depth_parallel
                })
            elif label == "Back":
                st.json({
                    "thorax_side_bend_right": flags.thorax_side_bend_right,
                    "thorax_side_bend_left": flags.thorax_side_bend_left,
                    "left_knee_valgus": flags.left_knee_valgus,
                    "right_knee_valgus": flags.right_knee_valgus
                })

        show_metrics_flags("Front", pf_front)
        show_metrics_flags("Side", pf_side)
        show_metrics_flags("Back", pf_back)

        # AI Analysis Section
        st.markdown("---")
        st.header("🤖 AI-Powered Analysis")
        
        # Store metrics and flags for AI analysis
        if pf_side:
            side_metrics, side_flags = analyze_squat_from_sequence(pf_side, score_thr=thr)
            
            if st.button("Get AI Analysis", type="primary"):
                with st.spinner("Analyzing with AI..."):
                    try:
                        # Build prompt for RAG analysis
                        prompt = build_squat_analysis_prompt(
                            side_metrics, 
                            side_flags,
                            "Front video uploaded" if vid_front else "No front video",
                            "Side video uploaded" if vid_side else "No side video",
                            "Back video uploaded" if vid_back else "No back video"
                        )
                        
                        # Get AI analysis using synchronous approach
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            rag_result = loop.run_until_complete(get_rag_chatbot_analysis(prompt))
                        finally:
                            loop.close()
                        
                        # Display AI Analysis Results
                        st.success("✅ AI Analysis Complete!")
                        
                        # Diagnosis Summary
                        st.subheader("📋 Diagnosis Summary")
                        st.write(rag_result["diagnosis_summary"])
                        
                        # Exercise Recommendations
                        st.subheader("💪 Exercise Recommendations")
                        if rag_result["exercise_recommendation"]:
                            for i, rec in enumerate(rag_result["exercise_recommendation"], 1):
                                st.markdown(f"**{i}.** {rec}")
                        else:
                            st.info("No specific exercise recommendations available.")
                        
                        # Save AI analysis to session
                        ai_analysis_data = {
                            "diagnosis_summary": rag_result["diagnosis_summary"],
                            "exercise_recommendations": rag_result["exercise_recommendation"],
                            "timestamp": st.session_state.get("timestamp", "Unknown")
                        }
                        
                        if st.button("Save AI Analysis"):
                            save_session(ai_analysis_data, "AI Analysis")
                            st.success("AI Analysis saved to session!")
                            
                    except Exception as e:
                        st.error(f"❌ Error during AI analysis: {str(e)}")
                        st.info("Please check if RAG endpoint is accessible and try again.")
        else:
            st.warning("⚠️ Upload and process side view video first to enable AI analysis.")


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
