from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, Tuple, List
import sys
import os
import httpx
import base64
import cv2
import numpy as np
import time
import uuid

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from modules.pose_estimators import get_pose_analyzer
from modules.video_estimation import run_video_estimation
from modules.squat_analysis import analyze_squat_from_sequence

app = FastAPI(title="Holowellness Squat Analysis API")

# RAG Chatbot Configuration
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://15.152.146.34/api/chat")
USER_ID = os.getenv("USER_ID", "60d5ec49e472e3a8e4e1d3b4")

async def get_rag_chatbot_analysis(prompt: str) -> dict:
    """Get analysis from RAG chatbot."""
    
    payload = {
        "query": prompt,
        "user_id": USER_ID
    }
    
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(RAG_ENDPOINT, json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("response", "")

            diagnosis = ""
            recommendations = []
            
            if "Diagnosis:" in content:
                parts = content.split("Diagnosis:", 1)
                if len(parts) > 1 and "Recommendations:" in parts[1]:
                    diag_part, rec_part = parts[1].split("Recommendations:", 1)
                    diagnosis = diag_part.strip()
                    recommendations = [line.strip('- ').strip() for line in rec_part.strip().split('\n') if line.strip()]
                elif len(parts) > 1:
                    diagnosis = parts[1].strip()
            elif content:
                diagnosis = content
                
            return {
                "diagnosis_summary": diagnosis,
                "exercise_recommendation": recommendations
            }

    except httpx.RequestError as e:
        error_message = f"Failed to connect to RAG chatbot at {RAG_ENDPOINT}. Details: {e}"
        return {"diagnosis_summary": error_message, "exercise_recommendation": []}
    except Exception as e:
        return {"diagnosis_summary": f"An unexpected error occurred: {e}", "exercise_recommendation": []}

def build_squat_analysis_prompt(squat_metrics: Dict[str, Any], squat_flags: Dict[str, Any], 
                               front_video_path: str, side_video_path: str, back_video_path: str) -> str:
    """Build prompt for RAG chatbot analysis of squat performance."""
    
    prompt = f"""
You are a biomechanics and physical therapy expert specializing in squat analysis. Based on the following squat metrics and flags, provide a comprehensive analysis and recommendations.

SQUAT ANALYSIS DATA:

Front View Metrics:
- Thorax side bend max: {squat_metrics.get('thorax_side_bend_max_deg', 'N/A')}°
- Pelvis drop at depth: {squat_metrics.get('pelvis_drop_deg_at_depth', 'N/A')}°
- Left foot external rotation: {squat_metrics.get('foot_ER_deg_L_at_depth', 'N/A')}°
- Right foot external rotation: {squat_metrics.get('foot_ER_deg_R_at_depth', 'N/A')}°
- Left knee valgus: {squat_metrics.get('knee_valgus_deg_L_at_depth', 'N/A')}°
- Right knee valgus: {squat_metrics.get('knee_valgus_deg_R_at_depth', 'N/A')}°
- Weight bearing ratio (right): {squat_metrics.get('com_shift_ratio_right', 'N/A')}

Side View Metrics:
- Trunk lean max: {squat_metrics.get('trunk_lean_max_deg', 'N/A')}°
- Left knee flexion max: {squat_metrics.get('knee_flex_max_deg_L', 'N/A')}°
- Right knee flexion max: {squat_metrics.get('knee_flex_max_deg_R', 'N/A')}°
- Left hip flexion max: {squat_metrics.get('hip_flex_max_deg_L', 'N/A')}°
- Right hip flexion max: {squat_metrics.get('hip_flex_max_deg_R', 'N/A')}°
- Left ankle dorsiflexion at depth: {squat_metrics.get('ankle_dorsi_deg_L_at_depth', 'N/A')}°
- Right ankle dorsiflexion at depth: {squat_metrics.get('ankle_dorsi_deg_R_at_depth', 'N/A')}°
- Squat depth (thigh angle): {squat_metrics.get('squat_depth_thigh_deg', 'N/A')}°

Back View Metrics:
- Thorax side bend max: {squat_metrics.get('thorax_side_bend_max_deg', 'N/A')}°
- Pelvis drop at depth: {squat_metrics.get('pelvis_drop_deg_at_depth', 'N/A')}°
- Left knee valgus: {squat_metrics.get('knee_valgus_deg_L_at_depth', 'N/A')}°
- Right knee valgus: {squat_metrics.get('knee_valgus_deg_R_at_depth', 'N/A')}°

FLAGS (Rule-based Findings):
"""
    
    for flag_name, flag_value in squat_flags.items():
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

def _process_angle(
    analyzer,
    upload: UploadFile,
    angle_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], int, str]:
    """Proses satu angle, kembalikan (metrics, flags, reps, video_path)."""
    upload.file.seek(0)
    rvid, _, met = run_video_estimation(
        analyzer,
        upload.file,
        0.45,  # fixed threshold
        record_video=True,
        extract_skeleton=False,
        compute_builtin_metrics=False,
        ui_mode=False,
    )
    pose_frames = met.get("pose_frames", []) if isinstance(met, dict) else []
    reps = int(met.get("squat_reps", 0)) if isinstance(met, dict) else 0
    
    timestamp = int(time.time())
    video_filename = f"{angle_name.lower()}_overlay_{timestamp}.mp4"
    
    output_dir = "overlay_video_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_path = ""
    if rvid:
        video_path = os.path.join(output_dir, video_filename)
        with open(video_path, "wb") as f:
            f.write(rvid)
    
    if not pose_frames:
        return {}, {}, reps, video_path
    
    metrics, flags = analyze_squat_from_sequence(pose_frames, score_thr=0.45)
    return metrics.__dict__, flags.__dict__, reps, video_path

@app.post("/squat-analysis")
async def squat_analysis_api(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    back: UploadFile = File(...),
):
    try:
        print(f"Processing squat analysis request...")
        print(f"Front video: {front.filename}")
        print(f"Side video: {side.filename}")
        print(f"Back video: {back.filename}")

        # Inisialisasi analyzer - MediaPipe fixed
        try:
            analyzer = get_pose_analyzer("MediaPipe", None)
            if analyzer is None:
                return JSONResponse(status_code=500, content={"error": "MediaPipe model failed to load."})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to init MediaPipe model: {e}"})

        # Proses tiap angle
        front_metrics_all, front_flags_all, reps_front, front_video = _process_angle(analyzer, front, "Front")
        side_metrics_all, side_flags_all, reps_side, side_video = _process_angle(analyzer, side, "Side")
        back_metrics_all, back_flags_all, reps_back, back_video = _process_angle(analyzer, back, "Back")
        
        # Get RAG analysis for comprehensive diagnosis
        rag_prompt = build_squat_analysis_prompt(
            side_metrics_all,
            side_flags_all,
            front_video, side_video, back_video
        )
        
        rag_result = await get_rag_chatbot_analysis(rag_prompt)

        # Filter relevant metrics per angle
        front_metrics = {}
        if front_metrics_all:
            front_metrics = {
                "thorax_side_bend_max_deg": float(front_metrics_all.get("thorax_side_bend_max_deg", 0.0)),
                "pelvis_drop_deg_at_depth": float(front_metrics_all.get("pelvis_drop_deg_at_depth", 0.0)),
                "foot_ER_deg_L_at_depth": float(front_metrics_all.get("foot_ER_deg_L_at_depth", 0.0)),
                "foot_ER_deg_R_at_depth": float(front_metrics_all.get("foot_ER_deg_R_at_depth", 0.0)),
                "knee_valgus_deg_L_at_depth": float(front_metrics_all.get("knee_valgus_deg_L_at_depth", 0.0)),
                "knee_valgus_deg_R_at_depth": float(front_metrics_all.get("knee_valgus_deg_R_at_depth", 0.0)),
                "com_shift_ratio_right": float(front_metrics_all.get("com_shift_ratio_right", 0.5)),
            }

        side_metrics = {}
        if side_metrics_all:
            side_metrics = {
                "trunk_lean_max_deg": float(side_metrics_all.get("trunk_lean_max_deg", 0.0)),
                "knee_flex_max_deg_L": float(side_metrics_all.get("knee_flex_max_deg_L", 0.0)),
                "knee_flex_max_deg_R": float(side_metrics_all.get("knee_flex_max_deg_R", 0.0)),
                "hip_flex_max_deg_L": float(side_metrics_all.get("hip_flex_max_deg_L", 0.0)),
                "hip_flex_max_deg_R": float(side_metrics_all.get("hip_flex_max_deg_R", 0.0)),
                "ankle_dorsi_deg_L_at_depth": float(side_metrics_all.get("ankle_dorsi_deg_L_at_depth", 0.0)),
                "ankle_dorsi_deg_R_at_depth": float(side_metrics_all.get("ankle_dorsi_deg_R_at_depth", 0.0)),
                "squat_depth_thigh_deg": float(side_metrics_all.get("squat_depth_thigh_deg", 0.0)),
            }

        back_metrics = {}
        if back_metrics_all:
            back_metrics = {
                "thorax_side_bend_max_deg": float(back_metrics_all.get("thorax_side_bend_max_deg", 0.0)),
                "pelvis_drop_deg_at_depth": float(back_metrics_all.get("pelvis_drop_deg_at_depth", 0.0)),
                "knee_valgus_deg_L_at_depth": float(back_metrics_all.get("knee_valgus_deg_L_at_depth", 0.0)),
                "knee_valgus_deg_R_at_depth": float(back_metrics_all.get("knee_valgus_deg_R_at_depth", 0.0)),
            }

        return {
            "front": {
                "metrics": front_metrics, 
                "flags": front_flags_all, 
                "squat_reps": reps_front,
                "video_overlay_path": front_video
            },
            "side": {
                "metrics": side_metrics, 
                "flags": side_flags_all, 
                "squat_reps": reps_side,
                "video_overlay_path": side_video
            },
            "back": {
                "metrics": back_metrics, 
                "flags": back_flags_all, 
                "squat_reps": reps_back,
                "video_overlay_path": back_video
            },
            "ai_analysis": {
                "diagnosis_summary": rag_result["diagnosis_summary"],
                "exercise_recommendations": rag_result["exercise_recommendation"]
            }
        }
    except Exception as e:
        print(f"Error in squat analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500, 
            content={"error": f"Internal server error: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
