from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, Tuple, List
import math
import sys
import os
import httpx
import base64
import cv2
import numpy as np
import time
import uuid
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import io
from dotenv import load_dotenv

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from modules.pose_estimators import get_pose_analyzer
from modules.video_estimation import run_video_estimation
from modules.squat_analysis import analyze_squat_from_sequence
from modules.thresholding import filter_and_score_metrics
import base64

app = FastAPI(title="Holowellness Squat Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# RAG Chatbot Configuration
load_dotenv()
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://15.152.36.109/api/chat")
USER_ID = os.getenv("USER_ID", "60d5ec49e472e3a8e4e1d3b4")
# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)
BUCKET_NAME = AWS_BUCKET_NAME

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Squat Analysis API is running"}

# RAG Chatbot Configuration
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://56.155.140.42/api/chat")
USER_ID = os.getenv("USER_ID", "60d5ec49e472e3a8e4e1d3b4")

async def get_rag_chatbot_analysis(prompt: str) -> dict:
    """Get analysis from RAG chatbot only (no fallback)."""
    
    payload = {
        "query": prompt,
        "user_id": USER_ID
    }
    
    try:
        print(f"🤖 Calling RAG chatbot with prompt length: {len(prompt)}")
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(RAG_ENDPOINT, json=payload)
            response.raise_for_status()
            
            data = response.json()
            content = data.get("response", "")
            
            print(f"📝 RAG response length: {len(content)}")
            print(f"📝 RAG response preview: {content[:200]}...")
            
            # Try to parse JSON payload (explanation/recommendations) if present
            try:
                import json as _json
                if isinstance(content, str) and content.strip().startswith("{"):
                    parsed = _json.loads(content)
                    if isinstance(parsed, dict):
                        expl = parsed.get("explanation", "")
                        recs = parsed.get("recommendations", []) or parsed.get("exercise_recommendation", [])
                        if isinstance(recs, str):
                            recs = [recs]
                        if isinstance(recs, list):
                            return {"diagnosis_summary": expl or "", "exercise_recommendation": recs}
            except Exception:
                pass

            # Check if content is empty or too short
            if not content or len(content.strip()) < 10:
                print("⚠️ RAG response too short")
                return {"diagnosis_summary": "", "exercise_recommendation": []}

            diagnosis = ""
            recommendations = []
            
            # Try multiple parsing strategies
            if "Diagnosis:" in content:
                parts = content.split("Diagnosis:", 1)
                if len(parts) > 1 and "Recommendations:" in parts[1]:
                    diag_part, rec_part = parts[1].split("Recommendations:", 1)
                    diagnosis = diag_part.strip()
                    recommendations = [line.strip('- ').strip() for line in rec_part.strip().split('\n') if line.strip() and line.strip().startswith('-')]
                elif len(parts) > 1:
                    diagnosis = parts[1].strip()
            elif "診斷：" in content:
                # Handle Traditional Chinese diagnosis
                parts = content.split("診斷：", 1)
                if len(parts) > 1 and "建議：" in parts[1]:
                    diag_part, rec_part = parts[1].split("建議：", 1)
                    diagnosis = diag_part.strip()
                    recommendations = [line.strip('- ').strip() for line in rec_part.strip().split('\n') if line.strip() and line.strip().startswith('-')]
                elif len(parts) > 1:
                    diagnosis = parts[1].strip()
            elif "diagnosis:" in content.lower():
                # Handle lowercase diagnosis
                parts = content.lower().split("diagnosis:", 1)
                if len(parts) > 1 and "recommendations:" in parts[1]:
                    diag_part, rec_part = parts[1].split("recommendations:", 1)
                    diagnosis = diag_part.strip()
                    recommendations = [line.strip('- ').strip() for line in rec_part.strip().split('\n') if line.strip() and line.strip().startswith('-')]
                elif len(parts) > 1:
                    diagnosis = parts[1].strip()
            elif "analysis:" in content.lower():
                # Handle analysis format
                parts = content.lower().split("analysis:", 1)
                if len(parts) > 1 and "recommendations:" in parts[1]:
                    diag_part, rec_part = parts[1].split("recommendations:", 1)
                    diagnosis = diag_part.strip()
                    recommendations = [line.strip('- ').strip() for line in rec_part.strip().split('\n') if line.strip() and line.strip().startswith('-')]
                elif len(parts) > 1:
                    diagnosis = parts[1].strip()
            elif content:
                # Fallback: try to extract meaningful content (also supports JSON-like keys)
                lines = content.strip().split('\n')
                diagnosis_lines = []
                rec_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line.lower().startswith('"explanation"') or line.lower().startswith('explanation'):
                        # crude JSON-like extraction
                        val = line.split(':', 1)[-1].strip().strip('", ')
                        if val:
                            diagnosis_lines.append(val)
                        continue
                    if line.lower().startswith('"recommendations"') or line.lower().startswith('recommendations'):
                        continue
                    if line and not line.startswith('-') and len(line) > 20:
                        diagnosis_lines.append(line)
                    elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        rec_lines.append(line.strip('-•* '))
                
                diagnosis = ' '.join(diagnosis_lines[:3])
                recommendations = rec_lines[:3]
                
            # Validate response quality
            if len(diagnosis) < 20:
                print("⚠️ Diagnosis too short from RAG")
                return {"diagnosis_summary": "", "exercise_recommendation": []}
            
            # If no recommendations, keep empty (no default)
            if not recommendations:
                recommendations = []
            
            print(f"✅ Parsed diagnosis: {len(diagnosis)} chars")
            print(f"✅ Parsed recommendations: {len(recommendations)} items")
                
            return {
                "diagnosis_summary": diagnosis,
                "exercise_recommendation": recommendations
            }

    except httpx.RequestError as e:
        print(f"❌ RAG request failed: {e}")
        return {"diagnosis_summary": "", "exercise_recommendation": []}
    except Exception as e:
        print(f"❌ RAG unexpected error: {e}")
        return {"diagnosis_summary": "", "exercise_recommendation": []}

async def get_rag_side_analysis(side_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call RAG with a compact, per-side payload and return explanation/recommendations only."""
    try:
        import json as _json
        prompt = (
            "Provide a brief explanation (2-3 sentences) and 3 actionable recommendations for this view.\n" +
            f"DATA:\n{_json.dumps(side_payload, ensure_ascii=False)}\n" +
            "Respond in JSON with keys: explanation (string), recommendations (array of 3 strings)."
        )
        res = await get_rag_chatbot_analysis(prompt)
        # map to expected keys
        return {
            "explanation": res.get("diagnosis_summary", ""),
            "recommendations": res.get("exercise_recommendation", [])
        }
    except Exception:
        return {"explanation": "", "recommendations": []}

# Removed fallback analysis per requirement; RAG-only mode

def build_squat_analysis_prompt(
    front_metrics: Dict[str, Any],
    side_metrics: Dict[str, Any],
    back_metrics: Dict[str, Any],
    front_flags: Dict[str, Any],
    side_flags: Dict[str, Any],
    back_flags: Dict[str, Any],
    sex: Optional[str],
    age: Optional[int],
    reps_front: int,
    reps_side: int,
    reps_back: int,
) -> str:
    """Bangun prompt RAG berisi 14 metrik terfilter per sudut + flags.

    - Hanya memuat metrik yang sudah disaring (CSV) agar konsisten dengan scoring.
    - Nilai dibulatkan 2 desimal, null dibuang.
    """
    import json as _json

    def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (d or {}).items():
            if v is None:
                continue
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    continue
                out[k] = round(v, 2)
            else:
                out[k] = v
        return out

    def _true_flags(flags: Dict[str, Any]) -> List[str]:
        return [k for k, v in (flags or {}).items() if v is True]

    payload = {
        "context": {
            "sex": sex or "",
            "age": age if age is not None else "",
            "reps": {"front": reps_front, "side": reps_side, "back": reps_back},
        },
        "metrics": {
            "front": _compact(front_metrics),
            "side": _compact(side_metrics),
            "back": _compact(back_metrics),
        },
        "flags": {
            "front": _true_flags(front_flags),
            "side": _true_flags(side_flags),
            "back": _true_flags(back_flags),
        },
        "instructions": {
            "format": {
                "Diagnosis": "2-3 sentences, concise",
                "Recommendations": 3,
                "MaxWords": 150
            },
            "focus": [
                "prioritize Poor then Partial",
                "consider cross-view patterns",
                "safety and actionable cues"
            ]
        }
    }

    prompt = (
        "Analyze this squat assessment data and provide a concise report.\n\n"+
        f"DATA:\n{_json.dumps(payload, ensure_ascii=False)}\n\n"+
        "Respond in this exact format:\n"+
        "Diagnosis: ...\n"+
        "Recommendations:\n- ...\n- ...\n- ...\n"
    )
    return prompt

def _process_angle(
    analyzer,
    upload: UploadFile,
    angle_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], int, str, List[Dict[str, Any]], float, List[Dict[str, Any]]]:
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
        try:
            video_path = os.path.join(output_dir, video_filename)
            with open(video_path, "wb") as f:
                f.write(rvid)
            print(f"✅ Video overlay saved: {video_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save video overlay: {e}")
            video_path = ""  # Reset path if save fails
    else:
        print(f"⚠️ No video data available for {angle_name}")
    
    if not pose_frames:
        return {}, {}, reps, video_path, [], None, []
    
    # Ambil fps & rep_events dari hasil video estimation
    fps = met.get("fps") if isinstance(met, dict) else None
    rep_events = met.get("rep_events") if isinstance(met, dict) else None

    metrics, flags = analyze_squat_from_sequence(pose_frames, score_thr=0.45, rep_events=rep_events, fps=fps)
    bottom_snaps = met.get("bottom_snapshots") if isinstance(met, dict) else []
    return metrics.__dict__, flags.__dict__, reps, video_path, rep_events or [], float(fps) if fps else None, bottom_snaps or []


@app.post("/squat-analysis")
async def squat_analysis_api(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    back: UploadFile = File(...),
    sex: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
):
    start_time = time.time()
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
        print("🔄 Processing Front view...")
        front_metrics_all, front_flags_all, reps_front, front_video, front_rep_events, front_fps, _ = _process_angle(analyzer, front, "Front")
        print(f"✅ Front processed: {reps_front} reps, video: {front_video}")
        
        print("🔄 Processing Side view...")
        side_metrics_all, side_flags_all, reps_side, side_video, side_rep_events, side_fps, side_bottom_snaps = _process_angle(analyzer, side, "Side")
        print(f"✅ Side processed: {reps_side} reps, video: {side_video}")
        
        print("🔄 Processing Back view...")
        back_metrics_all, back_flags_all, reps_back, back_video, back_rep_events, back_fps, _ = _process_angle(analyzer, back, "Back")
        print(f"✅ Back processed: {reps_back} reps, video: {back_video}")
        
        # Validate that we have at least some data
        if not side_metrics_all and not front_metrics_all and not back_metrics_all:
            return JSONResponse(
                status_code=400, 
                content={"error": "No valid pose data detected in any video. Please check video quality and ensure person is visible."}
            )
        
        # Filter relevant metrics per angle (only 14 metrics from CSV + statuses)
        front_metrics = {}
        front_metric_status = {}
        front_filtered_full = {}
        if front_metrics_all:
            front_filtered_full = filter_and_score_metrics(front_metrics_all, sex=sex, age=age)
            front_metrics = {k: v for k, v in front_filtered_full.items() if not k.endswith("__status")}
            front_metric_status = {k.replace("__status", ""): v for k, v in front_filtered_full.items() if k.endswith("__status")}

        side_metrics = {}
        side_metric_status = {}
        side_filtered_full = {}
        if side_metrics_all:
            side_filtered_full = filter_and_score_metrics(side_metrics_all, sex=sex, age=age)
            side_metrics = {k: v for k, v in side_filtered_full.items() if not k.endswith("__status")}
            side_metric_status = {k.replace("__status", ""): v for k, v in side_filtered_full.items() if k.endswith("__status")}

        back_metrics = {}
        back_metric_status = {}
        back_filtered_full = {}
        if back_metrics_all:
            back_filtered_full = filter_and_score_metrics(back_metrics_all, sex=sex, age=age)
            back_metrics = {k: v for k, v in back_filtered_full.items() if not k.endswith("__status")}
            back_metric_status = {k.replace("__status", ""): v for k, v in back_filtered_full.items() if k.endswith("__status")}

        # Per-side RAG explanations (front/side/back)
        try:
            front_rag = await get_rag_side_analysis({"sex": sex, "age": age, "metrics": front_filtered_full, "reps": reps_front}) if front_filtered_full else {"explanation": "", "recommendations": []}
            side_rag  = await get_rag_side_analysis({"sex": sex, "age": age, "metrics": side_filtered_full,  "reps": reps_side }) if side_filtered_full  else {"explanation": "", "recommendations": []}
            back_rag  = await get_rag_side_analysis({"sex": sex, "age": age, "metrics": back_filtered_full,  "reps": reps_back }) if back_filtered_full  else {"explanation": "", "recommendations": []}
        except Exception as e:
            print(f"⚠️ RAG per-side failed: {e}")
            front_rag = side_rag = back_rag = {"explanation": "", "recommendations": []}

        # ── Overall scoring (Raw, Penalties, Final, Grade, Parallel achieved) ──────────
        def _status_priority(s: str) -> int:
            if s == "Poor": return 3
            if s == "Partial": return 2
            if s == "Good": return 1
            return 0  # N/A or missing

        def _status_penalty(s: str) -> int:
            if s == "Poor": return 10
            if s == "Partial": return 5
            return 0  # Good/N/A

        # Combine worst status across views for each metric key
        all_metric_keys = set(list(front_metric_status.keys()) + list(side_metric_status.keys()) + list(back_metric_status.keys()))
        combined_status: Dict[str, str] = {}
        total_penalties = 0
        for key in sorted(all_metric_keys):
            sts = []
            if key in front_metric_status: sts.append(front_metric_status[key])
            if key in side_metric_status: sts.append(side_metric_status[key])
            if key in back_metric_status: sts.append(back_metric_status[key])
            if not sts:
                continue
            worst = max(sts, key=_status_priority)
            combined_status[key] = worst
            total_penalties += _status_penalty(worst)

        raw_score = 100
        final_score = max(0, raw_score - total_penalties)

        if final_score >= 85:
            grade = "A"
        elif final_score >= 70:
            grade = "B"
        elif final_score >= 50:
            grade = "C"
        else:
            grade = "D"

        # Parallel achieved from depth status (prefer Side view)
        depth_status_side = side_metric_status.get("squat_depth_thigh_deg") if side_metric_status else None
        depth_status_any = depth_status_side or front_metric_status.get("squat_depth_thigh_deg") or back_metric_status.get("squat_depth_thigh_deg")
        parallel_achieved = bool(depth_status_any == "Good")

        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        try:
            # Ensure all data is JSON serializable
            def clean_for_json(obj):
                # Dict and list: recurse
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                # Numpy scalars → native
                try:
                    import numpy as _np
                    if isinstance(obj, (_np.integer,)):
                        return int(obj)
                    if isinstance(obj, (_np.floating,)):
                        val = float(obj)
                        return None if (math.isnan(val) or math.isinf(val)) else val
                    if isinstance(obj, (_np.bool_,)):
                        return bool(obj)
                except Exception:
                    pass
                # Floats: sanitize NaN/Inf
                if isinstance(obj, float):
                    return None if (math.isnan(obj) or math.isinf(obj)) else obj
                # Primitives
                if isinstance(obj, (int, str, bool)) or obj is None:
                    return obj
                # Fallback: stringify
                return str(obj)
            
            # Build side snapshots (only for Side view, reps 2..4) and save PNG files
            side_snapshots = []
            try:
                import cv2 as _cv
                import os as _os
                import base64 as _b64
                out_dir = _os.path.join("repetition output")
                _os.makedirs(out_dir, exist_ok=True)
                # Prefer in-memory snapshots captured at bottom
                snaps_map = {s.get("rep"): s.get("frame") for s in side_bottom_snaps if isinstance(s, dict)}
                needed_reps = [2, 3, 4]
                present_reps = set()
                for rep_num in needed_reps:
                    img = snaps_map.get(rep_num)
                    if img is None:
                        continue
                    ts = int(time.time())
                    fname = f"side_rep{rep_num}_down_{ts}.png"
                    fpath = _os.path.join(out_dir, fname)
                    if _cv.imwrite(fpath, img):
                        with open(fpath, 'rb') as f:
                            b64 = _b64.b64encode(f.read()).decode('ascii')
                        side_snapshots.append({"rep": rep_num, "image_base64": b64, "file_path": fpath})
                        present_reps.add(rep_num)
                # Fallback to reading overlay video at bottom_time_sec for missing reps
                missing = [r for r in needed_reps if r not in present_reps]
                if missing and side_rep_events and side_fps and side_video and os.path.exists(side_video):
                    cap = _cv.VideoCapture(side_video)
                    total = int(cap.get(_cv.CAP_PROP_FRAME_COUNT) or 0)
                    for rep_num in missing:
                        idx = rep_num - 1
                        if 0 <= idx < len(side_rep_events):
                            bt = side_rep_events[idx].get("bottom_time_sec")
                            if bt is None:
                                continue
                            fi = max(0, min(int(round(bt * side_fps)), total - 1))
                            cap.set(_cv.CAP_PROP_POS_FRAMES, fi)
                            ok, frame = cap.read()
                            if not ok or frame is None:
                                continue
                            ts = int(time.time())
                            fname = f"side_rep{rep_num}_down_{ts}.png"
                            fpath = _os.path.join(out_dir, fname)
                            if _cv.imwrite(fpath, frame):
                                with open(fpath, 'rb') as f:
                                    b64 = _b64.b64encode(f.read()).decode('ascii')
                                side_snapshots.append({"rep": rep_num, "image_base64": b64, "file_path": fpath})
                    cap.release()
            except Exception as _e:
                print(f"⚠️ Side snapshots error: {_e}")

            response_data = {
                "front": {
                    "metrics": clean_for_json(front_metrics), 
                    "metric_status": clean_for_json(front_metric_status), 
                    "explanation": str(front_rag.get("explanation", "")),
                    "recommendations": clean_for_json(front_rag.get("recommendations", [])),
                    "squat_reps": int(reps_front),
                    "video_overlay_path": str(front_video)
                },
                "side": {
                    "metrics": clean_for_json(side_metrics), 
                    "metric_status": clean_for_json(side_metric_status), 
                    "snapshots_down": side_snapshots,
                    "explanation": str(side_rag.get("explanation", "")),
                    "recommendations": clean_for_json(side_rag.get("recommendations", [])),
                    "squat_reps": int(reps_side),
                    "video_overlay_path": str(side_video)
                },
                "back": {
                    "metrics": clean_for_json(back_metrics), 
                    "metric_status": clean_for_json(back_metric_status), 
                    "explanation": str(back_rag.get("explanation", "")),
                    "recommendations": clean_for_json(back_rag.get("recommendations", [])),
                    "squat_reps": int(reps_back),
                    "video_overlay_path": str(back_video)
                },
                "processing_info": {
                    "processing_time_seconds": round(processing_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "completed"
                },
                "overall": {
                    "raw_score": int(raw_score),
                    "total_penalties": int(total_penalties),
                    "final_score": int(final_score),
                    "grade": grade,
                    "parallel_achieved": parallel_achieved
                }
            }
            
            # Validate response data before returning
            if not isinstance(response_data, dict):
                raise ValueError("Response data is not a dictionary")
            
            # Check if all required fields exist
            required_fields = ["front", "side", "back", "processing_info", "overall"]
            for field in required_fields:
                if field not in response_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Test JSON serialization
            import json
            try:
                json.dumps(response_data)
                print(f"✅ Response JSON serialization test passed")
            except Exception as json_error:
                print(f"❌ JSON serialization failed: {json_error}")
                raise ValueError(f"Response data not JSON serializable: {json_error}")
            
            print(f"✅ Response data validated successfully")
            print(f"✅ Squat analysis completed successfully")
            print(f"📊 Response summary:")
            print(f"   - Front metrics: {len(response_data['front']['metrics'])} items")
            print(f"   - Side metrics: {len(response_data['side']['metrics'])} items")
            print(f"   - Back metrics: {len(response_data['back']['metrics'])} items")
            # No AI analysis field in the new response structure
            
            # Return with proper headers
            from fastapi.responses import Response
            return Response(
                content=json.dumps(response_data, ensure_ascii=False),
                media_type="application/json",
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            
        except Exception as response_error:
            print(f"❌ Error building response: {response_error}")
            # Return a minimal valid response (no flags, no ai_analysis)
            minimal_response = {
                "front": {
                    "metrics": {},
                    "metric_status": {},
                    "explanation": "",
                    "recommendations": [],
                    "squat_reps": 0,
                    "video_overlay_path": ""
                },
                "side": {
                    "metrics": {},
                    "metric_status": {},
                    "snapshots_down": [],
                    "explanation": "",
                    "recommendations": [],
                    "squat_reps": 0,
                    "video_overlay_path": ""
                },
                "back": {
                    "metrics": {},
                    "metric_status": {},
                    "explanation": "",
                    "recommendations": [],
                    "squat_reps": 0,
                    "video_overlay_path": ""
                },
                "processing_info": {
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "completed_with_fallback"
                },
                "overall": {
                    "raw_score": 100,
                    "total_penalties": 0,
                    "final_score": 100,
                    "grade": "A",
                    "parallel_achieved": False
                }
            }
            
            # Return minimal response with proper headers
            from fastapi.responses import Response
            import json
            return Response(
                content=json.dumps(minimal_response, ensure_ascii=False),
                media_type="application/json",
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
    except Exception as e:
        print(f"❌ Error in squat analysis: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        print(f"❌ Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a more informative error response
        error_response = {
            "error": f"Internal server error: {str(e)}",
            "error_type": type(e).__name__,
            "timestamp": time.time(),
            "status": "failed"
        }
        
        return JSONResponse(
            status_code=500, 
            content=error_response
        )

def _process_angle_s3(
    analyzer,
    upload: UploadFile,
    angle_name: str,
    customer_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], int, str, List[Dict[str, Any]], float, List[Dict[str, Any]]]:
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
    video_filename = f"dynamic-movement-analysis-result/{customer_id}/{angle_name.lower()}_overlay_{timestamp}.mp4"
    
    if rvid:
        try:

            file_obj = io.BytesIO(rvid)
            s3_client.upload_fileobj(file_obj, BUCKET_NAME, video_filename)
            s3_url = f"https://{BUCKET_NAME}.s3.{s3_client.meta.region_name}.amazonaws.com/{video_filename}"
            
            print(f"✅ Video overlay saved: {s3_url}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save video overlay: {e}")
            s3_url = ""  # Reset path if save fails
    else:
        print(f"⚠️ No video data available for {angle_name}")
        s3_url = ""
    
    if not pose_frames:
        return {}, {}, reps, s3_url, [], None, []
    
    # Ambil fps & rep_events dari hasil video estimation
    fps = met.get("fps") if isinstance(met, dict) else None
    rep_events = met.get("rep_events") if isinstance(met, dict) else None

    metrics, flags = analyze_squat_from_sequence(pose_frames, score_thr=0.45, rep_events=rep_events, fps=fps)
    bottom_snaps = met.get("bottom_snapshots") if isinstance(met, dict) else []
    return metrics.__dict__, flags.__dict__, reps, s3_url, rep_events or [], float(fps) if fps else None, bottom_snaps or []

@app.post("/squat-analysis-s3")
async def squat_analysis_api(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    back: UploadFile = File(...),
    customer_id: str = Form(...),
    sex: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
):
    start_time = time.time()
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
        print("🔄 Processing Front view...")
        front_metrics_all, front_flags_all, reps_front, front_video, front_rep_events, front_fps, _ = _process_angle_s3(analyzer, front, "Front", customer_id)
        print(f"✅ Front processed: {reps_front} reps, video: {front_video}")
        
        print("🔄 Processing Side view...")
        side_metrics_all, side_flags_all, reps_side, side_video, side_rep_events, side_fps, side_bottom_snaps = _process_angle_s3(analyzer, side, "Side", customer_id)
        print(f"✅ Side processed: {reps_side} reps, video: {side_video}")
        
        print("🔄 Processing Back view...")
        back_metrics_all, back_flags_all, reps_back, back_video, back_rep_events, back_fps, _ = _process_angle_s3(analyzer, back, "Back", customer_id)
        print(f"✅ Back processed: {reps_back} reps, video: {back_video}")
        
        # Validate that we have at least some data
        if not side_metrics_all and not front_metrics_all and not back_metrics_all:
            return JSONResponse(
                status_code=400, 
                content={"error": "No valid pose data detected in any video. Please check video quality and ensure person is visible."}
            )
        
        # Filter relevant metrics per angle (only 14 metrics from CSV + statuses)
        front_metrics = {}
        front_metric_status = {}
        front_filtered_full = {}
        if front_metrics_all:
            front_filtered_full = filter_and_score_metrics(front_metrics_all, sex=sex, age=age)
            front_metrics = {k: v for k, v in front_filtered_full.items() if not k.endswith("__status")}
            front_metric_status = {k.replace("__status", ""): v for k, v in front_filtered_full.items() if k.endswith("__status")}

        side_metrics = {}
        side_metric_status = {}
        side_filtered_full = {}
        if side_metrics_all:
            side_filtered_full = filter_and_score_metrics(side_metrics_all, sex=sex, age=age)
            side_metrics = {k: v for k, v in side_filtered_full.items() if not k.endswith("__status")}
            side_metric_status = {k.replace("__status", ""): v for k, v in side_filtered_full.items() if k.endswith("__status")}

        back_metrics = {}
        back_metric_status = {}
        back_filtered_full = {}
        if back_metrics_all:
            back_filtered_full = filter_and_score_metrics(back_metrics_all, sex=sex, age=age)
            back_metrics = {k: v for k, v in back_filtered_full.items() if not k.endswith("__status")}
            back_metric_status = {k.replace("__status", ""): v for k, v in back_filtered_full.items() if k.endswith("__status")}

        # Per-side RAG explanations (front/side/back)
        try:
            front_rag = await get_rag_side_analysis({"sex": sex, "age": age, "metrics": front_filtered_full, "reps": reps_front}) if front_filtered_full else {"explanation": "", "recommendations": []}
            side_rag  = await get_rag_side_analysis({"sex": sex, "age": age, "metrics": side_filtered_full,  "reps": reps_side }) if side_filtered_full  else {"explanation": "", "recommendations": []}
            back_rag  = await get_rag_side_analysis({"sex": sex, "age": age, "metrics": back_filtered_full,  "reps": reps_back }) if back_filtered_full  else {"explanation": "", "recommendations": []}
        except Exception as e:
            print(f"⚠️ RAG per-side failed: {e}")
            front_rag = side_rag = back_rag = {"explanation": "", "recommendations": []}

        # ── Overall scoring (Raw, Penalties, Final, Grade, Parallel achieved) ──────────
        def _status_priority(s: str) -> int:
            if s == "Poor": return 3
            if s == "Partial": return 2
            if s == "Good": return 1
            return 0  # N/A or missing

        def _status_penalty(s: str) -> int:
            if s == "Poor": return 10
            if s == "Partial": return 5
            return 0  # Good/N/A

        # Combine worst status across views for each metric key
        all_metric_keys = set(list(front_metric_status.keys()) + list(side_metric_status.keys()) + list(back_metric_status.keys()))
        combined_status: Dict[str, str] = {}
        total_penalties = 0
        for key in sorted(all_metric_keys):
            sts = []
            if key in front_metric_status: sts.append(front_metric_status[key])
            if key in side_metric_status: sts.append(side_metric_status[key])
            if key in back_metric_status: sts.append(back_metric_status[key])
            if not sts:
                continue
            worst = max(sts, key=_status_priority)
            combined_status[key] = worst
            total_penalties += _status_penalty(worst)

        raw_score = 100
        final_score = max(0, raw_score - total_penalties)

        if final_score >= 85:
            grade = "A"
        elif final_score >= 70:
            grade = "B"
        elif final_score >= 50:
            grade = "C"
        else:
            grade = "D"

        # Parallel achieved from depth status (prefer Side view)
        depth_status_side = side_metric_status.get("squat_depth_thigh_deg") if side_metric_status else None
        depth_status_any = depth_status_side or front_metric_status.get("squat_depth_thigh_deg") or back_metric_status.get("squat_depth_thigh_deg")
        parallel_achieved = bool(depth_status_any == "Good")

        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        try:
            # Ensure all data is JSON serializable
            def clean_for_json(obj):
                # Dict and list: recurse
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                # Numpy scalars → native
                try:
                    import numpy as _np
                    if isinstance(obj, (_np.integer,)):
                        return int(obj)
                    if isinstance(obj, (_np.floating,)):
                        val = float(obj)
                        return None if (math.isnan(val) or math.isinf(val)) else val
                    if isinstance(obj, (_np.bool_,)):
                        return bool(obj)
                except Exception:
                    pass
                # Floats: sanitize NaN/Inf
                if isinstance(obj, float):
                    return None if (math.isnan(obj) or math.isinf(obj)) else obj
                # Primitives
                if isinstance(obj, (int, str, bool)) or obj is None:
                    return obj
                # Fallback: stringify
                return str(obj)
            
            # Build side snapshots (only for Side view, reps 2..4) and save PNG files
            # Build side snapshots (only for Side view, reps 2..4) and upload to S3
            side_snapshots = []
            try:
                import cv2 as _cv
                import time as _time
                import io as _io

                # Init boto3 client (gunakan konfigurasi AWS-mu)
                s3 = s3_client

                # Prefer in-memory snapshots captured at bottom
                snaps_map = {s.get("rep"): s.get("frame") for s in side_bottom_snaps if isinstance(s, dict)}
                needed_reps = [2, 3, 4]
                present_reps = set()

                for rep_num in needed_reps:
                    img = snaps_map.get(rep_num)
                    if img is None:
                        continue

                    # Encode frame ke PNG
                    ok, buffer = _cv.imencode(".png", img)
                    if not ok:
                        continue

                    # Upload ke S3
                    ts = int(_time.time())
                    object_key = f"dynamic-movement-analysis-result/snapshots/{customer_id}/side_rep{rep_num}_down_{ts}.png"
                    s3.put_object(
                        Bucket=BUCKET_NAME,
                        Key=object_key,
                        Body=buffer.tobytes(),
                        ContentType="image/png"
                    )

                    # Buat URL publik
                    s3_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{object_key}"

                    side_snapshots.append({
                        "rep": rep_num,
                        "s3_url": s3_url
                    })
                    present_reps.add(rep_num)

                # Fallback ke overlay video untuk rep yang hilang
                missing = [r for r in needed_reps if r not in present_reps]
                if missing and side_rep_events and side_fps and side_video and os.path.exists(side_video):
                    cap = _cv.VideoCapture(side_video)
                    total = int(cap.get(_cv.CAP_PROP_FRAME_COUNT) or 0)
                    for rep_num in missing:
                        idx = rep_num - 1
                        if 0 <= idx < len(side_rep_events):
                            bt = side_rep_events[idx].get("bottom_time_sec")
                            if bt is None:
                                continue
                            fi = max(0, min(int(round(bt * side_fps)), total - 1))
                            cap.set(_cv.CAP_PROP_POS_FRAMES, fi)
                            ok, frame = cap.read()
                            if not ok or frame is None:
                                continue

                            # Encode frame ke PNG
                            ok2, buffer = _cv.imencode(".png", frame)
                            if not ok2:
                                continue

                            ts = int(_time.time())
                            object_key = f"dynamic-movement-analysis-result/snapshots/{customer_id}/side_rep{rep_num}_down_{ts}.png"
                            s3.put_object(
                                Bucket=BUCKET_NAME,
                                Key=object_key,
                                Body=buffer.tobytes(),
                                ContentType="image/png"
                            )

                            s3_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{object_key}"

                            side_snapshots.append({
                                "rep": rep_num,
                                "s3_url": s3_url
                            })
                    cap.release()
            except Exception as e:
                print("Error generating side snapshots:", str(e))


            response_data = {
                "front": {
                    "metrics": clean_for_json(front_metrics), 
                    "metric_status": clean_for_json(front_metric_status), 
                    "explanation": str(front_rag.get("explanation", "")),
                    "recommendations": clean_for_json(front_rag.get("recommendations", [])),
                    "squat_reps": int(reps_front),
                    "video_overlay_path": str(front_video)
                },
                "side": {
                    "metrics": clean_for_json(side_metrics), 
                    "metric_status": clean_for_json(side_metric_status), 
                    "snapshots_down": side_snapshots,
                    "explanation": str(side_rag.get("explanation", "")),
                    "recommendations": clean_for_json(side_rag.get("recommendations", [])),
                    "squat_reps": int(reps_side),
                    "video_overlay_path": str(side_video)
                },
                "back": {
                    "metrics": clean_for_json(back_metrics), 
                    "metric_status": clean_for_json(back_metric_status), 
                    "explanation": str(back_rag.get("explanation", "")),
                    "recommendations": clean_for_json(back_rag.get("recommendations", [])),
                    "squat_reps": int(reps_back),
                    "video_overlay_path": str(back_video)
                },
                "processing_info": {
                    "processing_time_seconds": round(processing_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "completed"
                },
                "overall": {
                    "raw_score": int(raw_score),
                    "total_penalties": int(total_penalties),
                    "final_score": int(final_score),
                    "grade": grade,
                    "parallel_achieved": parallel_achieved
                }
            }
            
            # Validate response data before returning
            if not isinstance(response_data, dict):
                raise ValueError("Response data is not a dictionary")
            
            # Check if all required fields exist
            required_fields = ["front", "side", "back", "processing_info", "overall"]
            for field in required_fields:
                if field not in response_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Test JSON serialization
            import json
            try:
                json.dumps(response_data)
                print(f"✅ Response JSON serialization test passed")
            except Exception as json_error:
                print(f"❌ JSON serialization failed: {json_error}")
                raise ValueError(f"Response data not JSON serializable: {json_error}")
            
            print(f"✅ Response data validated successfully")
            print(f"✅ Squat analysis completed successfully")
            print(f"📊 Response summary:")
            print(f"   - Front metrics: {len(response_data['front']['metrics'])} items")
            print(f"   - Side metrics: {len(response_data['side']['metrics'])} items")
            print(f"   - Back metrics: {len(response_data['back']['metrics'])} items")
            # No AI analysis field in the new response structure
            
            # Return with proper headers
            from fastapi.responses import Response
            return Response(
                content=json.dumps(response_data, ensure_ascii=False),
                media_type="application/json",
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
            
        except Exception as response_error:
            print(f"❌ Error building response: {response_error}")
            # Return a minimal valid response (no flags, no ai_analysis)
            minimal_response = {
                "front": {
                    "metrics": {},
                    "metric_status": {},
                    "explanation": "",
                    "recommendations": [],
                    "squat_reps": 0,
                    "video_overlay_path": ""
                },
                "side": {
                    "metrics": {},
                    "metric_status": {},
                    "snapshots_down": [],
                    "explanation": "",
                    "recommendations": [],
                    "squat_reps": 0,
                    "video_overlay_path": ""
                },
                "back": {
                    "metrics": {},
                    "metric_status": {},
                    "explanation": "",
                    "recommendations": [],
                    "squat_reps": 0,
                    "video_overlay_path": ""
                },
                "processing_info": {
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "completed_with_fallback"
                },
                "overall": {
                    "raw_score": 100,
                    "total_penalties": 0,
                    "final_score": 100,
                    "grade": "A",
                    "parallel_achieved": False
                }
            }
            
            # Return minimal response with proper headers
            from fastapi.responses import Response
            import json
            return Response(
                content=json.dumps(minimal_response, ensure_ascii=False),
                media_type="application/json",
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
    except Exception as e:
        print(f"❌ Error in squat analysis: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        print(f"❌ Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a more informative error response
        error_response = {
            "error": f"Internal server error: {str(e)}",
            "error_type": type(e).__name__,
            "timestamp": time.time(),
            "status": "failed"
        }
        
        return JSONResponse(
            status_code=500, 
            content=error_response
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
