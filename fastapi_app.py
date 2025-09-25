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
                # Fallback: try to extract meaningful content
                lines = content.strip().split('\n')
                diagnosis_lines = []
                rec_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('-') and len(line) > 20:
                        diagnosis_lines.append(line)
                    elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        rec_lines.append(line.strip('-•* '))
                
                diagnosis = ' '.join(diagnosis_lines[:3])  # Take first 3 meaningful lines
                recommendations = rec_lines[:3]  # Take first 3 recommendations
                
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
        return {}, {}, reps, video_path
    
    # Ambil fps & rep_events dari hasil video estimation
    fps = met.get("fps") if isinstance(met, dict) else None
    rep_events = met.get("rep_events") if isinstance(met, dict) else None

    metrics, flags = analyze_squat_from_sequence(pose_frames, score_thr=0.45, rep_events=rep_events, fps=fps)
    return metrics.__dict__, flags.__dict__, reps, video_path


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
        front_metrics_all, front_flags_all, reps_front, front_video = _process_angle(analyzer, front, "Front")
        print(f"✅ Front processed: {reps_front} reps, video: {front_video}")
        
        print("🔄 Processing Side view...")
        side_metrics_all, side_flags_all, reps_side, side_video = _process_angle(analyzer, side, "Side")
        print(f"✅ Side processed: {reps_side} reps, video: {side_video}")
        
        print("🔄 Processing Back view...")
        back_metrics_all, back_flags_all, reps_back, back_video = _process_angle(analyzer, back, "Back")
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

        # Get RAG analysis for comprehensive diagnosis (uses full filtered incl. statuses)
        try:
            rag_prompt = build_squat_analysis_prompt(
                front_filtered_full, side_filtered_full, back_filtered_full,
                front_flags_all, side_flags_all, back_flags_all,
                sex, age, reps_front, reps_side, reps_back
            )
            print(f"🤖 Calling RAG with prompt length: {len(rag_prompt)}")
            rag_result = await get_rag_chatbot_analysis(rag_prompt)
            print(f"✅ RAG analysis completed (no fallback mode)")
            if not rag_result:
                rag_result = {"diagnosis_summary": "", "exercise_recommendation": []}
        except Exception as e:
            print(f"⚠️ RAG analysis failed: {e}")
            rag_result = {"diagnosis_summary": "", "exercise_recommendation": []}

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
            
            response_data = {
                "front": {
                    "metrics": clean_for_json(front_metrics), 
                    "metric_status": clean_for_json(front_metric_status), 
                    "flags": clean_for_json(front_flags_all), 
                    "squat_reps": int(reps_front),
                    "video_overlay_path": str(front_video)
                },
                "side": {
                    "metrics": clean_for_json(side_metrics), 
                    "metric_status": clean_for_json(side_metric_status), 
                    "flags": clean_for_json(side_flags_all), 
                    "squat_reps": int(reps_side),
                    "video_overlay_path": str(side_video)
                },
                "back": {
                    "metrics": clean_for_json(back_metrics), 
                    "metric_status": clean_for_json(back_metric_status), 
                    "flags": clean_for_json(back_flags_all), 
                    "squat_reps": int(reps_back),
                    "video_overlay_path": str(back_video)
                },
                "ai_analysis": {
                    "diagnosis_summary": str(rag_result.get("diagnosis_summary", "Analysis completed")),
                    "exercise_recommendations": clean_for_json(rag_result.get("exercise_recommendation", ["Focus on proper squat form"]))
                },
                "processing_info": {
                    "processing_time_seconds": round(processing_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "completed"
                }
            }
            
            # Validate response data before returning
            if not isinstance(response_data, dict):
                raise ValueError("Response data is not a dictionary")
            
            # Check if all required fields exist
            required_fields = ["front", "side", "back", "ai_analysis"]
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
            print(f"   - AI analysis: {len(response_data['ai_analysis']['diagnosis_summary'])} chars")
            
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
            # Return a minimal valid response
            minimal_response = {
                "front": {"metrics": {}, "flags": {}, "squat_reps": 0, "video_overlay_path": ""},
                "side": {"metrics": {}, "flags": {}, "squat_reps": 0, "video_overlay_path": ""},
                "back": {"metrics": {}, "flags": {}, "squat_reps": 0, "video_overlay_path": ""},
                "ai_analysis": {
                    "diagnosis_summary": "Analysis completed with basic metrics",
                    "exercise_recommendations": ["Focus on proper squat form", "Practice regularly", "Consider professional guidance"]
                },
                "processing_info": {
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "completed_with_fallback"
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
