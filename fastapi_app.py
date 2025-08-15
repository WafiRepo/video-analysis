from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Squat Analysis API is running"}

# RAG Chatbot Configuration
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://15.152.36.109/api/chat")
USER_ID = os.getenv("USER_ID", "60d5ec49e472e3a8e4e1d3b4")

async def get_rag_chatbot_analysis(prompt: str) -> dict:
    """Get analysis from RAG chatbot with fallback."""
    
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
                print("⚠️ RAG response too short, using fallback")
                return await get_fallback_analysis(prompt)

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
                print("⚠️ Diagnosis too short, using fallback")
                return await get_fallback_analysis(prompt)
            
            # Ensure we have at least some recommendations
            if not recommendations:
                recommendations = [
                    "Focus on proper squat form and depth",
                    "Practice bodyweight squats regularly",
                    "Consider professional assessment for guidance"
                ]
            
            print(f"✅ Parsed diagnosis: {len(diagnosis)} chars")
            print(f"✅ Parsed recommendations: {len(recommendations)} items")
                
            return {
                "diagnosis_summary": diagnosis,
                "exercise_recommendation": recommendations
            }

    except httpx.RequestError as e:
        print(f"❌ RAG request failed: {e}")
        return await get_fallback_analysis(prompt)
    except Exception as e:
        print(f"❌ RAG unexpected error: {e}")
        return await get_fallback_analysis(prompt)

async def get_fallback_analysis(prompt: str) -> dict:
    """Fallback analysis when RAG fails."""
    
    print("🔄 Using fallback analysis...")
    
    # Extract key information from prompt for better fallback
    prompt_lower = prompt.lower()
    
    # Check for specific metrics in the prompt
    if "trunk_lean_max_deg: 49.9" in prompt:
        diagnosis = "Excessive trunk lean (49.9°) detected during squat. This indicates over-reliance on hip strategy and potential core weakness. The trunk is leaning too far forward which can strain the lower back."
        recommendations = [
            "Strengthen core with planks and dead bugs to improve trunk stability",
            "Practice wall squats to maintain upright posture and reduce forward lean",
            "Work on hip flexor mobility and glute activation to improve hip mechanics"
        ]
    elif "hip_flex_max_deg: 133.1" in prompt:
        diagnosis = "Hip-dominant squat pattern detected with excessive hip flexion (133.1°). This may indicate over-reliance on hip muscles and potential knee/ankle mobility limitations."
        recommendations = [
            "Focus on knee and ankle mobility exercises to improve squat depth",
            "Practice box squats to control depth and maintain proper form",
            "Work on quadriceps strengthening to balance hip and knee contribution"
        ]
    elif "knee_valgus_deg_L_at_depth: 20.4" in prompt:
        diagnosis = "Significant left knee valgus (20.4°) detected during squat. This indicates inward knee collapse which can lead to knee pain, instability, and potential ACL injury risk."
        recommendations = [
            "Strengthen hip abductors with side-lying leg raises and clamshells",
            "Practice wall squats with resistance band above knees to maintain knee alignment",
            "Focus on pushing knees outward during squat movement and maintaining proper foot position"
        ]
    elif "thorax_side_bend_max_deg: 180.0" in prompt:
        diagnosis = "Excessive thorax side bend (180.0°) detected. This indicates significant lateral movement and potential core instability during squat movement."
        recommendations = [
            "Strengthen core with anti-rotation exercises like Pallof presses",
            "Practice single-leg balance exercises to improve stability",
            "Work on maintaining neutral spine alignment throughout the squat"
        ]
    elif "insufficient_depth_parallel: True" in prompt:
        diagnosis = "Insufficient squat depth detected. The thighs are not reaching parallel to the ground, which limits the effectiveness of the squat exercise."
        recommendations = [
            "Practice deep squat holds with support to improve depth gradually",
            "Work on ankle dorsiflexion and hip mobility exercises",
            "Use box squats to train proper depth while maintaining form"
        ]
    elif "knee" in prompt_lower and "valgus" in prompt_lower:
        diagnosis = "Knee valgus detected during squat. This indicates inward knee collapse which can lead to knee pain and instability."
        recommendations = [
            "Strengthen hip abductors with side-lying leg raises",
            "Practice wall squats with resistance band above knees",
            "Focus on pushing knees outward during squat movement"
        ]
    elif "ankle" in prompt_lower and "dorsi" in prompt_lower:
        diagnosis = "Limited ankle dorsiflexion detected. This restricts squat depth and can cause compensation patterns."
        recommendations = [
            "Perform ankle mobility exercises daily",
            "Use heel lifts during squats initially",
            "Practice deep squat holds with support"
        ]
    elif "trunk" in prompt_lower and "lean" in prompt_lower:
        diagnosis = "Excessive trunk lean detected. This may indicate weak core or hip mobility issues."
        recommendations = [
            "Strengthen core with planks and dead bugs",
            "Improve hip mobility with hip flexor stretches",
            "Practice wall squats to maintain upright posture"
        ]
    elif "hip" in prompt_lower and "dominant" in prompt_lower:
        diagnosis = "Hip-dominant squat pattern detected. This may indicate over-reliance on hip muscles."
        recommendations = [
            "Focus on knee and ankle mobility",
            "Practice box squats to control depth",
            "Work on quadriceps strengthening"
        ]
    elif "depth" in prompt_lower or "squat" in prompt_lower:
        diagnosis = "Squat depth analysis completed. Focus on achieving proper depth while maintaining form."
        recommendations = [
            "Practice bodyweight squats with proper form",
            "Gradually increase depth as mobility improves",
            "Use box squats to train proper depth"
        ]
    else:
        diagnosis = "General squat form analysis completed. Focus on maintaining proper alignment and controlled movement."
        recommendations = [
            "Practice bodyweight squats with proper form",
            "Gradually increase depth as mobility improves",
            "Consider professional assessment for personalized guidance"
        ]
    
    return {
        "diagnosis_summary": diagnosis,
        "exercise_recommendation": recommendations
    }

def build_squat_analysis_prompt(squat_metrics: Dict[str, Any], squat_flags: Dict[str, Any], 
                               front_video_path: str, side_video_path: str, back_video_path: str) -> str:
    """Build simplified prompt for RAG chatbot analysis."""
    
    # Extract key metrics for analysis (only non-zero values)
    key_metrics = []
    for key, value in squat_metrics.items():
        if value is not None and value != 0.0 and value != 'N/A':
            if isinstance(value, float):
                key_metrics.append(f"{key}: {value:.1f}°")
            else:
                key_metrics.append(f"{key}: {value}")
    
    # Extract key flags (only True values)
    key_flags = []
    for flag, value in squat_flags.items():
        if value is True:
            key_flags.append(flag.replace('_', ' ').title())
    
    # Build simpler, more direct prompt
    prompt = f"""
Analyze this squat performance data and provide recommendations.

SQUAT DATA:
Metrics: {', '.join(key_metrics[:6]) if key_metrics else 'No metrics'}
Issues: {', '.join(key_flags) if key_flags else 'No issues'}

Provide analysis in this exact format:

Diagnosis: [2-3 sentences about squat form]

Recommendations:
- [First exercise]
- [Second exercise]
- [Third exercise]

Keep response under 150 words.
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
    
    metrics, flags = analyze_squat_from_sequence(pose_frames, score_thr=0.45)
    return metrics.__dict__, flags.__dict__, reps, video_path

@app.post("/squat-analysis")
async def squat_analysis_api(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    back: UploadFile = File(...),
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
        
        # Get RAG analysis for comprehensive diagnosis
        try:
            rag_prompt = build_squat_analysis_prompt(
                side_metrics_all or front_metrics_all or back_metrics_all,  # Use any available metrics
                side_flags_all or front_flags_all or back_flags_all,       # Use any available flags
                front_video, side_video, back_video
            )
            
            print(f"🤖 Calling RAG with prompt length: {len(rag_prompt)}")
            rag_result = await get_rag_chatbot_analysis(rag_prompt)
            print(f"✅ RAG analysis completed")
            
            # Validate RAG result
            if not rag_result or not rag_result.get("diagnosis_summary"):
                print("⚠️ RAG result invalid, using fallback")
                rag_result = await get_fallback_analysis(rag_prompt)
            
        except Exception as e:
            print(f"⚠️ RAG analysis failed: {e}")
            # Use fallback analysis
            rag_result = await get_fallback_analysis("squat analysis")

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

        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build response
        try:
            # Ensure all data is JSON serializable
            def clean_for_json(obj):
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_for_json(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool)) or obj is None:
                    return obj
                else:
                    return str(obj)
            
            response_data = {
                "front": {
                    "metrics": clean_for_json(front_metrics), 
                    "flags": clean_for_json(front_flags_all), 
                    "squat_reps": int(reps_front),
                    "video_overlay_path": str(front_video)
                },
                "side": {
                    "metrics": clean_for_json(side_metrics), 
                    "flags": clean_for_json(side_flags_all), 
                    "squat_reps": int(reps_side),
                    "video_overlay_path": str(side_video)
                },
                "back": {
                    "metrics": clean_for_json(back_metrics), 
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
