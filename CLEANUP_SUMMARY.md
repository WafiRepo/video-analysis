# Cleanup Summary - FastAPI & Streamlit Project

## Files Removed (First Cleanup):
- `recorded_pose_video.mp4` (3.8MB) - Recording result video
- `extracted_skeleton_video.mp4` (1.5MB) - Extracted skeleton video
- `video_estimation.log` (799KB) - Log file
- `packages.txt` - Unnecessary package file
- `src/fastapi_app.py` - Duplicate file (already exists in root)
- `src/video_estimation.log` (309KB) - Log file in src
- `src/extracted_skeleton_video.mp4` (306KB) - Skeleton video in src
- `src/recorded_pose_video.mp4` (349KB) - Recording video in src
- `src/Coklat Tua & Biru Tua  - Logo 2 Fix.png` (264KB) - Unnecessary logo
- `__pycache__/` - Python cache (root and src)
- `src/__pycache__/` - Python cache in src

## Files Removed (Second Cleanup - Modules):
- `src/modules/webcam_transformers.py` (12KB) - Not used in main application
- `src/modules/session_history.py` (811B) - Only used for session history (not critical)
- `src/modules/comparison_mode.py` (3.1KB) - Not used in main application
- `src/modules/image_analysis.py` (10KB) - Not used in main application

## Remaining Files (Only Essential Ones):

### Root Directory:
- `fastapi_app.py` - FastAPI application
- `README.md` - Project documentation
- `run_app.py` - **IMPORTANT**: Main script to run Streamlit
- `requirements.txt` - Python dependencies
- `src/` - Source code directory
- `models/` - Pose estimation models
- `assets/` - Video assets for testing
- `.git/` - Git repository
- `.cursor/` - Cursor IDE config
- `.devcontainer/` - Dev container config

### src/ Directory:
- `app.py` - Streamlit application

### src/modules/ Directory (Only Essential):
- `__init__.py` - Package marker
- `video_estimation.py` - **IMPORTANT**: Core video processing
- `squat_analysis.py` - **IMPORTANT**: Custom squat analysis algorithm
- `pose_estimators.py` - **IMPORTANT**: Pose estimation models
- `helpers.py` - **IMPORTANT**: Biomechanical calculations
- `config.py` - **IMPORTANT**: Streamlit configuration

## Total Cleanup Results:
- **Total files removed**: ~15 files + 2 cache directories
- **Total size cleaned**: ~7.5MB
- **Project structure**: Very clean, only contains truly necessary files

## File Dependencies:
- `video_estimation.py` → `pose_estimators.py`, `squat_analysis.py`
- `pose_estimators.py` → `helpers.py`
- `app.py` → `video_estimation.py`, `squat_analysis.py`
- `fastapi_app.py` → `video_estimation.py`, `squat_analysis.py`

## How to Run:

### FastAPI:
```bash
python fastapi_app.py
# or
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

### Streamlit:
```bash
python run_app.py
# or
streamlit run src/app.py
```

## FastAPI Response Example:

### Success Response Structure:
```json
{
  "front": {
    "metrics": {
      "thorax_side_bend_max_deg": 12.5,
      "pelvis_drop_deg_at_depth": 3.2,
      "foot_ER_deg_L_at_depth": 25.8,
      "foot_ER_deg_R_at_depth": 24.1,
      "knee_valgus_deg_L_at_depth": 2.1,
      "knee_valgus_deg_R_at_depth": 1.8,
      "com_shift_ratio_right": 0.52
    },
    "flags": {
      "thorax_side_bend_right": false,
      "thorax_side_bend_left": false,
      "right_foot_ER": true,
      "left_foot_ER": true,
      "left_knee_valgus": false,
      "right_knee_valgus": false,
      "weight_bearing_right": false,
      "weight_bearing_left": false
    },
    "squat_reps": 5,
    "video_overlay_path": "overlay_video_output/front_overlay_1703123456.mp4"
  },
  "side": {
    "metrics": {
      "trunk_lean_max_deg": 28.3,
      "knee_flex_max_deg_L": 125.7,
      "knee_flex_max_deg_R": 126.2,
      "hip_flex_max_deg_L": 98.4,
      "hip_flex_max_deg_R": 97.8,
      "ankle_dorsi_deg_L_at_depth": 18.2,
      "ankle_dorsi_deg_R_at_depth": 17.9,
      "squat_depth_thigh_deg": 8.5
    },
    "flags": {
      "knee_dominant": true,
      "hip_dominant": false,
      "trunk_leans_anterior": true,
      "insufficient_right_ankle_dorsi": false,
      "insufficient_left_ankle_dorsi": false,
      "thoracolumbar_hyperextension": false,
      "insufficient_depth_parallel": false
    },
    "squat_reps": 5,
    "video_overlay_path": "overlay_video_output/side_overlay_1703123456.mp4"
  },
  "back": {
    "metrics": {
      "thorax_side_bend_max_deg": 11.8,
      "pelvis_drop_deg_at_depth": 2.9,
      "knee_valgus_deg_L_at_depth": 1.9,
      "knee_valgus_deg_R_at_depth": 2.3
    },
    "flags": {
      "thorax_side_bend_right": false,
      "thorax_side_bend_left": false,
      "left_knee_valgus": false,
      "right_knee_valgus": false
    },
    "squat_reps": 5,
    "video_overlay_path": "overlay_video_output/back_overlay_1703123456.mp4"
  }
}
```

### Error Response Example:
```json
{
  "error": "MediaPipe model failed to load."
}
```

## Notes:
- All unused files have been removed
- Only core functionality for FastAPI and Streamlit remains
- Project is now very lean and efficient
- Video overlay files are saved in `overlay_video_output/` folder with unique timestamps
- Model: MediaPipe (fixed), Threshold: 0.45 (fixed)
