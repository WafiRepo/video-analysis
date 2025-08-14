import streamlit as st
import cv2
import time
import zipfile
import tempfile
import imageio
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import logging

logging.basicConfig(filename='video_estimation.log', level=logging.INFO, format='%(asctime)s %(message)s')

def _angle_at_joint(a, b, c):
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return None
    cosine_angle = np.dot(ba, bc) / max(1e-9, denom)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)

def _map_points_to_squat_landmarks(points, frame_shape):
    """Peta list keypoints -> dict landmark bernama untuk analisis squat.
    Format nilai: (x,y,z,score); z=0.0, score=1.0 jika tersedia, hanya titik yang ada dimasukkan.
    """
    result = {}
    if points is None:
        return result
    n = len(points)
    def add(name, idx):
        if 0 <= idx < n and points[idx] is not None:
            x, y = points[idx]
            result[name] = (float(x), float(y), 0.0, 1.0)
    # MediaPipe 33 kp
    if n == 33:
        add('LSHO', 11); add('RSHO', 12)
        add('LHIP', 23); add('RHIP', 24)
        add('LKNE', 25); add('RKNE', 26)
        add('LANK', 27); add('RANK', 28)
        add('LTOE', 31); add('RTOE', 32)
    # MoveNet 17 kp
    elif n == 17:
        add('LSHO', 5); add('RSHO', 6)
        add('LHIP', 11); add('RHIP', 12)
        add('LKNE', 13); add('RKNE', 14)
        add('LANK', 15); add('RANK', 16)
        # toe tidak tersedia di MoveNet single-pose
    # OpenPose (COCO) ~19 kp
    else:
        add('RSHO', 2); add('LSHO', 5)
        add('RHIP', 8); add('RKNE', 9); add('RANK', 10)
        add('LHIP', 11); add('LKNE', 12); add('LANK', 13)
    return result

def generate_report(original_image, pose_image, metrics_df):
    """
    Generate a ZIP report containing the original image, processed pose image, and body metrics CSV.
    """
    report_zip = BytesIO()
    with zipfile.ZipFile(report_zip, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        with BytesIO() as buffer:
            Image.fromarray(original_image).save(buffer, format='PNG')
            zip_file.writestr("original_image.png", buffer.getvalue())
        with BytesIO() as buffer:
            # Convert BGR to RGB for correct color representation
            rgb_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb_image).save(buffer, format='PNG')
            zip_file.writestr("pose_image.png", buffer.getvalue())
        csv_data = metrics_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr("body_metrics.csv", csv_data)
    report_zip.seek(0)
    return report_zip

def save_session(metrics, session_type):
    """
    Save the current session metrics into Streamlit's session state.
    """
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    session_data = {
        "session_type": session_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    }
    st.session_state.session_history.append(session_data)
    st.success("Session saved successfully!")

def share_session():
    """
    Stub function to indicate session sharing.
    """
    st.success("Session shared to cloud!")

def run_video_estimation(analyzer, video_file, threshold, record_video=False, extract_skeleton=False, compute_builtin_metrics=True, ui_mode=True):
    """
    Process an uploaded video file using the given analyzer.
    
    - analyzer: Your pose estimation analyzer instance.
    - video_file: The uploaded video file.
    - threshold: Confidence threshold for keypoint detection.
    - record_video: If True, record and offer download of the processed video.
    - extract_skeleton: If True, extract a skeleton-only video.
    
    The function displays frames in real time and updates real-time metrics.
    Additionally, it records metrics (with frame number and elapsed time) and offers a CSV download.
    """
    # Create a temporary file to store the uploaded video.
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty() if ui_mode else None  # Placeholder to update video frames in real time
    recorded_frames = []
    skeleton_frames = []
    metrics_placeholder = st.empty() if compute_builtin_metrics else None  # Placeholder for real-time metrics display
    
    # Try to obtain total frame count for progress estimation
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    progress_bar = (st.progress(0) if (ui_mode and total_frames > 0) else None)
    frame_count = 0

    if ui_mode:
        st.info("Processing video...")
    start_time = time.time()
    last_points = None
    
    # List to store metrics for each frame
    metrics_list = []

    no_pose_detected = True
    frame_idx = 0
    last_valid_metrics = {}
    # Process video frame-by-frame
    pose_frames = []
    # Rep counter (berbasis sudut lutut rata-rata dengan hysteresis)
    rep_count = 0
    phase = "up"  # up -> turun; down -> naik
    knee_angle_ema = None
    alpha_ema = 0.2
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for faster processing (adjust as needed)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        try:
            # Detect pose keypoints
            points = analyzer.detect_pose(frame, threshold)
            if points is not None and any(p is not None for p in points):
                no_pose_detected = False
                num_points = sum([1 for p in points if p is not None])
                logging.info(f"Frame {frame_idx}: Pose detected, {num_points} keypoints.")
            else:
                logging.info(f"Frame {frame_idx}: No pose detected.")
            last_points = points
            
            # Draw pose overlay on a copy of the frame
            overlay_frame = analyzer.draw_pose(frame.copy(), points, threshold)
            frame_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Frame {frame_idx}: Error during frame processing: {e}")
            if ui_mode:
                st.error(f"Error during frame processing: {e}")
            continue
        
        # Update the frame display in real time
        if ui_mode and frame_placeholder is not None:
            frame_placeholder.image(frame_rgb, channels="RGB")
        recorded_frames.append(frame_rgb)
        
        # Map ke landmark bernama untuk analisis squat dan simpan
        named = _map_points_to_squat_landmarks(points, frame.shape)
        pose_frames.append(named)

        # Repetition counting via knee angles (gunakan rata-rata L/R jika ada)
        left_knee = None
        right_knee = None
        if all(k in named for k in ["LHIP","LKNE","LANK"]):
            left_knee = _angle_at_joint((named['LHIP'][0], named['LHIP'][1]),
                                        (named['LKNE'][0], named['LKNE'][1]),
                                        (named['LANK'][0], named['LANK'][1]))
        if all(k in named for k in ["RHIP","RKNE","RANK"]):
            right_knee = _angle_at_joint((named['RHIP'][0], named['RHIP'][1]),
                                         (named['RKNE'][0], named['RKNE'][1]),
                                         (named['RANK'][0], named['RANK'][1]))
        knee_angles = [ang for ang in [left_knee, right_knee] if ang is not None]
        if knee_angles:
            knee_angle = float(sum(knee_angles) / len(knee_angles))
            if knee_angle_ema is None:
                knee_angle_ema = knee_angle
            else:
                knee_angle_ema = alpha_ema * knee_angle + (1 - alpha_ema) * knee_angle_ema
            # Hysteresis thresholds (atur sesuai data): bottom < 100°, top > 160°
            if phase == "up" and knee_angle_ema < 100.0:
                phase = "down"  # mencapai bawah
            elif phase == "down" and knee_angle_ema > 160.0:
                rep_count += 1
                phase = "up"

        # If skeleton extraction is enabled, process the skeleton image
        if extract_skeleton:
            if hasattr(analyzer, 'draw_skeleton'):
                skeleton_frame = analyzer.draw_skeleton(points, frame.shape)
            else:
                blank = np.zeros_like(frame)
                skeleton_frame = analyzer.draw_pose(blank.copy(), points, threshold)
            skeleton_frames.append(skeleton_frame)
        
        # Update and display real-time metrics (opsional)
        if compute_builtin_metrics:
            metrics = analyzer.calculate_body_metrics(points)
            if metrics:  # Simpan metrics terakhir yang valid
                last_valid_metrics = metrics
            if ui_mode and metrics_placeholder is not None:
                metrics_placeholder.markdown("**Realtime Metrics:** " + str(metrics))
        
        # Record metrics along with frame number and elapsed time
        current_time = time.time() - start_time
        row = {"frame": frame_count, "time": current_time}
        if compute_builtin_metrics:
            # 'metrics' only exists when compute_builtin_metrics=True
            row.update(metrics)
        metrics_list.append(row)
        
        # Update progress bar if total frame count is available
        frame_count += 1
        if ui_mode and progress_bar:
            progress_bar.progress(min(int((frame_count / total_frames) * 100), 100))
        
        # Small delay to simulate real-time processing (adjust if needed)
        time.sleep(0.03)
        frame_idx += 1
    
    cap.release()
    elapsed_time = time.time() - start_time
    if ui_mode:
        st.success(f"Video processing complete. Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
    
    rec_video_bytes = None
    skel_video_bytes = None
    
    # Create a DataFrame for the per-frame metrics & CSV download (opsional)
    if ui_mode and compute_builtin_metrics:
        metrics_df = pd.DataFrame(metrics_list)
        csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Metrics CSV",
                           data=csv_metrics,
                           file_name="video_metrics.csv",
                           mime="text/csv")
    
    # If recording is enabled, write the processed video to file for download
    if record_video and recorded_frames:
        video_filename = "recorded_pose_video.mp4"
        imageio.mimwrite(video_filename, recorded_frames, fps=30, codec='libx264')
        with open(video_filename, "rb") as vid_file:
            rec_video_bytes = vid_file.read()
        if ui_mode:
            st.download_button(label="Download Recorded Video", 
                               data=rec_video_bytes, 
                               file_name=video_filename, 
                               mime="video/mp4")
    
    # If skeleton extraction is enabled, write the skeleton video to file for download
    if ui_mode and extract_skeleton and skeleton_frames:
        skeleton_filename = "extracted_skeleton_video.mp4"
        imageio.mimwrite(skeleton_filename, skeleton_frames, fps=30, codec='libx264')
        with open(skeleton_filename, "rb") as vid_file:
            skel_video_bytes = vid_file.read()
        st.download_button(label="Download Skeleton Video", 
                           data=skel_video_bytes, 
                           file_name=skeleton_filename, 
                           mime="video/mp4")
    
    metrics_final = (last_valid_metrics if compute_builtin_metrics else {}) or {}
    # Sisipkan hasil tambahan
    metrics_final["pose_frames"] = pose_frames
    metrics_final["squat_reps"] = rep_count
    if ui_mode and no_pose_detected:
        logging.warning("Tidak ada pose yang terdeteksi pada seluruh video.")
        st.warning("Tidak ada pose yang terdeteksi pada video. Pastikan video cukup terang dan tubuh terlihat jelas.")
    elif ui_mode and not metrics_final:
        logging.info("Video berhasil diproses, namun tidak ada metrics yang dapat dihitung dari pose terakhir.")
        st.info("Video berhasil diproses, namun tidak ada metrics yang dapat dihitung dari pose terakhir.")
    return rec_video_bytes, skel_video_bytes, metrics_final
