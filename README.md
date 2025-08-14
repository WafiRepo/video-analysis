# HOLOWELLNESS EXPERT POSTURE ANALYSIS TOOL

Aplikasi analisis postur squat berbasis pose estimation dengan deteksi otomatis repetisi dan analisis biomekanik. Tersedia dalam dua versi: **Streamlit UI** dan **FastAPI Service**.

## 🚀 Cara Menjalankan Aplikasi

### 🖥️ Streamlit UI (Interactive Web App)

#### Metode 1: Jalankan Langsung (Recommended)
```bash
python run_app.py
```

#### Metode 2: Manual
```bash
cd src
streamlit run app.py
```

#### Metode 3: Dari Root Directory
```bash
streamlit run src/app.py
```

### 🔌 FastAPI Service (REST API)

#### Metode 1: Jalankan Langsung
```bash
python fastapi_app.py
```

#### Metode 2: Dengan Uvicorn
```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

## 📋 Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Pastikan Model Files Tersedia
- `models/movenet_lightning_fp16.tflite` - untuk MoveNet
- `models/graph_opt.pb` - untuk OpenPose

## 🎯 Fitur Utama

### Analisis Squat Otomatis
- **Deteksi Repetisi**: Menghitung jumlah squat secara otomatis
- **Pose Estimation**: Support MediaPipe, MoveNet, dan OpenPose
- **Metrik Biomekanik**: 
  - Trunk lean, thorax side bend, pelvis drop
  - Knee/hip flexion, ankle dorsiflexion
  - Foot external rotation, knee valgus
  - Squat depth, weight distribution

### Rule-based Flags
- Knee dominant vs Hip dominant
- Trunk lean anterior
- Thorax side bend (kiri/kanan)
- Foot external rotation
- Knee valgus
- Insufficient ankle dorsiflexion
- Thoracolumbar hyperextension
- Insufficient depth
- Weight bearing asymmetry

### 🤖 AI-Powered Analysis (RAG Integration)
- **Diagnosis Summary**: Analisis komprehensif form squat dari AI expert
- **Exercise Recommendations**: Rekomendasi latihan spesifik untuk perbaikan
- **Professional Insights**: Analisis biomekanik tingkat expert
- **Personalized Guidance**: Rekomendasi berdasarkan metrik individual

## 📹 Cara Penggunaan

### 🖥️ Streamlit UI
1. **Pilih Model**: MediaPipe (Recommended), MoveNet, atau OpenPose
2. **Set Threshold**: Confidence threshold 0.1 - 1.0 (disarankan 0.5-0.7)
3. **Upload Video**: Format MP4, AVI, MOV, GIF (minimal 5 repetisi squat)
4. **Pilih Opsi**: Record processed video, extract skeleton video
5. **Lihat Hasil**: Video dengan pose overlay, metrik, dan flags
6. **🤖 AI Analysis**: Klik "Get AI Analysis" untuk diagnosis dan rekomendasi

### 🔌 FastAPI Service

#### Endpoint: `POST /squat-analysis`

**Input:**
- `front`: Video file (Front view)
- `side`: Video file (Side view)
- `back`: Video file (Back view)

**Output:**
```json
{
  "front": {
    "metrics": {...},
    "flags": {...},
    "squat_reps": 5,
    "video_overlay_path": "overlay_video_output/front_overlay_1703123456.mp4"
  },
  "side": {...},
  "back": {...},
  "ai_analysis": {
    "diagnosis_summary": "Comprehensive analysis of squat form...",
    "exercise_recommendations": [
      "Focus on maintaining neutral spine during descent",
      "Improve ankle dorsiflexion with calf stretches",
      "Address knee valgus with hip abduction exercises"
    ]
  }
}
```

**Fitur FastAPI:**
- **Model Fixed**: MediaPipe (tidak bisa diubah)
- **Threshold Fixed**: 0.45 (tidak bisa diubah)
- **Video Overlay**: Otomatis tersimpan di folder `overlay_video_output/`
- **Multi-Angle**: Analisis 3 angle sekaligus (Front, Side, Back)
- **🤖 AI Analysis**: Otomatis mendapatkan diagnosis dan rekomendasi dari RAG chatbot

## 🤖 AI-Powered Analysis (RAG Integration)

### Fitur RAG Chatbot
Aplikasi ini terintegrasi dengan **RAG (Retrieval-Augmented Generation)** chatbot untuk memberikan analisis AI yang profesional:

#### **Diagnosis Summary**
- Analisis komprehensif form squat berdasarkan metrik biomekanik
- Identifikasi masalah form dan kompensasi otot
- Evaluasi kualitas gerakan dan risiko cedera

#### **Exercise Recommendations**
- Rekomendasi latihan spesifik untuk perbaikan
- Progresi latihan yang aman dan efektif
- Fokus pada keseimbangan otot dan mobilitas

#### **How It Works**
1. **Data Collection**: Metrik squat dan flags dikumpulkan dari 3 angle
2. **Prompt Building**: Data diformat menjadi prompt terstruktur
3. **AI Analysis**: RAG chatbot menganalisis dan memberikan insight
4. **Response Parsing**: Diagnosis dan rekomendasi diparse dan ditampilkan

#### **Configuration**
```bash
# Environment variables (optional)
RAG_ENDPOINT=http://15.152.146.34/api/chat
USER_ID=60d5ec49e472e3a8e4e1d3b4
```

### Streamlit UI
- **Button "Get AI Analysis"** muncul setelah video diproses
- **Real-time Analysis** dengan progress indicator
- **Save to Session** untuk menyimpan hasil AI analysis

### FastAPI Service
- **Automatic RAG Analysis** pada setiap request squat analysis
- **Structured Response** dengan diagnosis dan rekomendasi
- **Error Handling** untuk RAG service yang tidak tersedia
- **Video Analysis Only**: Fokus pada analisis video squat, tidak ada analisis gambar statis

## 🔧 Troubleshooting

### Error: "Pose model failed to load"
- Pastikan path model benar
- Download model yang sesuai

### Error: "Tidak ada pose yang terdeteksi"
- Turunkan confidence threshold
- Pastikan video cukup terang
- Coba model lain

### Error: "Repetisi < 5"
- Upload video dengan lebih banyak repetisi
- Pastikan gerakan squat jelas dan lengkap

### Error: "ModuleNotFoundError: No module named 'modules'"
- Jalankan dari root directory: `python run_app.py`
- Atau jalankan: `streamlit run src/app.py`

## 📁 Struktur File

```
Posture Analysis/
├── fastapi_app.py          # FastAPI service
├── run_app.py              # Script untuk menjalankan Streamlit
├── requirements.txt         # Dependencies
├── README.md              # Dokumentasi ini
├── CLEANUP_SUMMARY.md     # Summary cleanup project
├── overlay_video_output/   # Output video overlay (FastAPI)
│   └── .gitkeep
├── models/                # Model files
│   ├── movenet_lightning_fp16.tflite
│   └── graph_opt.pb
└── src/
    ├── app.py             # Aplikasi Streamlit utama
    └── modules/
        ├── __init__.py
        ├── pose_estimators.py    # Pose estimation models
        ├── video_estimation.py   # Video processing
        ├── squat_analysis.py     # Analisis squat custom
        ├── helpers.py            # Helper functions
        └── config.py             # Streamlit configuration
```

## 🎥 Contoh Video yang Cocok

- **Posisi**: Frontal view (depan) atau lateral view (samping)
- **Durasi**: Minimal 5 repetisi squat
- **Kualitas**: Pencahayaan cukup, tubuh terlihat jelas
- **Gerakan**: Squat standar (dari berdiri → jongkok → berdiri)

## 💾 Simpan Hasil

### Streamlit UI
Klik tombol **"Save metrics"** untuk menyimpan hasil analisis ke session history aplikasi.

### FastAPI
Video overlay otomatis tersimpan di folder `overlay_video_output/` dengan nama unik berdasarkan timestamp.

## 🧹 Project Cleanup

Project telah dibersihkan dari file-file yang tidak diperlukan:
- **Total file yang dihapus**: ~15 files + 2 cache directories
- **Total ukuran yang dibersihkan**: ~7.5MB
- **Struktur project**: Sangat lean dan efisien

## 🆘 Support

Jika ada masalah, pastikan:
1. Semua dependencies terinstall
2. Model files tersedia
3. Video sesuai kriteria
4. Jalankan dari directory yang benar

## 📊 API Documentation

Setelah menjalankan FastAPI, buka browser dan kunjungi:
- **API Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

**Happy Analyzing! 🏋️‍♂️📊**