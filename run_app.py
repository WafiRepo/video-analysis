import subprocess
import sys
import os

def main():
    """Jalankan aplikasi Streamlit untuk analisis squat"""
    
    # Pastikan kita berada di directory yang benar
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, "src")
    
    if not os.path.exists(src_dir):
        print(f"Error: Directory 'src' tidak ditemukan di {script_dir}")
        return
    
    # Cek apakah file app.py ada
    app_file = os.path.join(src_dir, "app.py")
    if not os.path.exists(app_file):
        print(f"Error: File 'app.py' tidak ditemukan di {src_dir}")
        return
    
    print("🚀 Menjalankan HOLOWELLNESS EXPERT POSTURE ANALYSIS TOOL...")
    print(f"📁 Working directory: {src_dir}")
    print("🌐 Aplikasi akan terbuka di browser...")
    print("⏹️  Tekan Ctrl+C untuk menghentikan aplikasi")
    print("-" * 60)
    
    try:
        # Jalankan Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ], cwd=src_dir)
    except KeyboardInterrupt:
        print("\n🛑 Aplikasi dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error menjalankan aplikasi: {e}")
        print("\n💡 Pastikan semua dependencies terinstall:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
