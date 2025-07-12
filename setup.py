#!/usr/bin/env python3
"""
Setup script for YOLOv11 Emotion Recognition
This script installs all required dependencies and sets up the environment.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and print output"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    # Basic dependencies
    basic_deps = [
        "pip install --upgrade pip",
        "pip install ultralytics",
        "pip install opencv-python",
        "pip install torch torchvision",
        "pip install Pillow",
        "pip install numpy",
        "pip install ollama",
    ]
    
    for dep in basic_deps:
        if not run_command(dep):
            print(f"❌ Failed to install: {dep}")
            return False
    
    # Optional advanced dependencies
    print("\nInstalling optional dependencies for advanced features...")
    advanced_deps = [
        "pip install deepface",
        "pip install tensorflow",
        "pip install mediapipe",  # For eye tracking
        "pip install matplotlib",
        "pip install tqdm",
        "pip install PyYAML",
        "pip install requests",
        "pip install scipy"
    ]
    
    for dep in advanced_deps:
        if not run_command(dep):
            print(f"⚠️  Warning: Could not install {dep} - some features may not work")
    
    return True

def test_installation():
    """Test if installation was successful"""
    print("\nTesting installation...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV import failed")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError:
        print("❌ Ultralytics import failed")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   CUDA not available (will use CPU)")
    except ImportError:
        print("❌ PyTorch import failed")
        return False
    
    try:
        from deepface import DeepFace
        print("✅ DeepFace imported successfully (advanced emotion recognition available)")
    except ImportError:
        print("⚠️  DeepFace not available (will use basic emotion recognition)")
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully (eye tracking available)")
    except ImportError:
        print("⚠️  MediaPipe not available (eye tracking disabled in medical version)")
    
    return True

def main():
    """Main setup function"""
    print("=== YOLOv11 Emotion Recognition Setup ===\n")
    
    if not check_python_version():
        sys.exit(1)
    
    if not install_dependencies():
        print("❌ Installation failed!")
        sys.exit(1)
    
    if not test_installation():
        print("❌ Installation test failed!")
        sys.exit(1)
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now run the emotion recognition scripts:")
    print("  python emotion_recognition_simple.py   # Simple version (RECOMMENDED)")
    print("  python emotion_recognition_advanced.py # Advanced version with video support")
    print("  python emotion_recognition_medical.py  # Medical version with eye tracking")
    print("\nUsage examples:")
    print("  python emotion_recognition_advanced.py --video video.mp4  # Analyze video file")
    print("  python emotion_recognition_medical.py --model l          # Medical monitoring")
    print("  python emotion_recognition_medical.py --log session.json # With medical logging")
    print("\nControls:")
    print("  'q' - quit")
    print("  'c' - capture screenshot")
    print("  'm' - toggle print mode")
    print("  's' - toggle smoothing (advanced version only)")
    print("  'SPACE' - pause/resume (video files only)")
    print("\n🏥 Medical version includes:")
    print("  - Real-time eye tracking and gaze direction")
    print("  - Blink detection and counting")
    print("  - Medical-grade logging for patient monitoring")

if __name__ == "__main__":
    main() 