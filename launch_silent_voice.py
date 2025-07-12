#!/usr/bin/env python3
"""
Silent Voice Medical System Launcher
=====================================

Easy launcher for the Silent Voice medical monitoring system with intelligent decision engine.
Provides common configurations and quick start options.

Usage Examples:
  python launch_silent_voice.py                          # Webcam with decision engine
  python launch_silent_voice.py --video patient.mp4      # Video file analysis
  python launch_silent_voice.py --config icu_config      # ICU monitoring preset
  python launch_silent_voice.py --demo                   # Demo mode with test video
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

# Preset configurations for different medical scenarios
MEDICAL_PRESETS = {
    'default': {
        'description': 'Standard medical monitoring with decision engine',
        'args': ['--silent-voice', '--model', 'x'],
        'config': {
            "min_time_between_calls": 30.0,
            "critical_override_time": 10.0,
            "sustained_emotion_time": 45.0,
            "max_calls_per_session": 20,
            "enable_cost_optimization": True,
            "enable_medical_rules": True,
            "debug_mode": False
        }
    },
    'icu': {
        'description': 'ICU monitoring - more sensitive thresholds',
        'args': ['--silent-voice', '--model', 'x', '--patient-condition', 'ICU patient (critical care)', '--context', 'ICU room'],
        'config': {
            "min_time_between_calls": 20.0,
            "critical_override_time": 5.0,
            "sustained_emotion_time": 30.0,
            "max_calls_per_session": 30,
            "enable_cost_optimization": True,
            "enable_medical_rules": True,
            "debug_mode": False
        }
    },
    'als': {
        'description': 'ALS patient monitoring - optimized for subtle expressions',
        'args': ['--silent-voice', '--model', 'x', '--patient-condition', 'ALS patient (advanced)', '--context', 'hospital bed'],
        'config': {
            "min_time_between_calls": 25.0,
            "critical_override_time": 8.0,
            "sustained_emotion_time": 40.0,
            "max_calls_per_session": 25,
            "enable_cost_optimization": True,
            "enable_medical_rules": True,
            "debug_mode": False
        }
    },
    'stroke': {
        'description': 'Stroke patient monitoring - enhanced asymmetry detection',
        'args': ['--silent-voice', '--model', 'x', '--patient-condition', 'Stroke patient (post-acute)', '--context', 'rehabilitation ward'],
        'config': {
            "min_time_between_calls": 35.0,
            "critical_override_time": 12.0,
            "sustained_emotion_time": 50.0,
            "max_calls_per_session": 15,
            "enable_cost_optimization": True,
            "enable_medical_rules": True,
            "debug_mode": False
        }
    },
    'demo': {
        'description': 'Demo mode - frequent updates for demonstration',
        'args': ['--silent-voice', '--model', 'm', '--patient-condition', 'Demo patient', '--context', 'demonstration'],
        'config': {
            "min_time_between_calls": 15.0,
            "critical_override_time": 5.0,
            "sustained_emotion_time": 20.0,
            "max_calls_per_session": 50,
            "enable_cost_optimization": True,
            "enable_medical_rules": True,
            "debug_mode": True
        }
    }
}

def create_config_file(config_data, config_name="custom"):
    """Create a custom configuration file"""
    config_filename = f"gemma_decision_config_{config_name}.json"
    with open(config_filename, 'w') as f:
        json.dump(config_data, f, indent=2)
    return config_filename

def print_banner():
    """Print Silent Voice banner"""
    print("🗣️" * 25)
    print("🗣️  SILENT VOICE MEDICAL SYSTEM LAUNCHER  🗣️")
    print("🗣️" * 25)
    print()
    print("🧠 Intelligent AI Communication for Paralysis Patients")
    print("💰 Cost-Optimized with Smart Decision Engine")
    print("🏥 Medical-Grade Monitoring & Analysis")
    
    # Check for YOLO emotion models
    emotion_models = []
    for size in ['n', 's', 'm', 'l', 'x']:
        if os.path.exists(f'yolo11{size}_emotions.pt'):
            emotion_models.append(size.upper())
    
    if emotion_models:
        print(f"🎭 YOLO Emotion Detection Available: {', '.join(emotion_models)}")
    
    # Check for visual analysis capability
    try:
        import ollama
        print("📸 Visual Scene Analysis: Available (Ollama detected)")
    except ImportError:
        print("📸 Visual Scene Analysis: Not available (install Ollama)")
    
    print()

def list_presets():
    """List available medical presets"""
    print("📋 AVAILABLE MEDICAL PRESETS:")
    print("=" * 50)
    for preset_name, preset_config in MEDICAL_PRESETS.items():
        print(f"🏥 {preset_name.upper()}:")
        print(f"   {preset_config['description']}")
        
        # Show key configuration highlights
        config = preset_config['config']
        print(f"   • Timing: {config['min_time_between_calls']}s standard, {config['critical_override_time']}s critical")
        print(f"   • Budget: {config['max_calls_per_session']} calls/session")
        print(f"   • Debug: {'ON' if config['debug_mode'] else 'OFF'}")
        print()

def launch_system(preset='default', video_file=None, webcam_index=0, log_file=None, 
                 silent_voice_model=None, custom_args=None):
    """Launch the Silent Voice medical system"""
    
    # Get preset configuration
    if preset not in MEDICAL_PRESETS:
        print(f"❌ Error: Unknown preset '{preset}'")
        print("Available presets:", list(MEDICAL_PRESETS.keys()))
        return False
    
    preset_config = MEDICAL_PRESETS[preset]
    
    print(f"🚀 LAUNCHING: {preset.upper()} configuration")
    print(f"📖 Description: {preset_config['description']}")
    print()
    
    # Create custom config file for this session
    config_filename = create_config_file(preset_config['config'], preset)
    print(f"⚙️  Created config: {config_filename}")
    
    # Build command arguments
    cmd = ['python', 'emotion_recognition_medical.py']
    
    # Add preset arguments
    cmd.extend(preset_config['args'])
    
    # Add video source
    if video_file:
        if not os.path.exists(video_file):
            print(f"❌ Error: Video file '{video_file}' not found")
            return False
        cmd.extend(['--video', video_file])
        print(f"📹 Video source: {video_file}")
    else:
        cmd.extend(['--webcam', str(webcam_index)])
        print(f"📷 Webcam source: index {webcam_index}")
    
    # Add log file
    if not log_file:
        # Create log directory if it doesn't exist
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"📁 Created log directory: {log_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if video_file:
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            log_file = os.path.join(log_dir, f"silent_voice_log_{preset}_{video_name}_{timestamp}.json")
        else:
            log_file = os.path.join(log_dir, f"silent_voice_log_{preset}_webcam_{timestamp}.json")
    
    cmd.extend(['--log', log_file])
    print(f"📄 Log file: {log_file}")
    
    # Add Silent Voice model if provided
    if silent_voice_model:
        if os.path.exists(silent_voice_model):
            cmd.extend(['--silent-voice-model', silent_voice_model])
            print(f"🧠 AI Model: {silent_voice_model}")
        else:
            print(f"⚠️  Warning: Silent Voice model '{silent_voice_model}' not found - using fallback responses")
    else:
        print("🧠 Using Ollama with finetuned Gemma 3B model (hf.co/0xroyce/silent-voice-multimodal)")
    
    # Add any custom arguments
    if custom_args:
        cmd.extend(custom_args)
        print(f"🔧 Custom args: {' '.join(custom_args)}")
    
    print()
    print("🏁 STARTING SILENT VOICE MEDICAL SYSTEM...")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print()
    
    try:
        # Launch the system
        result = subprocess.run(cmd, check=True)
        print()
        print("✅ Silent Voice session completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Silent Voice system: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Silent Voice session interrupted by user")
        return True
    finally:
        # Cleanup config file
        if os.path.exists(config_filename):
            os.remove(config_filename)
            print(f"🧹 Cleaned up config: {config_filename}")

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description='Silent Voice Medical System Launcher with Intelligent Decision Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PRESET CONFIGURATIONS:
  default  - Standard medical monitoring (30s intervals, 20 calls/session)
  icu      - ICU monitoring (20s intervals, 30 calls/session, high sensitivity)
  als      - ALS patient monitoring (25s intervals, 25 calls/session)
  stroke   - Stroke patient monitoring (35s intervals, 15 calls/session)
  demo     - Demo mode (15s intervals, 50 calls/session, debug enabled)

EXAMPLES:
  python launch_silent_voice.py
  python launch_silent_voice.py --preset icu --video patient_icu.mp4
  python launch_silent_voice.py --preset als --webcam 1
  python launch_silent_voice.py --preset demo --video demo.mp4 --model path/to/gemma
  python launch_silent_voice.py --list-presets
        """
    )
    
    parser.add_argument('--preset', '-p', choices=list(MEDICAL_PRESETS.keys()), default='default',
                        help='Medical monitoring preset (default: default)')
    
    parser.add_argument('--video', '-v', type=str,
                        help='Path to video file (if not provided, uses webcam)')
    
    parser.add_argument('--webcam', '-w', type=int, default=0,
                        help='Webcam index if no video file (default: 0)')
    
    parser.add_argument('--log', '-l', type=str,
                        help='Custom log file path (auto-generated if not provided)')
    
    parser.add_argument('--model', '-m', type=str,
                        help='Path to Silent Voice AI model directory')
    
    parser.add_argument('--list-presets', action='store_true',
                        help='List all available medical presets and exit')
    
    parser.add_argument('--demo', action='store_true',
                        help='Quick demo mode (equivalent to --preset demo)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Handle special modes
    if args.list_presets:
        list_presets()
        return
    
    if args.demo:
        args.preset = 'demo'
        if not args.video:
            # Look for common demo video files
            demo_videos = ['patient_1.mp4', 'patient.mp4', 'demo.mp4', 'test.mp4']
            for demo_video in demo_videos:
                if os.path.exists(demo_video):
                    args.video = demo_video
                    print(f"🎬 Auto-selected demo video: {demo_video}")
                    break
            
            if not args.video:
                print("⚠️  Demo mode: No demo video found, using webcam")
    
    # Validate dependencies
    if not os.path.exists('emotion_recognition_medical.py'):
        print("❌ Error: emotion_recognition_medical.py not found")
        print("Please run this launcher from the Silent Voice project directory")
        sys.exit(1)
    
    if not os.path.exists('gemma_decision_engine.py'):
        print("❌ Error: gemma_decision_engine.py not found")
        print("Please ensure the decision engine module is in the same directory")
        sys.exit(1)
    
    # Launch the system
    success = launch_system(
        preset=args.preset,
        video_file=args.video,
        webcam_index=args.webcam,
        log_file=args.log,
        silent_voice_model=args.model
    )
    
    if success:
        print()
        print("🎯 SILENT VOICE SESSION SUMMARY:")
        print("   • Intelligent decision engine optimized API calls")
        print("   • Medical-grade emotion and eye tracking completed")
        print("   • Check log files for detailed analysis")
        print("   • Decision engine statistics show cost savings")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 