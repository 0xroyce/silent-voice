#!/usr/bin/env python3
"""
Silent Voice Medical Recognition with Eye Tracking
Advanced system for paralysis patients with Silent Voice AI integration.
Combines YOLO detection with trained Gemma 3n model for biosignal communication.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import sys
import argparse
import os
import warnings
import json
from datetime import datetime, timedelta
import math
warnings.filterwarnings("ignore")

# Try to import required libraries
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️  DeepFace not available - emotion recognition will be limited")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available - eye tracking will be disabled")
    print("Install with: pip install mediapipe")

# Try to import transformers for Silent Voice model
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available - Silent Voice model disabled")
    print("Install with: pip install torch transformers")

# Try to import the decision engine
try:
    from gemma_decision_engine import GemmaDecisionEngine, Priority
    DECISION_ENGINE_AVAILABLE = True
except ImportError:
    DECISION_ENGINE_AVAILABLE = False
    print("⚠️  Decision engine not available - using simple throttling")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not installed. Install with: pip install ollama")

# New imports for improvements
import threading
import queue
import psutil
from cryptography.fernet import Fernet
import yaml
from collections import Counter, deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Global config loader
def load_config(config_path='config.yaml'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        'blink_threshold': 0.2,
        'mouth_open_threshold': 0.08,
        'emotion_sustain_threshold': 2.0,
        'high_confidence_threshold': 0.7,
        'rapid_blink_window': 3.0,
        'rapid_blink_count': 5,
        'gaze_pattern_window': 5.0,
        'confidence_threshold': 0.3,
        'emotion_mode': 'deepface',
        'print_mode': 'medical',
        'alert_threshold': 10.0,
        'communication_patterns': {
            'urgent_attention': {'rapid_blinks': 5, 'emotion': ['fear', 'distress'], 'confidence': 0.7},
            'pain_signal': {'sustained_emotion': ['fear', 'sad', 'angry'], 'duration': 3.0, 'confidence': 0.6},
            'acknowledgment': {'blinks': 2, 'window': 1.0, 'emotion': ['neutral', 'happy']},
            'distress_escalation': {'emotion_sequence': ['sad', 'fear'], 'intensity_increase': True, 'duration': 5.0}
        }
    }

CONFIG = load_config()

class SilentVoiceIntegration:
    def __init__(self, model_path=None, device='auto'):
        self.model = None
        self.tokenizer = None
        self.device = device
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  Silent Voice integration disabled - transformers not available")
            return
        
        if model_path and os.path.exists(model_path):
            try:
                print(f"🧠 Loading Silent Voice model from: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                
                if device == 'auto':
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                else:
                    self.device = device
                
                self.model.to(self.device)
                print(f"✅ Silent Voice model loaded on {self.device}")
            except Exception as e:
                print(f"❌ Error loading Silent Voice model: {e}")
                self.model = None
                self.tokenizer = None
        
        self.model_name = 'hf.co/0xroyce/silent-voice-multimodal'
        if OLLAMA_AVAILABLE:
            print(f"🧠 Using Ollama model: {self.model_name}")
        else:
            print("⚠️  Ollama not available - will use fallback responses")
    
    def emotion_to_biosignal(self, emotion, confidence, eye_data=None, context=None, patient_condition=None, visual_context=None):
        """
        Generate rich biosignal data matching the fine-tuned model's training format.
        This creates nuanced descriptions similar to the training dataset.
        """
        import random
        
        # Determine signal type based on what's detected
        signal_types = []
        if emotion and confidence > 0.3:
            signal_types.append("facial")
        if eye_data and (eye_data.get('gaze_direction') != 'CENTER' or eye_data.get('is_blinking') or eye_data.get('eye_movement_velocity', 0) > 0.5):
            signal_types.append("eye_movement")
        if eye_data and 'facial_landmarks' in eye_data:
            mouth = eye_data['facial_landmarks'].get('mouth', {})
            if mouth.get('state') == 'open' or mouth.get('openness', 0) > 0.1:
                signal_types.append("muscle")
        
        signal_type = signal_types[0] if signal_types else "facial"
        
        # Generate rich biosignal description
        biosignal_parts = []
        
        # Emotion-based descriptions with nuance
        if emotion.lower() == 'happy':
            if confidence > 0.8:
                descriptors = ["Bright smile", "Beaming expression", "Clear happiness"]
            elif confidence > 0.6:
                descriptors = ["Slight smile", "Pleasant expression", "Soft smile"]
            else:
                descriptors = ["Hint of smile", "Subtle upturn of lips", "Faint smile"]
        elif emotion.lower() == 'sad':
            if confidence > 0.8:
                descriptors = ["Deep sadness in eyes", "Downturned expression", "Sorrowful look"]
            elif confidence > 0.6:
                descriptors = ["Melancholy expression", "Slight frown", "Sad eyes"]
            else:
                descriptors = ["Subtle sadness", "Slight downturn", "Hint of melancholy"]
        elif emotion.lower() == 'fear':
            if confidence > 0.8:
                descriptors = ["Wide-eyed fear", "Alarmed expression", "Fearful look"]
            elif confidence > 0.6:
                descriptors = ["Worried expression", "Anxious look", "Concerned face"]
            else:
                descriptors = ["Slight worry", "Hint of concern", "Mild anxiety"]
        elif emotion.lower() == 'angry':
            if confidence > 0.8:
                descriptors = ["Furrowed brow (frustration)", "Tense facial muscles", "Angry expression"]
            elif confidence > 0.6:
                descriptors = ["Frustrated look", "Tightened jaw", "Annoyed expression"]
            else:
                descriptors = ["Slight tension", "Mild frustration", "Hint of annoyance"]
        elif emotion.lower() == 'surprise':
            if confidence > 0.8:
                descriptors = ["Eyes wide with surprise", "Startled expression", "Shocked look"]
            elif confidence > 0.6:
                descriptors = ["Surprised look", "Raised eyebrows", "Unexpected reaction"]
            else:
                descriptors = ["Slight surprise", "Mild reaction", "Subtle startle"]
        elif emotion.lower() == 'disgust':
            descriptors = ["Wrinkled nose", "Disgusted expression", "Repulsed look"]
        else:  # neutral
            descriptors = ["Calm expression", "Neutral face", "Relaxed features"]
        
        biosignal_parts.append(random.choice(descriptors))
        
        # Add eye-specific descriptions
        if eye_data:
            # Gaze patterns
            gaze_dir = eye_data.get('gaze_direction', 'CENTER')
            if gaze_dir != 'CENTER':
                if gaze_dir in ['UP', 'UP-LEFT', 'UP-RIGHT']:
                    biosignal_parts.append("Looking upward")
                elif gaze_dir in ['DOWN', 'DOWN-LEFT', 'DOWN-RIGHT']:
                    biosignal_parts.append("Downcast gaze")
                elif 'LEFT' in gaze_dir:
                    biosignal_parts.append("Glancing left")
                elif 'RIGHT' in gaze_dir:
                    biosignal_parts.append("Looking to the right")
            
            # Blink patterns
            if eye_data.get('is_blinking'):
                biosignal_parts.append("Mid-blink")
            elif eye_data.get('blink_count', 0) > 5:
                biosignal_parts.append(f"Rapid blinking for {eye_data['blink_count']/10:.0f}s")
            
            # Eye movement
            velocity = eye_data.get('eye_movement_velocity', 0)
            if velocity > 1.0:
                biosignal_parts.append("Jerky eye movements")
            elif velocity > 0.5:
                biosignal_parts.append("Quick eye movements")
            
            # Eye openness
            if 'facial_landmarks' in eye_data:
                left_eye = eye_data['facial_landmarks']['left_eye']
                right_eye = eye_data['facial_landmarks']['right_eye']
                
                if left_eye['openness'] < 0.3 and right_eye['openness'] < 0.3:
                    biosignal_parts.append("Squinting slightly")
                elif left_eye['openness'] > 0.9 or right_eye['openness'] > 0.9:
                    biosignal_parts.append("Eyes wide")
                
                # Check for asymmetry
                if abs(left_eye['openness'] - right_eye['openness']) > 0.3:
                    biosignal_parts.append("Asymmetric eye opening")
                
                # Mouth state
                mouth = eye_data['facial_landmarks']['mouth']
                if mouth['state'] == 'open' and mouth['openness'] > 0.5:
                    biosignal_parts.append("Mouth open wide")
                elif mouth['state'] == 'open' and mouth['openness'] > 0.25:
                    biosignal_parts.append("Mouth slightly parted")
                elif mouth['openness'] > 0.1:
                    biosignal_parts.append("Lips parted")
                
                # Check for muscle tension
                if mouth.get('mar', 0) < 0.05 and emotion.lower() in ['angry', 'fear']:
                    biosignal_parts.append("Jaw clenched")
        
        # Determine urgency level
        if emotion.lower() in ['fear', 'angry', 'sad'] and confidence > 0.8:
            urgency_level = "high"
        elif emotion.lower() in ['fear', 'angry', 'sad'] and confidence > 0.5:
            urgency_level = "medium"
        else:
            urgency_level = "low"
        
        # Special patterns
        if eye_data and eye_data.get('blink_count', 0) > 10:
            urgency_level = "high"
        
        # Is this a simple query?
        is_simple = len(biosignal_parts) <= 2 and urgency_level == "low"
        
        # Create biosignal description
        if len(biosignal_parts) == 1:
            biosignal_description = biosignal_parts[0]
        elif len(biosignal_parts) == 2:
            biosignal_description = f"{biosignal_parts[0]}, {biosignal_parts[1]}"
        else:
            # Natural combination of descriptions
            main_parts = biosignal_parts[:2]
            extra_parts = biosignal_parts[2:]
            biosignal_description = f"{main_parts[0]}, {main_parts[1]}"
            if extra_parts:
                biosignal_description += f" with {', '.join(extra_parts)}"
        
        biosignal_description = f"{emotion} expression with confidence {confidence:.2f}"
        
        if eye_data and 'hr_data' in eye_data:
            hr = eye_data['hr_data']
            biosignal_description += f" and heart rate {hr['current_hr']:.0f} bpm (stress: {hr['stress_level']})"
        
        # Format context string
        context_parts = []
        if context:
            context_parts.append(context)
        if patient_condition:
            # Extract condition type
            condition_map = {
                "ALS": "ALS",
                "spinal": "spinal_cord_injury",
                "stroke": "stroke",
                "cerebral": "cerebral_palsy",
                "locked": "locked_in_syndrome",
                "ICU": "critical_care"
            }
            
            medical_condition = None
            for key, value in condition_map.items():
                if key.lower() in patient_condition.lower():
                    medical_condition = value
                    break
            
            if not medical_condition:
                medical_condition = "general_monitoring"
            
            context_parts.append(patient_condition)
        
        context_str = ", ".join(context_parts) if context_parts else ""
        
        # Build structured data matching training format
        structured_data = {
            "biosignal_description": biosignal_description,
            "context": context_str,
            "is_simple_query": is_simple,
            "medical_condition": medical_condition if patient_condition else None,
            "signal_type": signal_type,
            "urgency_level": urgency_level,
            "visual_data": None
        }
        
        # Add visual data if available
        if eye_data and 'facial_landmarks' in eye_data:
            landmarks = eye_data['facial_landmarks']
            structured_data["visual_data"] = {
                "expression": {
                    "confidence": confidence,
                    "intensity": "high" if confidence > 0.8 else "moderate" if confidence > 0.5 else "low",
                    "primary_emotion": emotion.lower()
                },
                "facial_landmarks": {
                    "left_eye": landmarks['left_eye']['center'],
                    "right_eye": landmarks['right_eye']['center'],
                    "mouth": landmarks['mouth']['center']
                }
            }
        
        # Create model input in conversational format
        model_input = f'{{ "content": "Biosignal: {biosignal_description}'
        if context_str:
            model_input += f'\\nContext: {context_str}'
        if urgency_level != "low":
            model_input += f'\\nUrgency: {urgency_level}'
        model_input += '", "role": "user" },'
        
        # Also support simple format for variety
        if is_simple or random.random() < 0.3:
            model_input = f"Biosignal: {biosignal_description}"
        
        # Add visual context if available
        if visual_context:
            # Append visual observation naturally
            if "Biosignal:" in model_input:
                model_input = model_input.replace(
                    f"Biosignal: {biosignal_description}",
                    f"Biosignal: {biosignal_description}\\nVisual: {visual_context}"
                )
            else:
                model_input += f"\\nVisual: {visual_context}"
        
        log_data = {
            'biosignal_description': biosignal_description,
            'context': context_str,
            'is_simple_query': is_simple,
            'medical_condition': medical_condition if patient_condition else None,
            'signal_type': signal_type,
            'urgency_level': urgency_level,
            'original_emotion': emotion,
            'confidence': confidence,
            'model_input': model_input,
            'visual_data': structured_data.get('visual_data')
        }
        
        return model_input, log_data
    
    def _determine_urgency(self, emotion, confidence, intensity):
        """This method is now unused - urgency is determined by AI"""
        return None
    
    def generate_response(self, biosignal_input, log_data=None):
        if not OLLAMA_AVAILABLE:
            response = self._generate_fallback_response(biosignal_input)
            return response, {
                'model_loaded': False,
                'response_type': 'fallback_no_ollama',
                'response': response
            }
        
        try:
            enhanced_input = biosignal_input
            
            print(f"  [OLLAMA] Generating response for biosignal with integrated visual context ({len(enhanced_input)} chars)...")
            start_time = time.time()
            
            system_prompt = """You are a person communicating through biosignals - facial expressions, eye movements, and subtle body language.

Based on the biosignal description provided, respond in FIRST PERSON with what you want to communicate.

Guidelines:
- Respond naturally as if you're the person experiencing these signals
- Keep responses concise (1-2 sentences) unless the urgency is high
- Match your response to the signal intensity and urgency level
- Consider the context (medical condition, location) when forming responses
- For simple queries, give simple responses
- For complex signals, express the underlying need or feeling

Signal interpretation examples:
- "Squinting slightly" → Could indicate discomfort, concentration, or difficulty seeing
- "Rapid blinking for 3s" → Urgent communication attempt 
- "Furrowed brow (frustration)" → Expressing difficulty or annoyance
- "Tense facial muscles" → Physical or emotional distress
- "Asymmetric muscle activation" → Possible physical limitation affecting one side
- "Jerky eye movements" → Difficulty controlling movements, possible neurological

Context matters:
- In ICU: Focus on immediate medical needs
- Therapy room: Related to treatment or exercises
- Early morning: Could relate to sleep, medication timing
- With specific conditions (ALS, stroke, etc.): Consider typical challenges

Respond directly with your message, no analysis or explanation."""
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': enhanced_input}
            ]
            
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
            )
            
            response = ''
            for chunk in stream:
                response += chunk['message']['content']
            
            generation_time = time.time() - start_time
            print(f"  [OLLAMA] Generated in {generation_time:.2f}s: {response[:100]}...")
            
            print(f"\n🗣️ SILENT VOICE GEMMA RESPONSE: \"{response}\"\n")
            
            response_type = 'ollama_generated'
            
            return response, {
                'model_loaded': True,
                'tokenizer_loaded': True,
                'device': 'ollama',
                'input_length': len(enhanced_input),
                'would_use_model': True,
                'response_type': response_type,
                'response': response,
                'generation_time': generation_time,
                'output_tokens': len(response.split()),
            }
        except Exception as e:
            print(f"  [OLLAMA] Error: {e}")
            response = self._generate_fallback_response(biosignal_input)
            return response, {
                'model_loaded': False,
                'response_type': 'error_fallback',
                'error': str(e),
                'response': response
            }
    
    def _generate_fallback_response(self, biosignal_input):
        """Generate fallback responses matching training data patterns"""
        input_lower = biosignal_input.lower()
        
        # Richer, more varied responses based on biosignal descriptions
        if 'bright smile' in input_lower or 'beaming' in input_lower:
            responses = [
                "I'm feeling really positive about my progress.",
                "Thank you, this is exactly what I needed!",
                "I'm so happy to see you today."
            ]
        elif 'slight smile' in input_lower or 'pleasant expression' in input_lower:
            responses = [
                "I'm content.",
                "Things are going well, thank you.",
                "I appreciate your help."
            ]
        elif 'deep sadness' in input_lower or 'sorrowful' in input_lower:
            responses = [
                "I'm feeling really down today.",
                "I need some emotional support.",
                "This is harder than I expected."
            ]
        elif 'melancholy' in input_lower or 'slight frown' in input_lower:
            responses = [
                "I'm feeling a bit low.",
                "Not my best day.",
                "I could use some encouragement."
            ]
        elif 'wide-eyed fear' in input_lower or 'alarmed' in input_lower:
            responses = [
                "Something's wrong - I need help now!",
                "I'm scared about what's happening.",
                "Please don't leave me alone."
            ]
        elif 'worried expression' in input_lower or 'anxious' in input_lower:
            responses = [
                "I'm concerned about something.",
                "Can we talk about what happens next?",
                "I'm feeling anxious about this."
            ]
        elif 'furrowed brow' in input_lower or 'frustrated look' in input_lower:
            responses = [
                "I'm so frustrated.",
                "This isn't working the way I need it to.",
                "I'm having trouble with this."
            ]
        elif 'tense facial muscles' in input_lower:
            responses = [
                "I'm uncomfortable.",
                "There's too much tension here.",
                "I need to relax but I can't."
            ]
        elif 'squinting' in input_lower:
            responses = [
                "I'm having trouble seeing clearly.",
                "The light is bothering my eyes.",
                "I'm trying to focus."
            ]
        elif 'rapid blinking' in input_lower:
            count = 0
            import re
            match = re.search(r'(\d+)s', input_lower)
            if match:
                count = int(match.group(1))
            
            if count >= 3:
                responses = ["I need help.", "Please pay attention to me.", "This is urgent."]
            else:
                responses = ["I'm trying to tell you something.", "Look at me please.", "I need to communicate."]
        elif 'jerky eye movements' in input_lower:
            responses = [
                "I'm having trouble controlling my movements.",
                "Something feels different today.",
                "My coordination is off."
            ]
        elif 'asymmetric' in input_lower:
            responses = [
                "One side feels different than the other.",
                "I'm noticing weakness on one side.",
                "Something's not balanced."
            ]
        elif 'mouth open wide' in input_lower:
            responses = [
                "I'm trying to speak.",
                "I need something urgently.",
                "Help me communicate."
            ]
        elif 'jaw clenched' in input_lower:
            responses = [
                "I'm tense.",
                "I'm holding in frustration.",
                "I'm trying to stay calm."
            ]
        elif 'looking upward' in input_lower:
            responses = [
                "Yes, that's right.",
                "I agree with that.",
                "Keep going."
            ]
        elif 'downcast gaze' in input_lower:
            responses = [
                "No, that's not it.",
                "I don't want that.",
                "Something's wrong."
            ]
        elif 'glancing left' in input_lower or 'looking to the right' in input_lower:
            responses = [
                "Look over there.",
                "I need something from that direction.",
                "Check that side please."
            ]
        else:
            # Generic responses for unmatched patterns
            responses = [
                "I'm trying to communicate.",
                "I have something to tell you.",
                "Please help me express myself."
            ]
        
        # Add urgency-based variations
        if 'urgency: high' in input_lower:
            # Prepend urgency to response
            response_index = hash(biosignal_input) % len(responses)
            base_response = responses[response_index]
            return f"{base_response} This is important!"
        elif 'urgency: critical' in input_lower:
            response_index = hash(biosignal_input) % len(responses)
            base_response = responses[response_index]
            return f"URGENT: {base_response}"
        
        # Use hash for consistent selection
        response_index = hash(biosignal_input) % len(responses)
        return responses[response_index]
    
    def analyze_visual_scene(self, frame, save_path=None):
        if not OLLAMA_AVAILABLE:
            return None
        
        try:
            if save_path is None:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                save_path = temp_file.name
                temp_cleanup = True
            else:
                temp_cleanup = False
            
            cv2.imwrite(save_path, frame)
            print(f"  [OLLAMA VISION] Analyzing visual scene...")
            
            vision_prompt = """Analyze this image and describe what you observe about the person in 2-3 sentences.

Focus on:
- The person's physical position and posture
- Any visible signs of comfort or discomfort
- Environmental factors that might affect their state
- Any medical equipment or assistive devices visible
- Lighting, setting, or other contextual details

Be objective and descriptive. Mention specific details that could be medically relevant.
Do not make assumptions about emotions - just describe what you see."""
            
            start_time = time.time()
            
            import base64
            with open(save_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode()
            
            messages = [{
                'role': 'user',
                'content': vision_prompt,
                'images': [image_data]
            }]
            
            stream = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
            )
            
            description = ''
            for chunk in stream:
                description += chunk['message']['content']
            
            analysis_time = time.time() - start_time
            print(f"  [OLLAMA VISION] Analyzed in {analysis_time:.2f}s")
            
            if temp_cleanup:
                os.unlink(save_path)
            
            return {
                'description': description,
                'analysis_time': analysis_time,
                'image_path': save_path if not temp_cleanup else None
            }
            
        except Exception as e:
            print(f"  [OLLAMA VISION] Error: {e}")
            return None
    
    def process_detection(self, emotion_data, eye_data, context=None, patient_condition=None, frame=None):
        visual_analysis = None
        visual_context = None
        if frame is not None:
            visual_analysis = self.analyze_visual_scene(frame)
            if visual_analysis and visual_analysis.get('description'):
                visual_context = visual_analysis['description']
        
        biosignal_input, log_data = self.emotion_to_biosignal(
            emotion_data['emotion'],
            emotion_data['confidence'],
            eye_data,
            context,
            patient_condition,
            visual_context
        )
        
        if visual_analysis:
            log_data['visual_analysis'] = visual_analysis['description']
        
        response, model_info = self.generate_response(biosignal_input, log_data)
        
        return {
            'biosignal_input': biosignal_input,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'detection_log': log_data,
            'model_info': model_info,
            'visual_analysis': visual_analysis
        }


class EyeTracker:
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self.face_mesh = None
            return
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        
        self.MOUTH_OUTER_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 78]
        self.MOUTH_INNER_LANDMARKS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 320, 307, 375, 321, 308]
        
        self.blink_threshold = CONFIG['blink_threshold']
        self.blink_frames = 0
        self.blink_count = 0
        self.last_blink_time = 0
        
        self.mouth_open_threshold = CONFIG['mouth_open_threshold']
        self.mouth_history = []
        self.mouth_open_frames = 0
        self.min_mouth_open_frames = 5
        
        self.gaze_history = []
        self.eye_position_history = []
        
        self.calibration_frames = []
        self.is_calibrated = False
        
        self.ear_buffer = deque(maxlen=5)
        self.mar_buffer = deque(maxlen=5)
        
    def calibrate_thresholds(self, frame):
        if len(self.calibration_frames) < 20:
            analysis = self.analyze_eyes(frame)
            if analysis:
                self.calibration_frames.append(analysis)
            return
        
        if not self.is_calibrated:
            ears = [f['ear'] for f in self.calibration_frames]
            mars = [f['mouth_mar'] for f in self.calibration_frames if 'mouth_mar' in f]
            if ears:
                self.blink_threshold = np.mean(ears) - 0.05
            if mars:
                self.mouth_open_threshold = np.mean(mars) + 0.02
            self.is_calibrated = True
            print(f"✅ Calibrated thresholds: Blink={self.blink_threshold:.3f}, Mouth={self.mouth_open_threshold:.3f}")
    
    def calculate_ear(self, eye_landmarks):
        if len(eye_landmarks) < 6:
            return 0
        
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def get_gaze_direction(self, left_iris, right_iris, left_eye_corners, right_eye_corners):
        try:
            left_center = np.mean(left_eye_corners, axis=0)
            left_iris_center = np.mean(left_iris, axis=0)
            left_gaze = left_iris_center - left_center
            
            right_center = np.mean(right_eye_corners, axis=0)
            right_iris_center = np.mean(right_iris, axis=0)
            right_gaze = right_iris_center - right_center
            
            avg_gaze = (left_gaze + right_gaze) / 2
            
            gaze_x = np.clip(avg_gaze[0] * 10, -1, 1)
            gaze_y = np.clip(avg_gaze[1] * 10, -1, 1)
            
            return gaze_x, gaze_y
        except:
            return 0, 0
    
    def get_mouth_landmarks(self, face_landmarks, w, h):
        outer_mouth = []
        inner_mouth = []
        
        for idx in self.MOUTH_OUTER_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            outer_mouth.append([landmark.x * w, landmark.y * h])
        
        for idx in self.MOUTH_INNER_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            inner_mouth.append([landmark.x * w, landmark.y * h])
        
        return np.array(outer_mouth), np.array(inner_mouth)
    
    def analyze_mouth_state(self, outer_mouth, inner_mouth, face_landmarks, h, w):
        upper_lip_center = face_landmarks.landmark[13]
        lower_lip_center = face_landmarks.landmark[14]
        left_corner = face_landmarks.landmark[61]
        right_corner = face_landmarks.landmark[291]
        
        upper_lip_y = upper_lip_center.y * h
        lower_lip_y = lower_lip_center.y * h
        left_corner_x = left_corner.x * w
        right_corner_x = right_corner.x * w
        left_corner_y = left_corner.y * h
        right_corner_y = right_corner.y * h
        
        vertical_dist = abs(lower_lip_y - upper_lip_y)
        horizontal_dist = abs(right_corner_x - left_corner_x)
        
        mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.0
        
        self.mar_buffer.append(mar)
        avg_mar = np.mean(self.mar_buffer)
        
        if avg_mar > self.mouth_open_threshold * 1.2:
            is_open = True
            self.mouth_open_frames += 1
        elif avg_mar < self.mouth_open_threshold * 0.8:
            is_open = False
            self.mouth_open_frames = 0
        else:
            is_open = self.mouth_open_frames >= self.min_mouth_open_frames
        
        if is_open and self.mouth_open_frames < self.min_mouth_open_frames:
            is_open = False
        
        if is_open:
            state = "open"
        else:
            center_y = (upper_lip_center.y + lower_lip_center.y) * h / 2
            if left_corner_y < center_y - 5 and right_corner_y < center_y - 5:
                state = "smile"
            else:
                state = "closed"
        
        if mar < 0.05:
            openness = 0.0
        elif mar < 0.08:
            openness = (mar - 0.05) / 0.03 * 0.3
        elif mar < 0.12:
            openness = 0.3 + (mar - 0.08) / 0.04 * 0.4
        else:
            openness = min(1.0, 0.7 + (mar - 0.12) / 0.08 * 0.3)
        
        if not hasattr(self, '_last_mouth_state'):
            self._last_mouth_state = state
            self._debug_counter = 0
        
        self._debug_counter += 1
        if state != self._last_mouth_state or self._debug_counter % 30 == 0:
            print(f"[DEBUG MOUTH] MAR: {mar:.4f}, AvgMAR: {avg_mar:.4f}, Threshold: {self.mouth_open_threshold:.3f}, State: {state}, Openness: {openness:.3f}")
            if self._debug_counter % 30 == 0:
                print(f"  [Calibration: Closed<0.05, Parted:0.05-0.08, Open:0.08-0.12, Wide:>0.12]")
            self._last_mouth_state = state
        
        return state, openness, mar
    
    def analyze_eyes(self, frame, face_region=None):
        if self.face_mesh is None:
            return None
        
        analysis_frame = face_region if face_region is not None else frame
        rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w = analysis_frame.shape[:2]
        
        left_eye_points = []
        right_eye_points = []
        left_iris_points = []
        right_iris_points = []
        
        for idx in self.LEFT_EYE_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            left_eye_points.append([landmark.x * w, landmark.y * h])
        
        for idx in self.RIGHT_EYE_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            right_eye_points.append([landmark.x * w, landmark.y * h])
        
        for idx in self.LEFT_IRIS:
            landmark = face_landmarks.landmark[idx]
            left_iris_points.append([landmark.x * w, landmark.y * h])
        
        for idx in self.RIGHT_IRIS:
            landmark = face_landmarks.landmark[idx]
            right_iris_points.append([landmark.x * w, landmark.y * h])
        
        left_eye_points = np.array(left_eye_points)
        right_eye_points = np.array(right_eye_points)
        left_iris_points = np.array(left_iris_points)
        right_iris_points = np.array(right_iris_points)
        
        outer_mouth, inner_mouth = self.get_mouth_landmarks(face_landmarks, w, h)
        mouth_state, mouth_openness, mouth_mar = self.analyze_mouth_state(outer_mouth, inner_mouth, face_landmarks, h, w)
        
        left_ear = self.calculate_ear(left_eye_points[:6])
        right_ear = self.calculate_ear(right_eye_points[:6])
        avg_ear = (left_ear + right_ear) / 2
        
        self.ear_buffer.append(avg_ear)
        avg_ear = np.mean(self.ear_buffer)
        
        left_eye_state = "open" if left_ear > self.blink_threshold else "closed"
        right_eye_state = "open" if right_ear > self.blink_threshold else "closed"
        
        left_openness = min(1.0, max(0.0, left_ear / 0.3))
        right_openness = min(1.0, max(0.0, right_ear / 0.3))
        
        is_blinking = avg_ear < self.blink_threshold
        if is_blinking:
            self.blink_frames += 1
        else:
            if self.blink_frames >= 2:
                self.blink_count += 1
                self.last_blink_time = time.time()
            self.blink_frames = 0
        
        gaze_x, gaze_y = self.get_gaze_direction(
            left_iris_points, right_iris_points,
            left_eye_points, right_eye_points
        )
        
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        mouth_center = np.mean(outer_mouth, axis=0)
        
        left_eye_center_norm = [left_eye_center[0] / w, left_eye_center[1] / h]
        right_eye_center_norm = [right_eye_center[0] / w, right_eye_center[1] / h]
        mouth_center_norm = [mouth_center[0] / w, mouth_center[1] / h]
        
        current_time = time.time()
        self.gaze_history.append({
            'time': current_time,
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'ear': avg_ear,
            'blink': is_blinking
        })
        
        self.gaze_history = [g for g in self.gaze_history if current_time - g['time'] < 5.0]
        
        return {
            'left_eye': left_eye_points,
            'right_eye': right_eye_points,
            'left_iris': left_iris_points,
            'right_iris': right_iris_points,
            'ear': avg_ear,
            'is_blinking': is_blinking,
            'blink_count': self.blink_count,
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'gaze_direction': self.get_gaze_label(gaze_x, gaze_y),
            'eye_movement_velocity': self.calculate_eye_velocity(),
            'outer_mouth': outer_mouth,
            'inner_mouth': inner_mouth,
            'mouth_mar': mouth_mar,
            'facial_landmarks': {
                'left_eye': {
                    'center': left_eye_center_norm,
                    'state': left_eye_state,
                    'openness': left_openness
                },
                'right_eye': {
                    'center': right_eye_center_norm,
                    'state': right_eye_state,
                    'openness': right_openness
                },
                'mouth': {
                    'center': mouth_center_norm,
                    'state': mouth_state,
                    'openness': mouth_openness,
                    'mar': mouth_mar
                }
            }
        }
    
    def get_gaze_label(self, gaze_x, gaze_y):
        if abs(gaze_x) < 0.2 and abs(gaze_y) < 0.2:
            return "CENTER"
        elif gaze_x > 0.3:
            return "RIGHT" if abs(gaze_y) < 0.3 else ("UP-RIGHT" if gaze_y < -0.3 else "DOWN-RIGHT")
        elif gaze_x < -0.3:
            return "LEFT" if abs(gaze_y) < 0.3 else ("UP-LEFT" if gaze_y < -0.3 else "DOWN-LEFT")
        elif gaze_y < -0.3:
            return "UP"
        elif gaze_y > 0.3:
            return "DOWN"
        else:
            return "CENTER"
    
    def calculate_eye_velocity(self):
        if len(self.gaze_history) < 2:
            return 0
        
        recent_gazes = self.gaze_history[-5:]
        if len(recent_gazes) < 2:
            return 0
        
        velocities = []
        for i in range(1, len(recent_gazes)):
            dt = recent_gazes[i]['time'] - recent_gazes[i-1]['time']
            if dt > 0:
                dx = recent_gazes[i]['gaze_x'] - recent_gazes[i-1]['gaze_x']
                dy = recent_gazes[i]['gaze_y'] - recent_gazes[i-1]['gaze_y']
                velocity = math.sqrt(dx*dx + dy*dy) / dt
                velocities.append(velocity)
        
        return np.mean(velocities) if velocities else 0
    
    def analyze_facial_symmetry(self, face_landmarks, w, h):
        left_eye_center = np.mean([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] 
                                  for idx in self.LEFT_EYE_LANDMARKS], axis=0)
        right_eye_center = np.mean([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] 
                                   for idx in self.RIGHT_EYE_LANDMARKS], axis=0)
        
        nose_tip = [face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h]
        mouth_center = np.mean([[face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] 
                               for idx in [78, 308]], axis=0)
        
        face_top = [face_landmarks.landmark[10].x * w, face_landmarks.landmark[10].y * h]
        face_bottom = [face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h]
        
        left_distances = [
            abs(left_eye_center[0] - nose_tip[0]),
            abs(mouth_center[0] - nose_tip[0])
        ]
        right_distances = [
            abs(right_eye_center[0] - nose_tip[0]),
            abs(mouth_center[0] - nose_tip[0])
        ]
        
        symmetry_scores = []
        for left_dist, right_dist in zip(left_distances, right_distances):
            if left_dist + right_dist > 0:
                symmetry_scores.append(1 - abs(left_dist - right_dist) / (left_dist + right_dist))
        
        overall_symmetry = np.mean(symmetry_scores) if symmetry_scores else 0.5
        
        left_avg = np.mean(left_distances)
        right_avg = np.mean(right_distances)
        
        if abs(left_avg - right_avg) < 5:
            affected_side = "none"
            drooping_level = "none"
        elif left_avg > right_avg:
            affected_side = "left"
            drooping_level = self.assess_drooping_level(overall_symmetry)
        else:
            affected_side = "right"
            drooping_level = self.assess_drooping_level(overall_symmetry)
        
        return {
            'facial_symmetry': overall_symmetry,
            'affected_side': affected_side,
            'drooping': drooping_level
        }
    
    def assess_drooping_level(self, symmetry_score):
        if symmetry_score > 0.9:
            return "none"
        elif symmetry_score > 0.7:
            return "mild"
        elif symmetry_score > 0.5:
            return "moderate"
        else:
            return "severe"
    
    def analyze_visual_context(self, frame, face_region):
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 80:
            lighting = "dim"
        elif brightness < 160:
            lighting = "artificial"
        else:
            lighting = "bright"
        
        face_h, face_w = face_region.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        
        face_ratio = (face_w * face_h) / (frame_w * frame_h)
        
        if face_ratio > 0.3:
            distance = "extreme_close_up"
        elif face_ratio > 0.15:
            distance = "close_up"
        elif face_ratio > 0.05:
            distance = "medium"
        else:
            distance = "far"
        
        camera_angle = "frontal"
        face_movement = "still"
        occlusions = "none"
        
        return {
            'lighting': lighting,
            'camera_angle': camera_angle,
            'distance': distance,
            'face_movement': face_movement,
            'occlusions': occlusions
        }
    
    def detect_expression_features(self, emotion, confidence):
        features = []
        
        if emotion.lower() == "happy":
            if confidence > 0.7:
                features.extend(["smile", "raised_cheeks"])
            if confidence > 0.8:
                features.append("crow's_feet")
        elif emotion.lower() == "sad":
            if confidence > 0.6:
                features.extend(["downturned_mouth", "drooping_eyelids"])
        elif emotion.lower() == "fear":
            if confidence > 0.6:
                features.extend(["wide_eyes", "raised_eyebrows"])
        elif emotion.lower() == "angry":
            if confidence > 0.6:
                features.extend(["furrowed_brow", "tightened_lips"])
        elif emotion.lower() == "surprise":
            if confidence > 0.6:
                features.extend(["raised_eyebrows", "wide_eyes", "open_mouth"])
        
        return features
    
    def assess_expression_intensity(self, confidence):
        if confidence > 0.9:
            return "extreme"
        elif confidence > 0.7:
            return "severe"
        elif confidence > 0.5:
            return "moderate"
        elif confidence > 0.3:
            return "mild"
        else:
            return "slight"

class PatternDetector:
    def __init__(self):
        self.EMOTION_SUSTAIN_THRESHOLD = CONFIG['emotion_sustain_threshold']
        self.HIGH_CONFIDENCE_THRESHOLD = CONFIG['high_confidence_threshold']
        self.RAPID_BLINK_WINDOW = CONFIG['rapid_blink_window']
        self.RAPID_BLINK_COUNT = CONFIG['rapid_blink_count']
        self.GAZE_PATTERN_WINDOW = CONFIG['gaze_pattern_window']
        
        self.emotion_history = []
        self.blink_history = []
        self.gaze_pattern_history = []
        self.last_significant_event = None
        self.last_event_time = None
        
        self.communication_patterns = CONFIG['communication_patterns']
        
    def should_trigger_update(self, emotion_data, eye_data, timestamp):
        triggers = []
        
        self._update_histories(emotion_data, eye_data, timestamp)
        
        sustained_emotion = self._check_sustained_emotion(timestamp)
        if sustained_emotion:
            triggers.append(sustained_emotion)
        
        blink_pattern = self._check_blink_patterns(eye_data, timestamp)
        if blink_pattern:
            triggers.append(blink_pattern)
        
        gaze_pattern = self._check_gaze_patterns(eye_data, timestamp)
        if gaze_pattern:
            triggers.append(gaze_pattern)
        
        intensity_change = self._check_intensity_changes(emotion_data, timestamp)
        if intensity_change:
            triggers.append(intensity_change)
        
        comm_pattern = self._check_communication_patterns(emotion_data, eye_data, timestamp)
        if comm_pattern:
            triggers.append(comm_pattern)
        
        if triggers and self.last_event_time:
            time_since_last = (timestamp - self.last_event_time).total_seconds()
            if time_since_last < 1.0:
                return None
        
        if triggers:
            self.last_event_time = timestamp
            return {
                'triggers': triggers,
                'context': self._get_context_summary(timestamp),
                'confidence': max(t.get('confidence', 0) for t in triggers)
            }
        
        return None
    
    def _update_histories(self, emotion_data, eye_data, timestamp):
        if emotion_data:
            self.emotion_history.append((timestamp, emotion_data['emotion'], emotion_data['confidence']))
            cutoff = timestamp - timedelta(seconds=30)
            self.emotion_history = [item for item in self.emotion_history if item[0] > cutoff]
        
        if eye_data and eye_data.get('is_blinking'):
            self.blink_history.append(timestamp)
            cutoff = timestamp - timedelta(seconds=10)
            self.blink_history = [t for t in self.blink_history if t > cutoff]
        
        if eye_data and 'gaze_direction' in eye_data:
            self.gaze_pattern_history.append((timestamp, eye_data['gaze_direction']))
            cutoff = timestamp - timedelta(seconds=10)
            self.gaze_pattern_history = [item for item in self.gaze_pattern_history if item[0] > cutoff]
    
    def _check_sustained_emotion(self, timestamp):
        if len(self.emotion_history) < 2:
            return None
        
        cutoff = timestamp - timedelta(seconds=self.EMOTION_SUSTAIN_THRESHOLD)
        recent_emotions = [item for item in self.emotion_history if item[0] > cutoff]
        
        if not recent_emotions:
            return None
        
        emotions = set(e for _, e, c in recent_emotions if c > self.HIGH_CONFIDENCE_THRESHOLD)
        
        if len(emotions) == 1:
            emotion = list(emotions)[0]
            avg_confidence = np.mean([c for _, e, c in recent_emotions if e == emotion])
            
            if emotion.lower() in ['fear', 'sad', 'angry', 'happy'] and avg_confidence > 0.6:
                return {
                    'type': 'sustained_emotion',
                    'emotion': emotion,
                    'duration': self.EMOTION_SUSTAIN_THRESHOLD,
                    'confidence': avg_confidence,
                    'description': f"Sustained {emotion.lower()} for {self.EMOTION_SUSTAIN_THRESHOLD}s"
                }
        
        return None
    
    def _check_blink_patterns(self, eye_data, timestamp):
        if not eye_data:
            return None
        
        cutoff = timestamp - timedelta(seconds=self.RAPID_BLINK_WINDOW)
        recent_blinks = [t for t in self.blink_history if t > cutoff]
        
        if len(recent_blinks) >= self.RAPID_BLINK_COUNT:
            return {
                'type': 'rapid_blinks',
                'count': len(recent_blinks),
                'window': self.RAPID_BLINK_WINDOW,
                'confidence': 0.9,
                'description': f"{len(recent_blinks)} blinks in {self.RAPID_BLINK_WINDOW}s - possible urgent communication"
            }
        
        if len(recent_blinks) == 2:
            blink_interval = (recent_blinks[1] - recent_blinks[0]).total_seconds()
            if 0.1 < blink_interval < 0.5:
                return {
                    'type': 'double_blink',
                    'confidence': 0.8,
                    'description': "Double blink detected - possible acknowledgment"
                }
        
        return None
    
    def _check_gaze_patterns(self, eye_data, timestamp):
        if not eye_data or len(self.gaze_pattern_history) < 3:
            return None
        
        recent_gazes = self.gaze_pattern_history[-8:]
        directions = [d for _, d in recent_gazes]
        
        circular_sequence = ['UP', 'UP-RIGHT', 'RIGHT', 'DOWN-RIGHT', 'DOWN', 'DOWN-LEFT', 'LEFT', 'UP-LEFT']
        
        for i in range(len(directions) - 3):
            sub_sequence = directions[i:i+4]
            for start_idx in range(len(circular_sequence)):
                expected = [circular_sequence[(start_idx + j) % len(circular_sequence)] for j in range(4)]
                if sub_sequence == expected:
                    return {
                        'type': 'circular_gaze',
                        'pattern': sub_sequence,
                        'confidence': 0.8,
                        'description': "Circular gaze pattern - intentional communication"
                    }
        
        return None
    
    def _check_intensity_changes(self, emotion_data, timestamp):
        if len(self.emotion_history) < 5:
            return None
        
        cutoff = timestamp - timedelta(seconds=5)
        recent = [item for item in self.emotion_history if item[0] > cutoff]
        
        if len(recent) < 3:
            return None
        
        confidences = [c for _, _, c in recent]
        if len(confidences) > 3:
            first_half = np.mean(confidences[:len(confidences)//2])
            second_half = np.mean(confidences[len(confidences)//2:])
            
            if second_half > first_half + 0.2:
                current_emotion = recent[-1][1]
                if current_emotion.lower() in ['fear', 'sad', 'angry']:
                    return {
                        'type': 'emotion_escalation',
                        'emotion': current_emotion,
                        'confidence': second_half,
                        'description': f"Escalating {current_emotion.lower()} - requires attention"
                    }
        
        return None
    
    def _check_communication_patterns(self, emotion_data, eye_data, timestamp):
        if emotion_data and emotion_data['emotion'].lower() in ['fear', 'sad', 'angry']:
            if eye_data and 'facial_landmarks' in eye_data:
                mouth_data = eye_data['facial_landmarks'].get('mouth', {})
                mouth_state = mouth_data.get('state')
                mouth_openness = mouth_data.get('openness', 0)
                
                if mouth_state == 'open' and mouth_openness > 0.5:
                    return {
                        'type': 'pain_signal',
                        'emotion': emotion_data['emotion'],
                        'confidence': 0.85,
                        'description': f"{emotion_data['emotion']} with clearly open mouth - possible pain or distress"
                    }
        
        if emotion_data and emotion_data['emotion'].lower() in ['fear', 'sad', 'angry']:
            if emotion_data['confidence'] > 0.8:
                recent_emotions = [e for _, e, c in self.emotion_history[-10:] if c > 0.7]
                if len(recent_emotions) > 5 and all(e.lower() in ['fear', 'sad', 'angry'] for e in recent_emotions):
                    return {
                        'type': 'sustained_distress',
                        'emotion': emotion_data['emotion'],
                        'confidence': 0.75,
                        'description': f"Sustained {emotion_data['emotion'].lower()} expression - needs attention"
                    }
        
        return None
    
    def _get_context_summary(self, timestamp):
        cutoff = timestamp - timedelta(seconds=10)
        recent_emotions = [item for item in self.emotion_history if item[0] > cutoff]
        
        emotion_summary = {}
        for _, emotion, conf in recent_emotions:
            if emotion not in emotion_summary:
                emotion_summary[emotion] = []
            emotion_summary[emotion].append(conf)
        
        for emotion in emotion_summary:
            emotion_summary[emotion] = np.mean(emotion_summary[emotion])
        
        return {
            'emotion_summary': emotion_summary,
            'total_blinks': len([t for t in self.blink_history if t > cutoff]),
            'dominant_emotion': max(emotion_summary.items(), key=lambda x: x[1])[0] if emotion_summary else None
        }


class MedicalEmotionRecognizer:
    def __init__(self, video_source=None, model_size='m', enable_eye_tracking=True, log_file=None, 
                 silent_voice_model=None, patient_condition=None, context=None):
        print("🏥 Initializing Silent Voice Medical Recognition System")
        print("Designed for paralysis patient monitoring with AI communication")
        print()
        
        self.video_source = video_source
        self.is_video_file = video_source is not None and video_source != 0
        self.model_size = model_size
        self.enable_eye_tracking = enable_eye_tracking and MEDIAPIPE_AVAILABLE
        self.patient_condition = patient_condition
        self.context = context or "monitoring session"
        
        self.emotion_mode = CONFIG['emotion_mode']
        
        self.yolo_emotion_classes = {
            0: ('Pain', 'pain'),
            1: ('Distress', 'distress'),
            2: ('Happy', 'happiness'),
            3: ('Concentration', 'concentration'),
            4: ('Neutral', 'neutral')
        }
        
        try:
            model_file = f'yolo11{self.model_size}.pt'
            emotion_model_file = f'yolo11{self.model_size}_emotions.pt'
            if os.path.exists(emotion_model_file):
                self.yolo_model = YOLO(emotion_model_file)
                self.emotion_mode = 'yolo'
                print(f"✅ YOLOv11{self.model_size.upper()} emotion model loaded")
                print("   Using YOLO for direct emotion detection (5 classes)")
            else:
                self.yolo_model = YOLO(model_file)
                print(f"✅ YOLOv11{self.model_size.upper()} model loaded")
                print("   Using YOLO for face detection only")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            sys.exit(1)
        
        if self.enable_eye_tracking:
            self.eye_tracker = EyeTracker()
            print("✅ Eye tracking system initialized")
        else:
            self.eye_tracker = None
            print("⚠️  Eye tracking disabled")
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ Backup face detection loaded")
        except:
            self.face_cascade = None
        
        self.pattern_detector = PatternDetector()
        print("✅ Smart pattern detection initialized")
        
        if DECISION_ENGINE_AVAILABLE:
            self.decision_engine = GemmaDecisionEngine()
            print("✅ Intelligent decision engine initialized")
        else:
            self.decision_engine = None
            print("⚠️  Using simple throttling for AI calls")
        
        self.silent_voice = SilentVoiceIntegration(
            model_path=silent_voice_model,
            device='auto'
        )
        print("✅ Silent Voice integration initialized")
        
        self.log_file = log_file
        self.session_start = datetime.now()
        self.medical_log = []
        
        self.encryption_key = Fernet.generate_key()
        
        self.previous_emotions = {}
        self.emotion_start_times = {}
        self.print_mode = CONFIG['print_mode']
        self.silent_voice_frame_counter = 0
        
        self.emotion_buffer = deque(maxlen=5)
        
        self.heart_rate_monitor = HeartRateMonitor()
        print("✅ Multi-modal input enabled with heart rate monitoring")
        
        self.predictive_analytics = PredictiveAnalytics()
        print("✅ Predictive analytics enabled for emotion trend forecasting")
        
        print("✅ Medical system ready for patient monitoring")
        print()
    
    def detect_faces(self, frame):
        confidence_threshold = CONFIG['confidence_threshold']
        
        h, w = frame.shape[:2]
        if max(h, w) > 1080:
            scale = 640 / max(h, w)
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_resized = frame
        
        if self.emotion_mode == 'yolo':
            results = self.yolo_model(frame_resized, verbose=False, conf=confidence_threshold)
        else:
            results = self.yolo_model(frame_resized, classes=[0], verbose=False, conf=confidence_threshold)
        
        faces = []
        
        scale_h = h / frame_resized.shape[0]
        scale_w = w / frame_resized.shape[1]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    x1 = int(x1 * scale_w)
                    y1 = int(y1 * scale_h)
                    x2 = int(x2 * scale_w)
                    y2 = int(y2 * scale_h)
                    
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0
                    
                    if confidence > confidence_threshold and 0.3 <= aspect_ratio <= 3.0 and width > 50 and height > 50:
                        face_data = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'method': 'yolo'
                        }
                        if self.emotion_mode == 'yolo':
                            class_id = int(box.cls[0].cpu().numpy())
                            emotion_name, emotion_type = self.yolo_emotion_classes.get(class_id, ('Unknown', 'unknown'))
                            face_data['emotion'] = emotion_name
                            face_data['emotion_confidence'] = float(confidence)
                            face_data['emotion_class'] = class_id
                        faces.append(face_data)
        
        return faces
    
    def recognize_emotion(self, face_img, face_data=None):
        yolo_emotion = None
        yolo_conf = 0.0
        if face_data and 'emotion' in face_data and self.emotion_mode == 'yolo':
            yolo_emotion = face_data['emotion']
            yolo_conf = face_data['emotion_confidence']
        
        deep_emotion = "Unknown"
        deep_conf = 0.0
        if DEEPFACE_AVAILABLE:
            try:
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                emotions = result['emotion']
                dominant_emotion = result['dominant_emotion']
                deep_emotion = dominant_emotion.capitalize()
                deep_conf = emotions[dominant_emotion] / 100.0
            except Exception as e:
                pass
        
        if yolo_emotion and deep_emotion:
            if yolo_emotion == deep_emotion:
                final_emotion = yolo_emotion
                final_conf = (yolo_conf + deep_conf) / 2
            else:
                if yolo_conf > deep_conf + 0.3:
                    final_emotion = yolo_emotion
                    final_conf = yolo_conf
                elif deep_conf > yolo_conf + 0.3:
                    final_emotion = deep_emotion
                    final_conf = deep_conf
                else:
                    final_emotion = "Unknown"
                    final_conf = 0.0
        elif yolo_emotion:
            final_emotion = yolo_emotion
            final_conf = yolo_conf
        elif deep_emotion:
            final_emotion = deep_emotion
            final_conf = deep_conf
        else:
            final_emotion = "Unknown"
            final_conf = 0.0
        
        self.emotion_buffer.append(final_emotion)
        most_common_emotion = Counter(self.emotion_buffer).most_common(1)[0][0]
        
        return most_common_emotion, final_conf
    
    def generate_comprehensive_analysis(self, frame, face_bbox, emotion_data, eye_data, timestamp):
        x1, y1, x2, y2 = face_bbox
        
        face_region = frame[y1:y2, x1:x2]
        
        visual_context = None
        medical_markers = None
        
        if self.eye_tracker and eye_data:
            visual_context = self.eye_tracker.analyze_visual_context(frame, face_region)
            
            rgb_frame = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            results = self.eye_tracker.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = face_region.shape[:2]
                medical_markers = self.eye_tracker.analyze_facial_symmetry(face_landmarks, w, h)
        
        if not visual_context:
            visual_context = {
                'lighting': 'artificial',
                'camera_angle': 'frontal',
                'distance': 'close_up',
                'face_movement': 'still',
                'occlusions': 'none'
            }
        
        if not medical_markers:
            medical_markers = {
                'facial_symmetry': 0.8,
                'affected_side': 'none',
                'drooping': 'none'
            }
        
        frame_h, frame_w = frame.shape[:2]
        normalized_bbox = {
            'format': 'normalized_xyxy',
            'x1': x1 / frame_w,
            'y1': y1 / frame_h,
            'x2': x2 / frame_w,
            'y2': y2 / frame_h
        }
        
        comprehensive_data = {
            'timestamp': timestamp.isoformat(),
            'visual_data': {
                'face_detected': True,
                'confidence': emotion_data.get('confidence', 0.0),
                'bounding_box': normalized_bbox,
                'expression': {
                    'primary_emotion': emotion_data.get('emotion', 'neutral').lower(),
                    'confidence': emotion_data.get('confidence', 0.0),
                    'features_detected': self.eye_tracker.detect_expression_features(
                        emotion_data.get('emotion', 'neutral'),
                        emotion_data.get('confidence', 0.0)
                    ) if self.eye_tracker else [],
                    'intensity': self.eye_tracker.assess_expression_intensity(
                        emotion_data.get('confidence', 0.0)
                    ) if self.eye_tracker else 'slight'
                },
                'visual_context': visual_context,
                'medical_markers': medical_markers
            }
        }
        
        if eye_data and 'facial_landmarks' in eye_data:
            comprehensive_data['visual_data']['facial_landmarks'] = eye_data['facial_landmarks']
        
        if eye_data:
            comprehensive_data['visual_data']['gaze_tracking'] = {
                'direction': eye_data.get('gaze_direction', 'CENTER'),
                'velocity': eye_data.get('eye_movement_velocity', 0.0),
                'blink_count': eye_data.get('blink_count', 0),
                'is_blinking': eye_data.get('is_blinking', False)
            }
        
        return comprehensive_data
    
    def log_medical_data(self, emotion_data, eye_data, timestamp, frame=None, face_bbox=None):
        entry = {
            'timestamp': timestamp.isoformat(),
            'session_time': (timestamp - self.session_start).total_seconds(),
            'emotion': emotion_data,
            'eye_tracking': eye_data
        }
        
        self.medical_log.append(entry)
        
        if self.print_mode == 'medical':
            self.print_medical_status(emotion_data, eye_data, timestamp)
        elif self.print_mode == 'gemma_3n' and frame is not None and face_bbox is not None:
            comprehensive_data = self.generate_comprehensive_analysis(frame, face_bbox, emotion_data, eye_data, timestamp)
            self.print_gemma_3n_format(comprehensive_data)
        elif self.print_mode == 'smart_gemma_3n' and frame is not None and face_bbox is not None:
            pattern_result = self.pattern_detector.should_trigger_update(emotion_data, eye_data, timestamp)
            
            if pattern_result:
                comprehensive_data = self.generate_comprehensive_analysis(frame, face_bbox, emotion_data, eye_data, timestamp)
                self.print_smart_gemma_3n_format(comprehensive_data, pattern_result)
        elif self.print_mode == 'silent_voice':
            if self.decision_engine and emotion_data:
                safe_eye_data = eye_data if eye_data is not None else {}
                should_call, priority, reason = self.decision_engine.should_call_gemma(
                    emotion_data, safe_eye_data, timestamp.timestamp(), self.context, self.patient_condition
                )
                
                if should_call:
                    silent_voice_result = self.silent_voice.process_detection(
                        emotion_data, eye_data, self.context, self.patient_condition, frame
                    )
                    print(f"🎯 DECISION: {priority.value.upper()} - {reason}")
                    self.print_silent_voice_format(silent_voice_result, emotion_data, eye_data, timestamp)
                else:
                    print(f"⏸️  DECISION: {priority.value.upper()} - {reason}")
            else:
                self.silent_voice_frame_counter += 1
                if self.silent_voice_frame_counter >= 150:
                    self.silent_voice_frame_counter = 0
                    silent_voice_result = self.silent_voice.process_detection(
                        emotion_data, eye_data, self.context, self.patient_condition, frame
                    )
                    self.print_silent_voice_format(silent_voice_result, emotion_data, eye_data, timestamp)
    
    def print_medical_status(self, emotion_data, eye_data, timestamp):
        session_time = (timestamp - self.session_start).total_seconds()
        
        print(f"[{session_time:06.1f}s] ", end="")
        
        if emotion_data:
            print(f"EMOTION: {emotion_data['emotion']} ({emotion_data['confidence']:.2f}) | ", end="")
        
        if eye_data:
            print(f"GAZE: {eye_data['gaze_direction']} | ", end="")
            print(f"BLINKS: {eye_data['blink_count']} | ", end="")
            if eye_data['is_blinking']:
                print("BLINKING | ", end="")
            print(f"EYE_VELOCITY: {eye_data['eye_movement_velocity']:.3f}")
        else:
            print()
    
    def print_gemma_3n_format(self, comprehensive_data):
        print("=" * 80)
        print("GEMMA 3N FORMAT OUTPUT")
        print("=" * 80)
        
        visual_data = comprehensive_data['visual_data']
        
        print(f"Timestamp: {comprehensive_data['timestamp']}")
        print(f"Face Detection: {visual_data['face_detected']} (confidence: {visual_data['confidence']:.3f})")
        
        bbox = visual_data['bounding_box']
        print(f"Bounding Box: ({bbox['x1']:.3f}, {bbox['y1']:.3f}, {bbox['x2']:.3f}, {bbox['y2']:.3f})")
        
        expression = visual_data['expression']
        print(f"Expression: {expression['primary_emotion']} (confidence: {expression['confidence']:.3f}, intensity: {expression['intensity']})")
        if expression['features_detected']:
            print(f"Features Detected: {', '.join(expression['features_detected'])}")
        
        if 'facial_landmarks' in visual_data:
            landmarks = visual_data['facial_landmarks']
            print(f"Left Eye: center({landmarks['left_eye']['center'][0]:.3f}, {landmarks['left_eye']['center'][1]:.3f}), "
                  f"state: {landmarks['left_eye']['state']}, openness: {landmarks['left_eye']['openness']:.3f}")
            print(f"Right Eye: center({landmarks['right_eye']['center'][0]:.3f}, {landmarks['right_eye']['center'][1]:.3f}), "
                  f"state: {landmarks['right_eye']['state']}, openness: {landmarks['right_eye']['openness']:.3f}")
            print(f"Mouth: center({landmarks['mouth']['center'][0]:.3f}, {landmarks['mouth']['center'][1]:.3f}), "
                  f"state: {landmarks['mouth']['state']}, openness: {landmarks['mouth']['openness']:.3f} [DEBUG: Required >0.3 for 'open']")
        
        if 'gaze_tracking' in visual_data:
            gaze = visual_data['gaze_tracking']
            print(f"Gaze: {gaze['direction']} (velocity: {gaze['velocity']:.3f}, blinks: {gaze['blink_count']})")
        
        context = visual_data['visual_context']
        print(f"Visual Context: lighting={context['lighting']}, angle={context['camera_angle']}, "
              f"distance={context['distance']}, movement={context['face_movement']}")
        
        medical = visual_data['medical_markers']
        print(f"Medical: symmetry={medical['facial_symmetry']:.3f}, affected={medical['affected_side']}, "
              f"drooping={medical['drooping']}")
        
        print("=" * 80)
        print()
    
    def print_smart_gemma_3n_format(self, comprehensive_data, pattern_result):
        print("\n" + "🧠" * 20)
        print("SMART GEMMA 3N UPDATE - SIGNIFICANT PATTERN DETECTED")
        print("🧠" * 20)
        
        print("\n🎯 TRIGGERS:")
        for trigger in pattern_result['triggers']:
            print(f"  • {trigger['type'].upper()}: {trigger['description']}")
            print(f"    Confidence: {trigger['confidence']:.2f}")
        
        context = pattern_result['context']
        print(f"\n📊 CONTEXT SUMMARY:")
        if context['dominant_emotion']:
            print(f"  • Dominant emotion: {context['dominant_emotion']}")
        print(f"  • Recent blinks: {context['total_blinks']}")
        if context['emotion_summary']:
            print("  • Emotion breakdown:")
            for emotion, conf in sorted(context['emotion_summary'].items(), key=lambda x: x[1], reverse=True):
                print(f"    - {emotion}: {conf:.2f}")
        
        print("\n📋 COMPREHENSIVE ANALYSIS:")
        visual_data = comprehensive_data['visual_data']
        
        expression = visual_data['expression']
        print(f"  • Expression: {expression['primary_emotion']} ({expression['confidence']:.3f}, intensity: {expression['intensity']})")
        if expression['features_detected']:
            print(f"  • Features: {', '.join(expression['features_detected'])}")
        
        if 'facial_landmarks' in visual_data:
            mouth = visual_data['facial_landmarks']['mouth']
            mar_value = mouth.get('mar', 0.0)
            print(f"  • Mouth: {mouth['state']} (openness: {mouth['openness']:.3f}, MAR: {mar_value:.4f})")
            if self.eye_tracker:
                print(f"    [Thresholds: open>{self.eye_tracker.mouth_open_threshold:.3f}, pain signal>0.5]")
        
        if 'gaze_tracking' in visual_data:
            gaze = visual_data['gaze_tracking']
            print(f"  • Gaze: {gaze['direction']} (velocity: {gaze['velocity']:.3f})")
        
        medical = visual_data['medical_markers']
        print(f"  • Medical: symmetry={medical['facial_symmetry']:.3f}, affected={medical['affected_side']}, drooping={medical['drooping']}")
        
        interpretation = self._generate_interpretation(pattern_result, comprehensive_data)
        print("\n💭 SUGGESTED INTERPRETATION:")
        print(f"  \"{interpretation}\"")
        
        print("\n" + "🧠" * 20 + "\n")
    
    def print_silent_voice_format(self, silent_voice_result, emotion_data, eye_data, timestamp):
        print("\n" + "🗣️" * 20)
        print("SILENT VOICE AI COMMUNICATION")
        print("🗣️" * 20)
        
        session_time = (timestamp - self.session_start).total_seconds()
        print(f"\n⏱️  Session Time: {session_time:.1f}s")
        print(f"📅 Timestamp: {timestamp.strftime('%H:%M:%S')}")
        
        print(f"\n🎯 DETECTION SUMMARY:")
        print(f"  • Emotion: {emotion_data['emotion']} (confidence: {emotion_data['confidence']:.2f})")
        
        if eye_data:
            print(f"  • Gaze: {eye_data.get('gaze_direction', 'CENTER')}")
            print(f"  • Blinks: {eye_data.get('blink_count', 0)}")
            print(f"  • Eye Velocity: {eye_data.get('eye_movement_velocity', 0):.3f}")
            
            if 'facial_landmarks' in eye_data:
                mouth = eye_data['facial_landmarks']['mouth']
                print(f"  • Mouth: {mouth['state']} (openness: {mouth['openness']:.2f})")
                left_eye = eye_data['facial_landmarks']['left_eye']
                right_eye = eye_data['facial_landmarks']['right_eye']
                print(f"  • Eyes: L={left_eye['state']} ({left_eye['openness']:.2f}), R={right_eye['state']} ({right_eye['openness']:.2f})")
        
        detection_log = silent_voice_result.get('detection_log', {})
        
        print(f"\n🧠 SILENT VOICE PROCESSING:")
        print(f"  • Biosignal: \"{detection_log.get('biosignal_description', 'N/A')}\"")
        print(f"  • Signal Type: {detection_log.get('signal_type', 'N/A')}")
        print(f"  • Urgency Level: {detection_log.get('urgency_level', 'N/A')}")
        if detection_log.get('medical_condition'):
            print(f"  • Medical Condition: {detection_log.get('medical_condition')}")
        print(f"  • Context: {detection_log.get('context', 'N/A')}")
        print(f"  • Simple Query: {detection_log.get('is_simple_query', False)}")
        
        visual_analysis = silent_voice_result.get('visual_analysis')
        if visual_analysis and visual_analysis.get('description'):
            print(f"  • Visual Analysis: \"{visual_analysis['description'][:100]}...\"")
            if visual_analysis.get('analysis_time'):
                print(f"    (Analysis took {visual_analysis['analysis_time']:.2f}s)")
        
        print(f"  • Model Input Format:")
        model_input = silent_voice_result['biosignal_input']
        for line in model_input.split('\n'):
            print(f"    {line}")
        
        model_info = silent_voice_result.get('model_info', {})
        if model_info:
            print(f"\n🤖 GEMMA 3N MODEL STATUS:")
            print(f"  • Model Loaded: {'✅' if model_info.get('model_loaded') else '❌'}")
            print(f"  • Device: {model_info.get('device', 'N/A')}")
            print(f"  • Response Type: {model_info.get('response_type', 'N/A')}")
            
            if model_info.get('generation_time'):
                print(f"  • Generation Time: {model_info['generation_time']:.2f}s")
                print(f"  • Output Tokens: {model_info.get('output_tokens', 0)}")
        
        print(f"\n💬 PATIENT COMMUNICATION:")
        print(f"  \"{silent_voice_result['response']}\"")
        
        if model_info.get('response_type') == 'ollama_generated':
            print(f"\n🗣️ SILENT VOICE GEMMA RESPONSE: \"{silent_voice_result['response']}\"")
        
        if self.patient_condition or self.context:
            print(f"\n📋 CONTEXT:")
            if self.patient_condition:
                print(f"  • Condition: {self.patient_condition}")
            if self.context:
                print(f"  • Environment: {self.context}")
        
        print("\n" + "🗣️" * 20 + "\n")
    
    def _generate_interpretation(self, pattern_result, comprehensive_data):
        """Generate a simple interpretation without hardcoded medical assumptions"""
        triggers = pattern_result['triggers']
        primary_trigger = triggers[0]
        
        emotion = comprehensive_data['visual_data']['expression']['primary_emotion']
        
        # Simple, non-medical interpretations
        if primary_trigger['type'] == 'rapid_blinks':
            return "I'm trying to get attention"
        elif primary_trigger['type'] == 'sustained_emotion':
            return f"I'm feeling {emotion} for a while now"
        elif primary_trigger['type'] == 'emotion_escalation':
            return f"My {emotion} feeling is getting stronger"
        elif primary_trigger['type'] == 'double_blink':
            return "I'm acknowledging something"
        elif primary_trigger['type'] == 'circular_gaze':
            return "I'm looking around at something"
        else:
            return f"I'm experiencing {emotion} and trying to communicate"
    
    def _convert_numpy_to_list(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def save_medical_log(self):
        if self.log_file and self.medical_log:
            try:
                serializable_log = self._convert_numpy_to_list(self.medical_log)
                
                log_data = {
                    'session_info': {
                        'start_time': self.session_start.isoformat(),
                        'duration_seconds': (datetime.now() - self.session_start).total_seconds(),
                        'model_used': self.model_size,
                        'eye_tracking_enabled': self.enable_eye_tracking
                    },
                    'data': serializable_log
                }
                
                json_data = json.dumps(log_data, indent=2)
                
                fernet = Fernet(self.encryption_key)
                encrypted_data = fernet.encrypt(json_data.encode())
                
                with open(self.log_file, 'wb') as f:
                    f.write(encrypted_data)
                
                print(f"📄 Medical log saved encrypted to: {self.log_file}")
            except Exception as e:
                print(f"❌ Error saving log: {e}")
    
    def run_medical_monitoring(self):
        if self.is_video_file:
            print(f"📹 Opening video file: {self.video_source}")
            cap = cv2.VideoCapture(self.video_source)
        else:
            print("📷 Starting webcam monitoring")
            cap = cv2.VideoCapture(self.video_source or 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        print("🏥 Medical monitoring started")
        print("📊 Real-time patient analysis active")
        print()
        print("Controls: 'q' - quit, 'c' - capture, 'm' - toggle mode")
        if self.log_file:
            print(f"📄 Logging to: {self.log_file}")
        print()
        
        frame_count = 0
        current_faces = set()
        analysis_start_time = None
        startup_delay = 5.0  # 5-second delay before analysis starts
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if not self.is_video_file:
                    frame = cv2.flip(frame, 1)
                
                timestamp = datetime.now()
                session_time = (timestamp - self.session_start).total_seconds()
                
                # Check if we're still in the startup delay period
                if analysis_start_time is None:
                    if session_time >= startup_delay:
                        analysis_start_time = timestamp
                        print(f"✅ Analysis started after {startup_delay}s delay")
                        print()
                    else:
                        # Show countdown on frame
                        remaining = startup_delay - session_time
                        countdown_text = f"Starting analysis in {remaining:.1f}s..."
                        cv2.putText(frame, countdown_text, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Still show session info but skip face detection
                        medical_info = f"Session: {session_time:.1f}s | Mode: {self.print_mode} | Waiting..."
                        cv2.putText(frame, medical_info, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        cv2.imshow('Medical Emotion & Eye Tracking', frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        
                        frame_count += 1
                        continue
                
                # Normal processing after delay
                if psutil.cpu_percent() > 80:
                    time.sleep(0.05)
                
                # Get multi-modal data
                hr_data = self.heart_rate_monitor.get_heart_rate()
                
                faces = self.detect_faces(frame)
                new_faces = set()
                
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = face['bbox']
                    face_id = f"patient_{i}"
                    new_faces.add(face_id)
                    
                    padding = 20
                    face_img = frame[max(0, y1-padding):min(frame.shape[0], y2+padding),
                                     max(0, x1-padding):min(frame.shape[1], x2+padding)]
                    
                    if face_img.size > 0:
                        emotion, confidence = self.recognize_emotion(face_img, face)
                        emotion_data = {
                            'emotion': emotion,
                            'confidence': confidence,
                            'face_id': face_id
                        }
                        emotion_data['hr_data'] = hr_data
                        
                        if self.enable_eye_tracking:
                            eye_data = self.eye_tracker.analyze_eyes(frame, face_img)
                        else:
                            eye_data = None
                        
                        current_time = timestamp
                        if face_id not in self.previous_emotions:
                            self.emotion_start_times[face_id] = current_time
                            if self.print_mode in ['medical', 'gemma_3n', 'smart_gemma_3n', 'silent_voice', 'on_change']:
                                self.log_medical_data(emotion_data, eye_data, timestamp, frame, face['bbox'])
                        else:
                            prev_emotion = self.previous_emotions[face_id]['emotion']
                            if emotion != prev_emotion:
                                duration = (current_time - self.emotion_start_times[face_id]).total_seconds()
                                emotion_data['duration'] = duration
                                self.emotion_start_times[face_id] = current_time
                                if self.print_mode in ['medical', 'gemma_3n', 'smart_gemma_3n', 'silent_voice', 'on_change']:
                                    self.log_medical_data(emotion_data, eye_data, timestamp, frame, face['bbox'])
                            elif self.print_mode == 'smart_gemma_3n':
                                self.log_medical_data(emotion_data, eye_data, timestamp, frame, face['bbox'])
                            elif self.print_mode == 'silent_voice':
                                self.log_medical_data(emotion_data, eye_data, timestamp, frame, face['bbox'])
                        
                        self.previous_emotions[face_id] = {
                            'emotion': emotion,
                            'confidence': confidence,
                            'timestamp': current_time
                        }
                        
                        if self.emotion_mode == 'yolo':
                            colors = {
                                'Pain': (0, 0, 255),
                                'Distress': (0, 128, 255),
                                'Happy': (0, 255, 0),
                                'Concentration': (255, 255, 0),
                                'Neutral': (128, 128, 128),
                                'Unknown': (64, 64, 64)
                            }
                        else:
                            colors = {
                                'Happy': (0, 255, 0), 'Sad': (255, 0, 0), 'Angry': (0, 0, 255),
                                'Surprise': (255, 255, 0), 'Fear': (128, 0, 128), 'Disgust': (0, 128, 128),
                                'Neutral': (128, 128, 128), 'Unknown': (64, 64, 64)
                            }
                        
                        color = colors.get(emotion, (0, 255, 0))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        if eye_data:
                            info_y = y1 - 40
                            cv2.putText(frame, f"Gaze: {eye_data['gaze_direction']}", (x1, info_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            info_y -= 20
                            cv2.putText(frame, f"Blinks: {eye_data['blink_count']}", (x1, info_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                            padding_offset = 20
                            if 'left_eye' in eye_data:
                                for point in eye_data['left_eye']:
                                    cv2.circle(frame, (int(point[0] + x1 - padding_offset), int(point[1] + y1 - padding_offset)), 1, (0, 255, 255), -1)
                            if 'right_eye' in eye_data:
                                for point in eye_data['right_eye']:
                                    cv2.circle(frame, (int(point[0] + x1 - padding_offset), int(point[1] + y1 - padding_offset)), 1, (0, 255, 255), -1)
                            
                            if 'outer_mouth' in eye_data:
                                for point in eye_data['outer_mouth']:
                                    cv2.circle(frame, (int(point[0] + x1 - padding_offset), int(point[1] + y1 - padding_offset)), 2, (255, 0, 255), -1)
                            if 'inner_mouth' in eye_data:
                                for point in eye_data['inner_mouth']:
                                    cv2.circle(frame, (int(point[0] + x1 - padding_offset), int(point[1] + y1 - padding_offset)), 1, (255, 128, 255), -1)
                            
                            if 'mouth_mar' in eye_data:
                                cv2.putText(frame, f"MAR: {eye_data['mouth_mar']:.3f}", (x1, y2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                disappeared_faces = current_faces - new_faces
                for face_id in disappeared_faces:
                    if face_id in self.previous_emotions:
                        del self.previous_emotions[face_id]
                        del self.emotion_start_times[face_id]
                
                current_faces = new_faces
                
                session_time = (timestamp - self.session_start).total_seconds()
                medical_info = f"Session: {session_time:.1f}s | Mode: {self.print_mode} | Patients: {len(faces)}"
                cv2.putText(frame, medical_info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Medical Emotion & Eye Tracking', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    filename = f'medical_capture_{timestamp_str}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"📸 Medical capture saved: {filename}")
                elif key == ord('m'):
                    modes = ['medical', 'gemma_3n', 'smart_gemma_3n', 'silent_voice', 'on_change', 'continuous', 'silent']
                    current_index = modes.index(self.print_mode)
                    self.print_mode = modes[(current_index + 1) % len(modes)]
                    print(f"🔄 Mode changed to: {self.print_mode}")
                    if self.print_mode == 'smart_gemma_3n':
                        print("   Smart mode: Only significant patterns trigger updates")
                    elif self.print_mode == 'silent_voice':
                        print("   Silent Voice mode: AI communication responses")
                        if self.patient_condition:
                            print(f"   Patient condition: {self.patient_condition}")
                        if self.context:
                            print(f"   Context: {self.context}")
                
                frame_count += 1
                
                # After emotion detection
                self.predictive_analytics.add_emotion(emotion)
                predicted = self.predictive_analytics.predict_next_emotion()
                if predicted != "Insufficient data":
                    print(f"🔮 Predicted next emotion: {predicted}")
                
        except KeyboardInterrupt:
            print("\n🛑 Medical monitoring stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_medical_log()
            
            session_duration = (datetime.now() - self.session_start).total_seconds()
            print(f"\n📊 Medical Session Summary:")
            print(f"   Duration: {session_duration:.1f} seconds")
            print(f"   Total frames: {frame_count}")
            print(f"   Log entries: {len(self.medical_log)}")
            if self.eye_tracker:
                print(f"   Total blinks detected: {self.eye_tracker.blink_count}")
            
            if self.decision_engine:
                stats = self.decision_engine.get_session_stats()
                print(f"\n🧠 DECISION ENGINE STATISTICS:")
                print(f"   Total AI calls made: {stats['total_calls']}")
                print(f"   Remaining budget: {stats['remaining_budget']}")
                print(f"   Events processed: {stats['events_in_history']}")
                print(f"   Call efficiency: {(stats['total_calls'] / max(1, len(self.medical_log))) * 100:.1f}% of detections")
                
                if self.log_file:
                    log_dir = os.path.dirname(self.log_file)
                    log_base = os.path.basename(self.log_file)
                    decision_log_base = log_base.replace('.json', '_decisions.json')
                    
                    if log_dir:
                        decision_log_file = os.path.join(log_dir, decision_log_base)
                    else:
                        decision_log_file = decision_log_base
                    
                    self.decision_engine.export_decision_log(decision_log_file)

class HeartRateMonitor:
    def __init__(self):
        self.current_hr = 70  # Simulated resting heart rate
        self.hr_history = deque(maxlen=30)  # Last 30 seconds of HR
        print("✅ Heart Rate Monitor initialized (simulated for prototype)")
    
    def get_heart_rate(self):
        # Simulate heart rate variation based on time
        variation = random.uniform(-5, 5)
        self.current_hr = max(60, min(100, self.current_hr + variation))
        self.hr_history.append(self.current_hr)
        avg_hr = sum(self.hr_history) / len(self.hr_history) if self.hr_history else 70
        stress_level = 'normal' if avg_hr < 80 else 'elevated' if avg_hr < 90 else 'high'
        return {
            'current_hr': self.current_hr,
            'avg_hr': avg_hr,
            'stress_level': stress_level
        }

class PredictiveAnalytics:
    def __init__(self):
        self.model = self.build_lstm_model()
        self.emotion_history = deque(maxlen=30)  # Last 30 readings
        self.emotion_map = {'Pain': 0, 'Distress': 1, 'Happy': 2, 'Concentration': 3, 'Neutral': 4, 'Unknown': 5}
        print("✅ Predictive Analytics initialized with LSTM model")
    
    def build_lstm_model(self):
        model = Sequential()
        model.add(LSTM(32, input_shape=(30, 1), return_sequences=False))
        model.add(Dense(6, activation='softmax'))  # 6 emotion classes
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model
    
    def add_emotion(self, emotion):
        self.emotion_history.append(self.emotion_map.get(emotion, 5))
    
    def predict_next_emotion(self):
        if len(self.emotion_history) < 30:
            return "Insufficient data"
        input_data = np.array(self.emotion_history).reshape(1, 30, 1)
        prediction = self.model.predict(input_data)
        predicted_class = np.argmax(prediction)
        return list(self.emotion_map.keys())[predicted_class]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Silent Voice Medical Recognition with Eye Tracking for Paralysis Patients')
    parser.add_argument('--video', '-v', type=str, help='Path to video file (if not provided, uses webcam)')
    parser.add_argument('--webcam', '-w', type=int, default=0, help='Webcam index (default: 0)')
    parser.add_argument('--model', '-m', type=str, choices=['n', 's', 'm', 'l', 'x'], default='m',
                        help='YOLO model size (default: medium for medical accuracy)')
    parser.add_argument('--no-eye-tracking', action='store_true', help='Disable eye tracking')
    parser.add_argument('--log', '-l', type=str, help='Save medical log to file (JSON format)')
    parser.add_argument('--gemma-3n', action='store_true', help='Use Gemma 3n format output (comprehensive analysis)')
    parser.add_argument('--smart', action='store_true', help='Use smart Gemma 3n mode (only significant patterns trigger updates)')
    parser.add_argument('--silent-voice', action='store_true', help='Use Silent Voice AI communication mode')
    parser.add_argument('--silent-voice-model', type=str, help='Path to trained Silent Voice model directory')
    parser.add_argument('--patient-condition', type=str, help='Patient medical condition (e.g., "ALS patient (advanced)")')
    parser.add_argument('--context', type=str, default='monitoring session', help='Current context/environment (default: monitoring session)')
    return parser.parse_args()

def main():
    print("🗣️ Silent Voice Medical Recognition with Eye Tracking")
    print("AI-powered communication system for paralysis patient monitoring")
    print("Features: Emotion analysis, gaze tracking, blink detection, AI communication")
    print("Output formats: Medical summary, Gemma 3n analysis, Silent Voice AI responses")
    print()
    
    if not DEEPFACE_AVAILABLE:
        print("⚠️  DeepFace not available - install with: pip install deepface tensorflow")
    
    if not MEDIAPIPE_AVAILABLE:
        print("⚠️  MediaPipe not available - install with: pip install mediapipe")
        print("Eye tracking will be disabled")
    
    print("\n📖 USAGE EXAMPLES:")
    print("• Basic medical monitoring:")
    print("  python emotion_recognition_medical.py")
    print("• Silent Voice AI communication:")
    print("  python emotion_recognition_medical.py --silent-voice --silent-voice-model path/to/model --patient-condition 'ALS patient (advanced)'")
    print("• Video analysis with Silent Voice:")
    print("  python emotion_recognition_medical.py --video patient_video.mp4 --silent-voice --context 'ICU room'")
    print("• Smart pattern detection:")
    print("  python emotion_recognition_medical.py --smart --log session.json")
    print()
    
    args = parse_arguments()
    
    if args.video:
        if not os.path.exists(args.video):
            print(f"❌ Error: Video file '{args.video}' not found")
            sys.exit(1)
        video_source = args.video
        print(f"📹 Using video file: {args.video}")
    else:
        video_source = args.webcam
        print(f"📷 Using webcam (index: {args.webcam})")
    
    print(f"🔧 Using YOLO model size: {args.model}")
    
    log_file = args.log
    if not log_file and video_source != args.webcam:
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        log_file = os.path.join(log_dir, f"medical_log_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    try:
        recognizer = MedicalEmotionRecognizer(
            video_source=video_source,
            model_size=args.model,
            enable_eye_tracking=not args.no_eye_tracking,
            log_file=log_file,
            silent_voice_model=args.silent_voice_model,
            patient_condition=args.patient_condition,
            context=args.context
        )
        
        if args.silent_voice:
            recognizer.print_mode = 'silent_voice'
            print("🗣️ Using Silent Voice AI communication mode")
            if args.patient_condition:
                print(f"   Patient condition: {args.patient_condition}")
            if args.context:
                print(f"   Context: {args.context}")
            if args.silent_voice_model:
                print(f"   Model: {args.silent_voice_model}")
            else:
                print("   Using fallback pattern-based responses")
        elif args.smart:
            recognizer.print_mode = 'smart_gemma_3n'
            print("🧠 Using Smart Gemma 3n mode - only significant patterns trigger updates")
            print("   Patterns detected: sustained emotions, rapid blinks, gaze patterns, escalations")
        elif args.gemma_3n:
            recognizer.print_mode = 'gemma_3n'
            print("🧠 Using Gemma 3n format output")
        
        recognizer.run_medical_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
