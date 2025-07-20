#!/usr/bin/env python3
"""
Silent Voice - Text-to-Speech with Emotional Context
Converts Silent Voice AI responses to natural speech with emotional inflection
"""

import os
import sys
import json
import time
from datetime import datetime
import threading
import queue

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️  pyttsx3 not available - install with: pip install pyttsx3")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not available - install with: pip install pygame")

class VoiceSynthesizer:
    def __init__(self, patient_id="default", voice_profile=None):
        self.patient_id = patient_id
        self.voice_profile = voice_profile or self._create_default_profile()
        self.tts_engine = None
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        
        if TTS_AVAILABLE:
            self._initialize_tts()
        else:
            print("⚠️  Text-to-speech not available")
            
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
            
        self.emotion_voices = {
            'pain': {'rate': 120, 'volume': 0.9, 'pitch': -10},
            'distress': {'rate': 130, 'volume': 0.8, 'pitch': -5},
            'happy': {'rate': 160, 'volume': 0.9, 'pitch': 10},
            'neutral': {'rate': 150, 'volume': 0.7, 'pitch': 0},
            'urgent': {'rate': 180, 'volume': 1.0, 'pitch': 15},
            'calm': {'rate': 140, 'volume': 0.6, 'pitch': -2}
        }
        
    def _create_default_profile(self):
        """Create default voice profile for patient"""
        return {
            'base_rate': 150,        # Words per minute
            'base_volume': 0.8,      # Volume level
            'base_pitch': 0,         # Pitch adjustment
            'voice_gender': 'neutral', # male/female/neutral
            'accent': 'american',    # Regional accent
            'age_group': 'adult'     # child/adult/elderly
        }
        
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Set voice properties based on profile
            voices = self.tts_engine.getProperty('voices')
            
            # Select appropriate voice
            target_voice = None
            gender_pref = self.voice_profile.get('voice_gender', 'neutral')
            
            for voice in voices:
                voice_name = voice.name.lower()
                if gender_pref == 'female' and any(term in voice_name for term in ['female', 'woman', 'zira', 'hazel']):
                    target_voice = voice.id
                    break
                elif gender_pref == 'male' and any(term in voice_name for term in ['male', 'man', 'david', 'mark']):
                    target_voice = voice.id
                    break
            
            if target_voice:
                self.tts_engine.setProperty('voice', target_voice)
                
            # Set base properties
            self.tts_engine.setProperty('rate', self.voice_profile['base_rate'])
            self.tts_engine.setProperty('volume', self.voice_profile['base_volume'])
            
            print(f"✅ Voice synthesizer initialized for patient {self.patient_id}")
            
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")
            self.tts_engine = None
    
    def speak_message(self, message, emotion_context=None, urgency_level='medium', biosignal_data=None, blocking=False):
        """
        Convert Silent Voice message to speech with emotional context
        
        Args:
            message: The patient's communication message
            emotion_context: Detected emotion ('pain', 'happy', etc.)
            urgency_level: 'low', 'medium', 'high', 'critical'
            biosignal_data: Additional context from biosignals
            blocking: If True, wait for speech to complete before returning
        """
        if not self.tts_engine:
            print(f"💬 [VOICE SYNTHESIS] {self.patient_id}: \"{message}\"")
            return
            
        # Determine voice settings based on emotion and urgency
        voice_settings = self._get_voice_settings(emotion_context, urgency_level, biosignal_data)
        
        # Apply voice settings
        self._apply_voice_settings(voice_settings)
        
        # Add emotional prefixes/suffixes for context
        enhanced_message = self._enhance_message(message, emotion_context, urgency_level)
        
        # Speak the message
        if blocking:
            # Blocking mode - wait for completion
            self._speak_blocking(enhanced_message)
        else:
            # Non-blocking mode - use thread
            self.speech_thread = threading.Thread(target=self._speak_threaded, args=(enhanced_message,))
            self.speech_thread.start()
        
        # Log the synthesis
        self._log_synthesis(message, enhanced_message, voice_settings, emotion_context, urgency_level)
    
    def _get_voice_settings(self, emotion_context, urgency_level, biosignal_data):
        """Determine optimal voice settings based on context"""
        base_settings = {
            'rate': self.voice_profile['base_rate'],
            'volume': self.voice_profile['base_volume'],
            'pitch': self.voice_profile['base_pitch']
        }
        
        # Apply emotion-based adjustments
        if emotion_context and emotion_context.lower() in self.emotion_voices:
            emotion_settings = self.emotion_voices[emotion_context.lower()]
            base_settings.update(emotion_settings)
        
        # Apply urgency adjustments
        urgency_multipliers = {
            'low': {'rate': 0.9, 'volume': 0.8},
            'medium': {'rate': 1.0, 'volume': 1.0},
            'high': {'rate': 1.1, 'volume': 1.2},
            'critical': {'rate': 1.3, 'volume': 1.5}
        }
        
        if urgency_level in urgency_multipliers:
            multiplier = urgency_multipliers[urgency_level]
            base_settings['rate'] = int(base_settings['rate'] * multiplier['rate'])
            base_settings['volume'] = min(1.0, base_settings['volume'] * multiplier['volume'])
        
        # Apply biosignal adjustments
        if biosignal_data:
            if 'heart_rate' in biosignal_data:
                hr = biosignal_data['heart_rate']
                if hr > 90:  # Elevated heart rate
                    base_settings['rate'] = int(base_settings['rate'] * 1.15)
                    base_settings['volume'] = min(1.0, base_settings['volume'] * 1.1)
                elif hr < 60:  # Low heart rate (calm)
                    base_settings['rate'] = int(base_settings['rate'] * 0.9)
                    base_settings['volume'] = base_settings['volume'] * 0.9
        
        return base_settings
    
    def _apply_voice_settings(self, settings):
        """Apply voice settings to TTS engine"""
        try:
            self.tts_engine.setProperty('rate', settings['rate'])
            self.tts_engine.setProperty('volume', settings['volume'])
            # Note: pitch adjustment would require more advanced TTS engines
        except Exception as e:
            print(f"⚠️  Error applying voice settings: {e}")
    
    def _enhance_message(self, message, emotion_context, urgency_level):
        """Add emotional context to message delivery"""
        enhanced = message
        
        # Add urgency indicators
        if urgency_level == 'critical':
            enhanced = f"URGENT: {enhanced}"
        elif urgency_level == 'high':
            enhanced = f"Important: {enhanced}"
        
        # Add emotional breathing/pauses for natural delivery
        if emotion_context == 'pain':
            enhanced = enhanced.replace('.', '... ').replace(',', '... ')
        elif emotion_context == 'happy':
            enhanced = enhanced.replace('.', '! ')
        elif emotion_context == 'distress':
            enhanced = enhanced.replace('.', '. Please help me.')
        
        return enhanced
    
    def _speak_threaded(self, message):
        """Speak message in separate thread to avoid blocking"""
        try:
            self.is_speaking = True
            print(f"🔊 [VOICE] {self.patient_id}: Speaking...")
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
            self.is_speaking = False
            print(f"✅ [VOICE] {self.patient_id}: Speech completed")
        except Exception as e:
            print(f"❌ [VOICE] Error during speech: {e}")
            self.is_speaking = False
    
    def _speak_blocking(self, message):
        """Speak message and wait for completion"""
        try:
            self.is_speaking = True
            print(f"🔊 [VOICE] {self.patient_id}: Speaking...")
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
            self.is_speaking = False
            print(f"✅ [VOICE] {self.patient_id}: Speech completed")
        except Exception as e:
            print(f"❌ [VOICE] Error during speech: {e}")
            self.is_speaking = False
    
    def wait_for_speech(self):
        """Wait for any ongoing speech to complete"""
        if hasattr(self, 'speech_thread') and self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=30)  # Max 30 seconds wait
    
    def _log_synthesis(self, original_message, enhanced_message, voice_settings, emotion_context, urgency_level):
        """Log voice synthesis details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': self.patient_id,
            'original_message': original_message,
            'enhanced_message': enhanced_message,
            'voice_settings': voice_settings,
            'emotion_context': emotion_context,
            'urgency_level': urgency_level
        }
        
        # Save to voice log
        log_file = f"log/voice_synthesis_{self.patient_id}_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Error logging voice synthesis: {e}")
    
    def is_currently_speaking(self):
        """Check if currently speaking"""
        return self.is_speaking
    
    def stop_speaking(self):
        """Stop current speech"""
        if self.tts_engine and self.is_speaking:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
            except Exception as e:
                print(f"⚠️  Error stopping speech: {e}")
    
    def adjust_voice_profile(self, new_profile):
        """Update voice profile for patient"""
        self.voice_profile.update(new_profile)
        if self.tts_engine:
            self._apply_voice_settings(self.voice_profile)
        print(f"✅ Voice profile updated for patient {self.patient_id}")

class VoiceManager:
    """Manages voice synthesis for multiple patients"""
    
    def __init__(self):
        self.patient_voices = {}
        self.active_speakers = set()
        
    def get_voice_synthesizer(self, patient_id, voice_profile=None):
        """Get or create voice synthesizer for patient"""
        if patient_id not in self.patient_voices:
            self.patient_voices[patient_id] = VoiceSynthesizer(patient_id, voice_profile)
        return self.patient_voices[patient_id]
    
    def speak_for_patient(self, patient_id, message, emotion_context=None, urgency_level='medium', biosignal_data=None, blocking=False):
        """
        Speak a message for a specific patient
        
        Args:
            patient_id: Patient identifier
            message: Message to speak
            emotion_context: Detected emotion
            urgency_level: Message urgency
            biosignal_data: Additional biosignal context
            blocking: If True, wait for speech to complete
        """
        if patient_id not in self.patient_voices:
            self.patient_voices[patient_id] = VoiceSynthesizer(patient_id)
        
        synthesizer = self.patient_voices[patient_id]
        
        # Check urgency and potentially interrupt
        if urgency_level == 'critical' and synthesizer.is_speaking:
            print(f"🚨 [VOICE] Interrupting current speech for critical message")
            # In a real implementation, would stop current speech
        
        # Queue or speak message
        synthesizer.speak_message(message, emotion_context, urgency_level, biosignal_data, blocking=blocking)
        self.active_patient = patient_id
    
    def wait_for_all_speech(self):
        """Wait for all patients' speech to complete"""
        for patient_id, synthesizer in self.patient_voices.items():
            synthesizer.wait_for_speech()

# Global voice manager instance
voice_manager = VoiceManager()

def speak_silent_voice_message(patient_id, message, emotion_context=None, urgency_level='medium', biosignal_data=None, blocking=False):
    """
    Main function to convert Silent Voice AI response to speech
    
    Usage:
        speak_silent_voice_message("P001", "I need help with my breathing", 
                                 emotion_context="distress", urgency_level="high",
                                 blocking=True)  # Wait for completion
    """
    voice_manager.speak_for_patient(patient_id, message, emotion_context, urgency_level, biosignal_data, blocking=blocking)

if __name__ == "__main__":
    # Demo the voice synthesis
    print("🔊 Silent Voice - Voice Synthesis Demo")
    
    demo_messages = [
        {
            'patient_id': 'P001',
            'message': 'I need help with my breathing.',
            'emotion_context': 'distress',
            'urgency_level': 'critical'
        },
        {
            'patient_id': 'P002', 
            'message': 'Thank you for adjusting my pillow.',
            'emotion_context': 'happy',
            'urgency_level': 'low'
        },
        {
            'patient_id': 'P003',
            'message': 'Something feels different today.',
            'emotion_context': 'concentration',
            'urgency_level': 'medium'
        }
    ]
    
    for demo in demo_messages:
        print(f"\n📢 Demo: {demo['patient_id']} - {demo['message']}")
        speak_silent_voice_message(**demo)
        time.sleep(3)  # Wait between demos
    
    print("\n✅ Voice synthesis demo completed") 