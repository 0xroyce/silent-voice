#!/usr/bin/env python3
"""
Gemma 3n Decision Engine for Silent Voice Medical System
========================================================

Intelligent gatekeeper that decides when to send data to the expensive Gemma 3n model.
Uses medical decision logic to minimize API calls while ensuring critical events are captured.

Key Features:
- Clinical significance scoring
- Temporal pattern analysis
- Cost-aware decision making
- Medical priority rules
- Configurable thresholds

Author: Silent Voice Medical Team
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    """Medical priority levels for AI communication"""
    CRITICAL = "critical"      # Immediate AI response needed
    HIGH = "high"              # Important but not urgent
    MEDIUM = "medium"          # Routine monitoring
    LOW = "low"                # Background information
    IGNORE = "ignore"          # Not worth AI processing


@dataclass
class MedicalEvent:
    """Represents a medical event for decision processing"""
    timestamp: float
    emotion: str
    confidence: float
    intensity: str
    gaze_direction: str
    blink_count: int
    eye_velocity: float
    mouth_state: str
    mouth_openness: float
    context: str
    patient_condition: str
    session_id: str


class GemmaDecisionEngine:
    """Intelligent decision engine for Gemma 3n API calls"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the decision engine with medical rules"""
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Medical decision thresholds
        self.CRITICAL_EMOTIONS = {'fear', 'sad', 'angry'}
        self.POSITIVE_EMOTIONS = {'happy', 'neutral'}
        self.DISTRESS_EMOTIONS = {'fear', 'sad', 'angry', 'disgust'}
        
        # Temporal thresholds
        self.MIN_TIME_BETWEEN_CALLS = 30.0  # Minimum 30 seconds between calls
        self.CRITICAL_OVERRIDE_TIME = 10.0   # Critical events can override after 10s
        self.SUSTAINED_EMOTION_TIME = 45.0   # Must sustain 45s for non-critical
        
        # Medical significance thresholds
        self.HIGH_CONFIDENCE_THRESHOLD = 0.8
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.6
        self.PAIN_SIGNAL_THRESHOLD = 0.4  # Mouth openness for pain
        
        # Pattern tracking
        self.event_history: List[MedicalEvent] = []
        self.last_api_call_time: float = 0.0
        self.last_critical_time: float = 0.0
        self.current_session_calls: int = 0
        self.max_calls_per_session: int = 20  # Budget limit
        
        # State tracking
        self.sustained_emotion_start: Optional[float] = None
        self.current_sustained_emotion: Optional[str] = None
        self.escalation_detected: bool = False
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "min_time_between_calls": 30.0,
            "critical_override_time": 10.0,
            "sustained_emotion_time": 45.0,
            "max_calls_per_session": 20,
            "enable_cost_optimization": True,
            "enable_medical_rules": True,
            "debug_mode": False
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except FileNotFoundError:
                print(f"Config file {config_file} not found, using defaults")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging for decision tracking"""
        logging.basicConfig(
            level=logging.INFO if self.config.get("debug_mode", False) else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("GemmaDecisionEngine")
    
    def should_call_gemma(self, emotion_data: Dict, eye_data: Dict, timestamp: float, 
                         context: str = "hospital bed", patient_condition: str = "ALS patient") -> Tuple[bool, Priority, str]:
        """
        Main decision function: Should we call Gemma 3n?
        
        Returns:
            Tuple of (should_call, priority, reason)
        """
        
        # Create medical event with safe attribute access
        facial_landmarks = eye_data.get('facial_landmarks', {}) if eye_data else {}
        mouth_data = facial_landmarks.get('mouth', {}) if facial_landmarks else {}
        
        event = MedicalEvent(
            timestamp=timestamp,
            emotion=emotion_data.get('emotion', 'unknown').lower(),
            confidence=emotion_data.get('confidence', 0.0),
            intensity=emotion_data.get('intensity', 'unknown'),
            gaze_direction=eye_data.get('gaze_direction', 'unknown') if eye_data else 'unknown',
            blink_count=eye_data.get('blink_count', 0) if eye_data else 0,
            eye_velocity=eye_data.get('eye_movement_velocity', 0.0) if eye_data else 0.0,
            mouth_state=mouth_data.get('state', 'unknown'),
            mouth_openness=mouth_data.get('openness', 0.0),
            context=context,
            patient_condition=patient_condition,
            session_id=str(int(timestamp))
        )
        
        # Add to history
        self.event_history.append(event)
        self._cleanup_old_events(timestamp)
        
        # Check budget constraints
        if self.current_session_calls >= self.max_calls_per_session:
            return False, Priority.IGNORE, "Session budget exceeded"
        
        # Apply medical decision rules
        priority, reason = self._assess_medical_priority(event)
        
        # Check timing constraints
        time_since_last = timestamp - self.last_api_call_time
        
        if priority == Priority.CRITICAL:
            # Critical events can override timing after shorter interval
            if time_since_last >= self.CRITICAL_OVERRIDE_TIME:
                self.last_api_call_time = timestamp
                self.last_critical_time = timestamp
                self.current_session_calls += 1
                self.logger.info(f"CRITICAL: {reason}")
                return True, priority, reason
            else:
                return False, priority, f"Critical cooling down ({time_since_last:.1f}s < {self.CRITICAL_OVERRIDE_TIME}s)"
        
        elif priority == Priority.HIGH:
            # High priority needs standard timing
            if time_since_last >= self.MIN_TIME_BETWEEN_CALLS:
                self.last_api_call_time = timestamp
                self.current_session_calls += 1
                self.logger.info(f"HIGH: {reason}")
                return True, priority, reason
            else:
                return False, priority, f"High priority cooling down ({time_since_last:.1f}s < {self.MIN_TIME_BETWEEN_CALLS}s)"
        
        elif priority == Priority.MEDIUM:
            # Medium priority needs longer interval and sustained pattern
            if time_since_last >= self.SUSTAINED_EMOTION_TIME:
                sustained_reason = self._check_sustained_pattern(event)
                if sustained_reason:
                    self.last_api_call_time = timestamp
                    self.current_session_calls += 1
                    self.logger.info(f"MEDIUM: {sustained_reason}")
                    return True, priority, sustained_reason
                else:
                    return False, priority, "Medium priority needs sustained pattern"
            else:
                return False, priority, f"Medium priority cooling down ({time_since_last:.1f}s < {self.SUSTAINED_EMOTION_TIME}s)"
        
        # LOW and IGNORE priorities don't trigger calls
        return False, priority, reason
    
    def _assess_medical_priority(self, event: MedicalEvent) -> Tuple[Priority, str]:
        """Assess medical priority based on clinical significance"""
        
        # CRITICAL: Immediate medical attention needed
        if self._is_critical_event(event):
            return Priority.CRITICAL, self._get_critical_reason(event)
        
        # HIGH: Important medical event
        if self._is_high_priority_event(event):
            return Priority.HIGH, self._get_high_priority_reason(event)
        
        # MEDIUM: Routine monitoring
        if self._is_medium_priority_event(event):
            return Priority.MEDIUM, self._get_medium_priority_reason(event)
        
        # LOW: Background information
        if self._is_low_priority_event(event):
            return Priority.LOW, self._get_low_priority_reason(event)
        
        # IGNORE: Not medically significant
        return Priority.IGNORE, "Not medically significant"
    
    def _is_critical_event(self, event: MedicalEvent) -> bool:
        """Check if event requires immediate attention"""
        
        # Severe pain signals
        if (event.emotion in ['fear', 'sad'] and 
            event.confidence > 0.9 and 
            event.mouth_openness > self.PAIN_SIGNAL_THRESHOLD):
            return True
        
        # Extreme distress with high confidence
        if (event.emotion == 'fear' and 
            event.confidence > 0.95 and 
            event.intensity == 'extreme'):
            return True
        
        # Rapid escalation pattern
        if self._detect_escalation_pattern(event):
            return True
        
        # Emergency blink patterns (5+ blinks in rapid succession)
        if event.blink_count >= 5 and self._is_rapid_blink_pattern():
            return True
        
        return False
    
    def _is_high_priority_event(self, event: MedicalEvent) -> bool:
        """Check if event is high priority"""
        
        # High confidence distress
        if (event.emotion in self.DISTRESS_EMOTIONS and 
            event.confidence > self.HIGH_CONFIDENCE_THRESHOLD):
            return True
        
        # Very confident positive emotions (patient clearly expressing joy/gratitude)
        if (event.emotion in self.POSITIVE_EMOTIONS and 
            event.confidence > 0.9 and
            event.emotion != 'neutral'):  # Exclude neutral from high priority
            return True
        
        # Sustained high-intensity emotion
        if (event.intensity in ['high', 'extreme'] and 
            event.confidence > self.MEDIUM_CONFIDENCE_THRESHOLD):
            return True
        
        # Significant state change
        if self._detect_significant_state_change(event):
            return True
        
        return False
    
    def _is_medium_priority_event(self, event: MedicalEvent) -> bool:
        """Check if event is medium priority"""
        
        # Moderate confidence distress
        if (event.emotion in self.DISTRESS_EMOTIONS and 
            event.confidence > self.MEDIUM_CONFIDENCE_THRESHOLD):
            return True
        
        # Significant positive emotions (patient expressing happiness/gratitude)
        if (event.emotion in self.POSITIVE_EMOTIONS and 
            event.confidence > 0.85 and
            event.intensity in ['high', 'extreme']):
            return True
        
        # Gaze pattern changes
        if self._detect_gaze_pattern_change(event):
            return True
        
        return False
    
    def _is_low_priority_event(self, event: MedicalEvent) -> bool:
        """Check if event is low priority"""
        
        # Positive emotions (good for morale tracking)
        if event.emotion in self.POSITIVE_EMOTIONS and event.confidence > 0.7:
            return True
        
        # Low confidence distress
        if (event.emotion in self.DISTRESS_EMOTIONS and 
            event.confidence > 0.4):
            return True
        
        return False
    
    def _get_critical_reason(self, event: MedicalEvent) -> str:
        """Get reason for critical priority"""
        reasons = []
        
        if event.mouth_openness > self.PAIN_SIGNAL_THRESHOLD:
            reasons.append("potential pain signal")
        
        if event.confidence > 0.95:
            reasons.append("extreme confidence")
        
        if self.escalation_detected:
            reasons.append("escalation pattern")
        
        return f"Critical: {', '.join(reasons)}"
    
    def _get_high_priority_reason(self, event: MedicalEvent) -> str:
        """Get reason for high priority"""
        if event.emotion in self.POSITIVE_EMOTIONS and event.emotion != 'neutral':
            return f"Positive communication: {event.emotion} ({event.confidence:.2f})"
        return f"High priority: {event.emotion} ({event.confidence:.2f}) - {event.intensity}"
    
    def _get_medium_priority_reason(self, event: MedicalEvent) -> str:
        """Get reason for medium priority"""
        if event.emotion in self.POSITIVE_EMOTIONS:
            return f"Positive expression: {event.emotion} ({event.confidence:.2f})"
        return f"Medium priority: {event.emotion} pattern"
    
    def _get_low_priority_reason(self, event: MedicalEvent) -> str:
        """Get reason for low priority"""
        return f"Low priority: {event.emotion} monitoring"
    
    def _detect_escalation_pattern(self, event: MedicalEvent) -> bool:
        """Detect if emotions are escalating in intensity"""
        if len(self.event_history) < 3:
            return False
        
        recent_events = self.event_history[-3:]
        
        # Check for increasing confidence in distress emotions
        distress_confidences = [e.confidence for e in recent_events 
                              if e.emotion in self.DISTRESS_EMOTIONS]
        
        if len(distress_confidences) >= 2:
            # Check if confidence is increasing
            if distress_confidences[-1] > distress_confidences[-2] + 0.1:
                self.escalation_detected = True
                return True
        
        return False
    
    def _detect_significant_state_change(self, event: MedicalEvent) -> bool:
        """Detect significant emotional state changes"""
        if len(self.event_history) < 2:
            return False
        
        prev_event = self.event_history[-2]
        
        # Distress to positive or vice versa
        if ((event.emotion in self.DISTRESS_EMOTIONS and 
             prev_event.emotion in self.POSITIVE_EMOTIONS) or
            (event.emotion in self.POSITIVE_EMOTIONS and 
             prev_event.emotion in self.DISTRESS_EMOTIONS)):
            return True
        
        # Significant confidence change
        if abs(event.confidence - prev_event.confidence) > 0.3:
            return True
        
        return False
    
    def _detect_gaze_pattern_change(self, event: MedicalEvent) -> bool:
        """Detect changes in gaze patterns"""
        if len(self.event_history) < 2:
            return False
        
        prev_event = self.event_history[-2]
        return event.gaze_direction != prev_event.gaze_direction
    
    def _is_rapid_blink_pattern(self) -> bool:
        """Check for rapid blinking pattern in recent history"""
        recent_time = time.time() - 5.0  # Last 5 seconds
        recent_events = [e for e in self.event_history if e.timestamp >= recent_time]
        
        total_blinks = sum(e.blink_count for e in recent_events)
        return total_blinks >= 5
    
    def _check_sustained_pattern(self, event: MedicalEvent) -> Optional[str]:
        """Check if there's a sustained emotion pattern worth reporting"""
        if event.emotion != self.current_sustained_emotion:
            self.sustained_emotion_start = event.timestamp
            self.current_sustained_emotion = event.emotion
            return None
        
        if self.sustained_emotion_start is None:
            return None
        
        duration = event.timestamp - self.sustained_emotion_start
        if duration >= self.SUSTAINED_EMOTION_TIME:
            return f"Sustained {event.emotion} for {duration:.1f}s"
        
        return None
    
    def _cleanup_old_events(self, current_time: float):
        """Remove old events from history"""
        cutoff_time = current_time - 300.0  # Keep last 5 minutes
        self.event_history = [e for e in self.event_history if e.timestamp >= cutoff_time]
    
    def get_session_stats(self) -> Dict:
        """Get statistics about current session"""
        return {
            'total_calls': self.current_session_calls,
            'remaining_budget': self.max_calls_per_session - self.current_session_calls,
            'last_call_time': self.last_api_call_time,
            'last_critical_time': self.last_critical_time,
            'events_in_history': len(self.event_history)
        }
    
    def reset_session(self):
        """Reset session counters"""
        self.current_session_calls = 0
        self.last_api_call_time = 0.0
        self.last_critical_time = 0.0
        self.event_history.clear()
        self.sustained_emotion_start = None
        self.current_sustained_emotion = None
        self.escalation_detected = False
    
    def export_decision_log(self, filename: str):
        """Export decision history for analysis"""
        log_data = {
            'session_stats': self.get_session_stats(),
            'config': self.config,
            'events': [
                {
                    'timestamp': e.timestamp,
                    'emotion': e.emotion,
                    'confidence': e.confidence,
                    'intensity': e.intensity,
                    'gaze_direction': e.gaze_direction,
                    'blink_count': e.blink_count,
                    'mouth_state': e.mouth_state,
                    'mouth_openness': e.mouth_openness
                } for e in self.event_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Decision log exported to {filename}")


def create_default_config(filename: str = "gemma_decision_config.json"):
    """Create a default configuration file"""
    config = {
        "min_time_between_calls": 30.0,
        "critical_override_time": 10.0,
        "sustained_emotion_time": 45.0,
        "max_calls_per_session": 20,
        "enable_cost_optimization": True,
        "enable_medical_rules": True,
        "debug_mode": False,
        "description": "Gemma 3n Decision Engine Configuration - Adjust these values to control AI call frequency"
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Default configuration created: {filename}")


if __name__ == "__main__":
    # Create default config if run directly
    create_default_config()
    
    # Example usage
    decision_engine = GemmaDecisionEngine()
    
    # Simulate some events
    test_events = [
        {"emotion": "Fear", "confidence": 0.95, "intensity": "extreme"},
        {"emotion": "Neutral", "confidence": 0.8, "intensity": "mild"},
        {"emotion": "Fear", "confidence": 0.6, "intensity": "moderate"},
    ]
    
    for i, event_data in enumerate(test_events):
        eye_data = {
            "gaze_direction": "CENTER",
            "blink_count": 0,
            "eye_movement_velocity": 0.0,
            "facial_landmarks": {
                "mouth": {"state": "closed", "openness": 0.0}
            }
        }
        
        should_call, priority, reason = decision_engine.should_call_gemma(
            event_data, eye_data, time.time() + i * 10
        )
        
        print(f"Event {i+1}: {should_call} - {priority.value} - {reason}")
    
    print("\nSession Stats:", decision_engine.get_session_stats()) 