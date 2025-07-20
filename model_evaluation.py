#!/usr/bin/env python3
"""
Silent Voice Model Evaluation Suite
Benchmarks fine-tuned Gemma 3n against base model for medical communication
"""

import json
import time
import numpy as np
from datetime import datetime
import ollama
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self):
        self.base_model = "hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M"  # Base Gemma 3n was unsloth/gemma-3n-E4B-it but we need Ollama compatible version
        self.finetuned_model = "hf.co/0xroyce/silent-voice-multimodal"
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self):
        """Load biosignal test cases with expected outputs"""
        return [
            {
                "biosignal": "Fear expression with confidence 0.95 and heart rate 95 bpm (stress: elevated)",
                "expected_category": "urgent_medical",
                "expected_sentiment": "distress",
                "scenario": "ICU patient experiencing discomfort"
            },
            {
                "biosignal": "Happy expression with confidence 0.87 and sustained gaze upward",
                "expected_category": "positive_communication", 
                "expected_sentiment": "positive",
                "scenario": "Patient expressing gratitude"
            },
            {
                "biosignal": "Concentration with confidence 0.75 and rapid blinking for 3s",
                "expected_category": "communication_attempt",
                "expected_sentiment": "neutral",
                "scenario": "Patient trying to communicate specific need"
            },
            {
                "biosignal": "Pain expression with confidence 0.92 and jaw clenched",
                "expected_category": "urgent_medical",
                "expected_sentiment": "pain",
                "scenario": "Patient in significant discomfort"
            },
            {
                "biosignal": "Neutral expression with confidence 0.65 and slow eye movements",
                "expected_category": "routine_monitoring",
                "expected_sentiment": "neutral", 
                "scenario": "Baseline patient state"
            }
        ]
    
    def evaluate_model(self, model_name, test_cases):
        """Evaluate a model on test cases"""
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"Testing case {i+1}/{len(test_cases)}: {case['scenario']}")
            
            # Generate response
            start_time = time.time()
            response = self._get_model_response(model_name, case['biosignal'])
            response_time = time.time() - start_time
            
            # Analyze response quality
            analysis = self._analyze_response(response, case)
            
            results.append({
                'case_id': i,
                'biosignal': case['biosignal'],
                'expected': case,
                'response': response,
                'response_time': response_time,
                'analysis': analysis
            })
            
        return results
    
    def _get_model_response(self, model_name, biosignal):
        """Get response from specified model"""
        try:
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a person communicating through biosignals. Respond in first person with what you want to communicate.'
                },
                {
                    'role': 'user', 
                    'content': f'Biosignal: {biosignal}'
                }
            ]
            
            stream = ollama.chat(model=model_name, messages=messages, stream=True)
            response = ''
            for chunk in stream:
                response += chunk['message']['content']
                
            return response.strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _analyze_response(self, response, expected_case):
        """Analyze response quality against expected output"""
        analysis = {
            'relevance_score': 0,
            'medical_appropriateness': 0,
            'first_person_score': 0,
            'urgency_match': 0,
            'overall_score': 0
        }
        
        response_lower = response.lower()
        
        # Check relevance to scenario
        scenario_keywords = {
            'urgent_medical': ['help', 'pain', 'hurt', 'uncomfortable', 'wrong', 'problem'],
            'positive_communication': ['thank', 'good', 'happy', 'better', 'yes', 'great'],
            'communication_attempt': ['need', 'want', 'tell', 'say', 'important'],
            'routine_monitoring': ['okay', 'fine', 'nothing', 'normal', 'comfortable']
        }
        
        expected_keywords = scenario_keywords.get(expected_case['expected_category'], [])
        relevance_matches = sum(1 for keyword in expected_keywords if keyword in response_lower)
        analysis['relevance_score'] = min(1.0, relevance_matches / max(1, len(expected_keywords)))
        
        # Check medical appropriateness (avoiding medical jargon)
        medical_jargon = ['diagnosis', 'symptoms', 'treatment', 'medication', 'disease']
        jargon_count = sum(1 for jargon in medical_jargon if jargon in response_lower)
        analysis['medical_appropriateness'] = max(0, 1 - (jargon_count * 0.3))
        
        # Check first person usage
        first_person_indicators = ['i ', 'me ', 'my ', "i'm", "i'll", "i've"]
        first_person_count = sum(1 for indicator in first_person_indicators if indicator in response_lower)
        analysis['first_person_score'] = min(1.0, first_person_count / 2)
        
        # Check urgency matching
        if expected_case['expected_category'] == 'urgent_medical':
            urgency_words = ['urgent', 'immediately', 'now', 'quickly', 'help']
            urgency_matches = sum(1 for word in urgency_words if word in response_lower)
            analysis['urgency_match'] = min(1.0, urgency_matches / 2)
        else:
            # Non-urgent should not have urgency words
            urgency_words = ['urgent', 'emergency', 'immediately', 'quickly']
            urgency_count = sum(1 for word in urgency_words if word in response_lower)
            analysis['urgency_match'] = max(0, 1 - (urgency_count * 0.5))
        
        # Calculate overall score
        analysis['overall_score'] = np.mean([
            analysis['relevance_score'],
            analysis['medical_appropriateness'], 
            analysis['first_person_score'],
            analysis['urgency_match']
        ])
        
        return analysis
    
    def compare_models(self):
        """Compare base vs fine-tuned models"""
        print("🧪 Evaluating Silent Voice Model Performance")
        print("=" * 60)
        
        # Test base model
        print("\n📊 Testing Base Gemma 3b Model...")
        base_results = self.evaluate_model(self.base_model, self.test_cases)
        
        # Test fine-tuned model
        print("\n🔥 Testing Fine-tuned Silent Voice Model...")
        finetuned_results = self.evaluate_model(self.finetuned_model, self.test_cases)
        
        # Compare results
        self._generate_comparison_report(base_results, finetuned_results)
        
        return base_results, finetuned_results
    
    def _generate_comparison_report(self, base_results, finetuned_results):
        """Generate detailed comparison report"""
        print("\n" + "="*80)
        print("📈 MODEL COMPARISON REPORT")
        print("="*80)
        
        # Calculate average scores
        base_scores = [r['analysis']['overall_score'] for r in base_results]
        finetuned_scores = [r['analysis']['overall_score'] for r in finetuned_results]
        
        base_avg = np.mean(base_scores)
        finetuned_avg = np.mean(finetuned_scores)
        improvement = ((finetuned_avg - base_avg) / base_avg) * 100
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Base Gemma 3n:       {base_avg:.3f}")
        print(f"   Fine-tuned Model:    {finetuned_avg:.3f}")
        print(f"   Improvement:         {improvement:+.1f}%")
        
        # Detailed metrics
        metrics = ['relevance_score', 'medical_appropriateness', 'first_person_score', 'urgency_match']
        
        print(f"\n📊 DETAILED METRICS:")
        for metric in metrics:
            base_metric = np.mean([r['analysis'][metric] for r in base_results])
            finetuned_metric = np.mean([r['analysis'][metric] for r in finetuned_results])
            metric_improvement = ((finetuned_metric - base_metric) / max(base_metric, 0.001)) * 100
            
            print(f"   {metric.replace('_', ' ').title():20} | Base: {base_metric:.3f} | Fine-tuned: {finetuned_metric:.3f} | {metric_improvement:+.1f}%")
        
        # Response time comparison
        base_times = [r['response_time'] for r in base_results]
        finetuned_times = [r['response_time'] for r in finetuned_results]
        
        print(f"\n⚡ RESPONSE TIMES:")
        print(f"   Base Model:          {np.mean(base_times):.2f}s (±{np.std(base_times):.2f})")
        print(f"   Fine-tuned Model:    {np.mean(finetuned_times):.2f}s (±{np.std(finetuned_times):.2f})")
        
        # Case-by-case analysis
        print(f"\n🔍 CASE-BY-CASE ANALYSIS:")
        for i, (base, finetuned) in enumerate(zip(base_results, finetuned_results)):
            scenario = self.test_cases[i]['scenario']
            base_score = base['analysis']['overall_score']
            finetuned_score = finetuned['analysis']['overall_score']
            case_improvement = finetuned_score - base_score
            
            print(f"   Case {i+1}: {scenario}")
            print(f"           Base: {base_score:.3f} | Fine-tuned: {finetuned_score:.3f} | Δ: {case_improvement:+.3f}")
            print(f"           Base Response: \"{base['response'][:50]}...\"")
            print(f"           Fine-tuned:    \"{finetuned['response'][:50]}...\"")
            print()
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"model_evaluation_{timestamp}.json"
        
        evaluation_data = {
            'timestamp': timestamp,
            'base_model': self.base_model,
            'finetuned_model': self.finetuned_model,
            'test_cases': self.test_cases,
            'base_results': base_results,
            'finetuned_results': finetuned_results,
            'summary': {
                'base_avg_score': base_avg,
                'finetuned_avg_score': finetuned_avg,
                'improvement_percent': improvement,
                'base_avg_time': np.mean(base_times),
                'finetuned_avg_time': np.mean(finetuned_times)
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
        
        print(f"💾 Detailed results saved to: {results_file}")

def main():
    evaluator = ModelEvaluator()
    base_results, finetuned_results = evaluator.compare_models()
    
    print("\n✅ Model evaluation completed!")
    print("📋 Use this data to demonstrate your fine-tuning effectiveness")

if __name__ == "__main__":
    main() 