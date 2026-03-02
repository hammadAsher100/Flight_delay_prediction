import joblib
import numpy as np
from .utils import load_artifacts, preprocess_input, preprocess_batch

class FlightDelayModel:
    def __init__(self):
        self.artifacts = None
        self.model = None
        self.loaded = False
    
    def load(self):
        """Load model and artifacts"""
        try:
            self.artifacts = load_artifacts()
            self.model = self.artifacts['model']
            self.loaded = True
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.loaded = False
    
    def predict(self, features: dict):
        """Predict for single flight"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Preprocess
        X = preprocess_input(features, self.artifacts)
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            # Classification model
            proba = self.model.predict_proba(X)[0]
            delay_probability = float(proba[1])  # Probability of delayed
            
            # Binary prediction
            prediction = "Delayed" if delay_probability > 0.5 else "Not Delayed"
            
            # For regression (if you have a separate regressor)
            delay_minutes = None
            
        else:
            # Regression model
            delay_minutes = float(self.model.predict(X)[0])
            delay_probability = 1.0 if delay_minutes > 15 else 0.0
            prediction = "Delayed" if delay_minutes > 15 else "Not Delayed"
        
        return {
            'delay_probability': delay_probability,
            'prediction': prediction,
            'delay_minutes': delay_minutes
        }
    
    def predict_batch(self, flights_data: list):
        """Predict for multiple flights"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Preprocess batch
        X = preprocess_batch(flights_data, self.artifacts)
        
        # Predict
        if hasattr(self.model, 'predict_proba'):
            # Classification model
            probas = self.model.predict_proba(X)
            delay_probabilities = [float(p[1]) for p in probas]
            predictions = ["Delayed" if p > 0.5 else "Not Delayed" for p in delay_probabilities]
            delay_minutes = [None] * len(predictions)
        else:
            # Regression model
            delay_minutes = self.model.predict(X).tolist()
            delay_probabilities = [1.0 if d > 15 else 0.0 for d in delay_minutes]
            predictions = ["Delayed" if d > 15 else "Not Delayed" for d in delay_minutes]
        
        # Calculate statistics
        delayed_count = sum(1 for p in predictions if p == "Delayed")
        total = len(predictions)
        delayed_percentage = (delayed_count / total) * 100 if total > 0 else 0
        
        results = []
        for i in range(total):
            results.append({
                'flight_id': i,
                'delay_probability': delay_probabilities[i],
                'prediction': predictions[i],
                'delay_minutes': delay_minutes[i] if delay_minutes[i] is not None else None
            })
        
        return {
            'predictions': results,
            'total_flights': total,
            'delayed_count': delayed_count,
            'delayed_percentage': delayed_percentage
        }

# Singleton instance
model = FlightDelayModel()