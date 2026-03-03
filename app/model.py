import joblib
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, cast
from .utils import load_artifacts, preprocess_input, preprocess_batch

# Setup logging
logger = logging.getLogger(__name__)

# Define types for better IDE support
from sklearn.base import BaseEstimator
import pandas as pd

class FlightDelayModel:
    def __init__(self):
        self.artifacts: Optional[Dict[str, Any]] = None
        self.model: Optional[BaseEstimator] = None
        self.loaded: bool = False
        self.model_type: Optional[str] = None  # 'classification' or 'regression'
    
    def load(self) -> None:
        """Load model and artifacts"""
        try:
            artifacts = load_artifacts()
            if not isinstance(artifacts, dict):
                raise TypeError("load_artifacts must return a dictionary")
            
            self.artifacts = artifacts
            self.model = artifacts.get('model')
            
            if self.model is None:
                raise KeyError("'model' not found in artifacts")
            
            # Determine model type
            if hasattr(self.model, 'predict_proba'):
                self.model_type = 'classification'
                # Check if model has been fitted with enough classes
                if hasattr(self.model, 'classes_'):
                    model_with_classes = cast(Any, self.model)
                    if len(model_with_classes.classes_) < 2:
                        logger.warning("Model only has one class! Retraining may be needed.")
            else:
                self.model_type = 'regression'
            
            self.loaded = True
            logger.info(f"✅ Model loaded successfully (type: {self.model_type})")
            print(f"✅ Model loaded successfully (type: {self.model_type})")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Model file not found: {e}")
            print(f"❌ Model file not found. Please ensure all model files are in the models/ directory.")
            self.loaded = False
        except KeyError as e:
            logger.error(f"❌ Missing key in artifacts: {e}")
            print(f"❌ Model artifacts are incomplete. Missing: {e}")
            self.loaded = False
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            print(f"❌ Error loading model: {e}")
            self.loaded = False
    
    def _check_loaded(self) -> None:
        """Check if model is loaded, raise appropriate error if not"""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.model is None:
            raise RuntimeError("Model is None. Load failed silently.")
        if self.artifacts is None:
            raise RuntimeError("Artifacts not loaded. Load failed silently.")
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict for single flight
        
        Args:
            features: Dictionary of flight features
            
        Returns:
            Dictionary with prediction results
        """
        self._check_loaded()
        
        # Type assertions for linter
        assert self.model is not None
        assert self.artifacts is not None
        assert self.model_type is not None
        
        try:
            # Preprocess
            X = preprocess_input(features, self.artifacts)
            
            if self.model_type == 'classification':
                # Classification model
                if hasattr(self.model, 'predict_proba'):
                    # Cast to Any to avoid linter errors
                    model_with_proba = cast(Any, self.model)
                    proba = model_with_proba.predict_proba(X)[0]
                    
                    # Handle case where model might only have one class
                    if len(proba) >= 2:
                        delay_probability = float(proba[1])
                    else:
                        delay_probability = float(proba[0])
                        logger.warning("Model only has one class!")
                else:
                    # Fallback for classifiers without predict_proba
                    model_with_predict = cast(Any, self.model)
                    pred = model_with_predict.predict(X)[0]
                    delay_probability = 1.0 if pred == 1 else 0.0
                
                # Binary prediction
                prediction = "Delayed" if delay_probability > 0.5 else "Not Delayed"
                delay_minutes = None
                
            else:
                # Regression model
                model_with_predict = cast(Any, self.model)
                delay_minutes = float(model_with_predict.predict(X)[0])
                delay_probability = 1.0 if delay_minutes > 15 else 0.0
                prediction = "Delayed" if delay_minutes > 15 else "Not Delayed"
            
            return {
                'delay_probability': round(delay_probability, 4),
                'prediction': prediction,
                'delay_minutes': round(delay_minutes, 2) if delay_minutes is not None else None
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Failed to make prediction: {e}")
    
    def predict_batch(self, flights_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict for multiple flights
        
        Args:
            flights_data: List of flight feature dictionaries
            
        Returns:
            Dictionary with batch prediction results and statistics
        """
        self._check_loaded()
        
        # Type assertions for linter
        assert self.model is not None
        assert self.artifacts is not None
        assert self.model_type is not None
        
        if not flights_data:
            return {
                'predictions': [],
                'total_flights': 0,
                'delayed_count': 0,
                'delayed_percentage': 0.0
            }
        
        try:
            # Preprocess batch
            X = preprocess_batch(flights_data, self.artifacts)
            
            if self.model_type == 'classification':
                # Classification model
                if hasattr(self.model, 'predict_proba'):
                    # Cast to Any to avoid linter errors
                    model_with_proba = cast(Any, self.model)
                    probas = model_with_proba.predict_proba(X)
                    
                    # Handle different shapes of probas
                    if len(probas.shape) == 2 and probas.shape[1] >= 2:
                        delay_probabilities = [float(p[1]) for p in probas]
                    else:
                        delay_probabilities = [float(p[0]) for p in probas]
                        logger.warning("Model only has one class!")
                else:
                    # Fallback
                    model_with_predict = cast(Any, self.model)
                    preds = model_with_predict.predict(X)
                    delay_probabilities = [1.0 if p == 1 else 0.0 for p in preds]
                
                predictions = ["Delayed" if p > 0.5 else "Not Delayed" for p in delay_probabilities]
                delay_minutes = [None] * len(predictions)
                
            else:
                # Regression model
                model_with_predict = cast(Any, self.model)
                delay_minutes = model_with_predict.predict(X).tolist()
                delay_probabilities = [1.0 if d > 15 else 0.0 for d in delay_minutes]
                predictions = ["Delayed" if d > 15 else "Not Delayed" for d in delay_minutes]
            
            # Calculate statistics
            delayed_count = sum(1 for p in predictions if p == "Delayed")
            total = len(predictions)
            delayed_percentage = (delayed_count / total) * 100 if total > 0 else 0
            
            # Create results list
            results = []
            for i in range(total):
                # You can add actual flight_id if it exists in the input data
                flight_id = flights_data[i].get('flight_id', i) if i < len(flights_data) else i
                
                delay_min_value = delay_minutes[i]
                results.append({
                    'flight_id': flight_id,
                    'delay_probability': round(delay_probabilities[i], 4),
                    'prediction': predictions[i],
                    'delay_minutes': round(float(delay_min_value), 2) if delay_min_value is not None else None
                })
            
            return {
                'predictions': results,
                'total_flights': total,
                'delayed_count': delayed_count,
                'delayed_percentage': round(delayed_percentage, 2),
                'model_type': self.model_type
            }
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise RuntimeError(f"Failed to make batch predictions: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        self._check_loaded()
        
        # Type assertions for linter
        assert self.artifacts is not None
        assert self.model is not None
        
        return {
            'loaded': self.loaded,
            'model_type': self.model_type,
            'model_class': self.model.__class__.__name__,
            'features': list(self.artifacts.get('feature_names', [])),
            'num_features': len(self.artifacts.get('feature_names', [])),
            'categorical_features': self.artifacts.get('categorical_cols', []),
            'numerical_features': self.artifacts.get('numerical_cols', [])
        }

# Singleton instance
model = FlightDelayModel()