"""
Utility functions for Flight Delay Prediction API
Handles data preprocessing, artifact loading, and feature engineering
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = Path("models")
REQUIRED_ARTIFACTS = ['model', 'encoder', 'numerical_cols', 'categorical_cols', 'feature_names']

def load_artifacts(model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all saved artifacts with error handling
    
    Args:
        model_path: Optional custom path to models directory
        
    Returns:
        Dictionary containing all artifacts
        
    Raises:
        FileNotFoundError: If any artifact file is missing
        ValueError: If required artifacts are missing
    """
    artifacts = {}
    base_path = Path(model_path) if model_path else MODELS_DIR
    
    # Check if models directory exists
    if not base_path.exists():
        raise FileNotFoundError(f"Models directory not found: {base_path.absolute()}")
    
    # Define artifact files with fallback options
    artifact_files = {
        'model': ['best_model.pkl', 'flight_delay_model.pkl', 'model.pkl'],
        'encoder': ['onehot_encoder.pkl', 'encoder.pkl'],
        'numerical_cols': ['numerical_columns.pkl', 'numerical_cols.pkl'],
        'categorical_cols': ['categorical_columns.pkl', 'categorical_cols.pkl'],
        'feature_names': ['feature_names.pkl', 'features.pkl'],
        'scaler': ['scaler_transform.pkl', 'scaler.pkl']  # Optional
    }
    
    missing_artifacts = []
    
    # Try to load each artifact
    for artifact_name, possible_files in artifact_files.items():
        loaded = False
        for filename in possible_files:
            file_path = base_path / filename
            if file_path.exists():
                try:
                    artifacts[artifact_name] = joblib.load(file_path)
                    logger.info(f"✅ Loaded {artifact_name} from {filename}")
                    loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    continue
        
        if not loaded and artifact_name in REQUIRED_ARTIFACTS:
            missing_artifacts.append(artifact_name)
            artifacts[artifact_name] = None
    
    # Check for missing required artifacts
    if missing_artifacts:
        error_msg = f"Missing required artifacts: {', '.join(missing_artifacts)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Log success
    logger.info(f"✅ Successfully loaded {len(artifacts)} artifacts from {base_path}")
    
    return artifacts

def validate_artifacts(artifacts: Dict[str, Any]) -> bool:
    """
    Validate that all required artifacts are present and have correct types
    
    Args:
        artifacts: Dictionary of loaded artifacts
        
    Returns:
        True if valid, raises exception otherwise
    """
    if not artifacts:
        raise ValueError("Artifacts dictionary is empty")
    
    for key in REQUIRED_ARTIFACTS:
        if key not in artifacts:
            raise ValueError(f"Missing required artifact: {key}")
        if artifacts[key] is None:
            raise ValueError(f"Artifact '{key}' is None")
    
    # Check specific types
    if not isinstance(artifacts['numerical_cols'], list):
        raise TypeError(f"numerical_cols should be list, got {type(artifacts['numerical_cols'])}")
    
    if not isinstance(artifacts['categorical_cols'], list):
        raise TypeError(f"categorical_cols should be list, got {type(artifacts['categorical_cols'])}")
    
    if not isinstance(artifacts['feature_names'], list):
        raise TypeError(f"feature_names should be list, got {type(artifacts['feature_names'])}")
    
    logger.info("✅ Artifact validation passed")
    return True

def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create datetime features from FLIGHT_DATE column
    
    Args:
        df: DataFrame with FLIGHT_DATE column
        
    Returns:
        DataFrame with added datetime features
    """
    if 'FLIGHT_DATE' in df.columns:
        try:
            df['FLIGHT_DATE'] = pd.to_datetime(df['FLIGHT_DATE'])
            df['day_of_week'] = df['FLIGHT_DATE'].dt.dayofweek
            df['month'] = df['FLIGHT_DATE'].dt.month
            df['day_of_month'] = df['FLIGHT_DATE'].dt.day
            df['quarter'] = df['FLIGHT_DATE'].dt.quarter
            df['day_of_year'] = df['FLIGHT_DATE'].dt.dayofyear
            df['week_of_year'] = df['FLIGHT_DATE'].dt.isocalendar().week.astype(int)
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            logger.debug(f"Created datetime features for {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error creating datetime features: {e}")
    
    return df

def create_distance_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create distance bucket features from DISTANCE column
    
    Args:
        df: DataFrame with DISTANCE column
        
    Returns:
        DataFrame with added distance bucket features
    """
    if 'DISTANCE' in df.columns and 'distance_bucket' not in df.columns:
        try:
            # Define distance bins and labels
            bins = [0, 500, 1000, 1500, 2000, 3000, 5000]
            labels = ['very_short', 'short', 'medium', 'medium_long', 'long', 'very_long']
            
            df['distance_bucket'] = pd.cut(
                df['DISTANCE'], 
                bins=bins,
                labels=labels,
                right=False
            )
            
            # Also create numerical distance category
            df['distance_category'] = pd.cut(
                df['DISTANCE'],
                bins=bins,
                labels=range(len(labels)),
                right=False
            ).astype(float)
            
            logger.debug(f"Created distance buckets for {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error creating distance buckets: {e}")
    
    return df

def create_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create delay-related features
    
    Args:
        df: DataFrame with delay columns
        
    Returns:
        DataFrame with added delay features
    """
    if 'ARRIVAL_DELAY' in df.columns and 'DEPARTURE_DELAY' in df.columns:
        try:
            # Avoid division by zero
            df['delay_ratio'] = df.apply(
                lambda row: row['ARRIVAL_DELAY'] / row['DEPARTURE_DELAY'] 
                if row['DEPARTURE_DELAY'] != 0 else 0, 
                axis=1
            )
            
            # Delay difference
            df['delay_difference'] = df['ARRIVAL_DELAY'] - df['DEPARTURE_DELAY']
            
            # Binary indicators
            df['departure_delayed'] = (df['DEPARTURE_DELAY'] > 15).astype(int)
            
            logger.debug(f"Created delay features for {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error creating delay features: {e}")
    
    return df

def preprocess_input(data: Dict[str, Any], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess single input for prediction with comprehensive error handling
    
    Args:
        data: Dictionary of flight features
        artifacts: Dictionary of loaded artifacts
        
    Returns:
        Preprocessed DataFrame ready for model prediction
        
    Raises:
        ValueError: If preprocessing fails
    """
    try:
        # Validate inputs
        if not artifacts:
            raise ValueError("Artifacts not provided. Model may not be loaded correctly.")
        
        # Validate required artifacts
        for key in ['categorical_cols', 'numerical_cols', 'encoder', 'feature_names']:
            if key not in artifacts:
                raise ValueError(f"Missing '{key}' in artifacts")
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        logger.debug(f"Input data shape: {df.shape}")
        
        # Extract artifacts
        categorical_cols = artifacts['categorical_cols']
        numerical_cols = artifacts['numerical_cols']
        encoder = artifacts['encoder']
        expected_features = artifacts['feature_names']
        
        # Apply feature engineering
        df = create_datetime_features(df)
        df = create_distance_buckets(df)
        df = create_delay_features(df)
        
        # Ensure all required categorical columns exist
        for col in categorical_cols:
            if col not in df.columns:
                logger.warning(f"Missing categorical column '{col}', filling with 'UNKNOWN'")
                df[col] = 'UNKNOWN'
        
        # Ensure all required numerical columns exist
        for col in numerical_cols:
            if col not in df.columns:
                logger.warning(f"Missing numerical column '{col}', filling with 0")
                df[col] = 0
        
        # Encode categorical features
        if categorical_cols and len(categorical_cols) > 0:
            try:
                encoded_array = encoder.transform(df[categorical_cols])
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=encoder.get_feature_names_out(categorical_cols),
                    index=df.index
                )
                logger.debug(f"Encoded shape: {encoded_df.shape}")
            except Exception as e:
                logger.error(f"Error encoding categorical features: {e}")
                raise ValueError(f"Failed to encode categorical features: {e}")
        else:
            encoded_df = pd.DataFrame(index=df.index)
            logger.debug("No categorical columns to encode")
        
        # Prepare numerical features
        if numerical_cols:
            numerical_df = df[numerical_cols].copy()
            # Handle any infinity or NaN in numerical columns
            numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            numerical_df = pd.DataFrame(index=df.index)
        
        # Combine features
        if not numerical_df.empty and not encoded_df.empty:
            final_features = pd.concat([numerical_df, encoded_df], axis=1)
        elif not numerical_df.empty:
            final_features = numerical_df
        elif not encoded_df.empty:
            final_features = encoded_df
        else:
            raise ValueError("No features generated from input data")
        
        # Ensure all expected features are present
        for feat in expected_features:
            if feat not in final_features.columns:
                logger.warning(f"Missing expected feature '{feat}', filling with 0")
                final_features[feat] = 0
        
        # Reorder columns to match training
        final_features = final_features[expected_features]
        
        # Final validation
        if final_features.isnull().any().any():
            logger.warning("NaN values found in final features, filling with 0")
            final_features = final_features.fillna(0)
        
        logger.debug(f"Final features shape: {final_features.shape}")
        
        return final_features.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error in preprocess_input: {e}")
        raise ValueError(f"Failed to preprocess input: {e}")

def preprocess_batch(flights_data: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess batch of inputs with comprehensive error handling
    
    Args:
        flights_data: List of flight feature dictionaries
        artifacts: Dictionary of loaded artifacts
        
    Returns:
        Preprocessed DataFrame ready for model prediction
        
    Raises:
        ValueError: If preprocessing fails or batch is empty
    """
    try:
        # Validate inputs
        if not flights_data:
            raise ValueError("Empty batch data")
        
        if not artifacts:
            raise ValueError("Artifacts not provided")
        
        logger.debug(f"Processing batch of {len(flights_data)} flights")
        
        # Convert to DataFrame
        df = pd.DataFrame(flights_data)
        
        # Extract artifacts
        categorical_cols = artifacts['categorical_cols']
        numerical_cols = artifacts['numerical_cols']
        encoder = artifacts['encoder']
        expected_features = artifacts['feature_names']
        
        # Apply feature engineering
        df = create_datetime_features(df)
        df = create_distance_buckets(df)
        df = create_delay_features(df)
        
        # Handle missing columns
        for col in categorical_cols:
            if col not in df.columns:
                logger.warning(f"Missing categorical column '{col}' in batch, filling with 'UNKNOWN'")
                df[col] = 'UNKNOWN'
        
        for col in numerical_cols:
            if col not in df.columns:
                logger.warning(f"Missing numerical column '{col}' in batch, filling with 0")
                df[col] = 0
        
        # Encode categorical features
        if categorical_cols and len(categorical_cols) > 0:
            try:
                encoded_array = encoder.transform(df[categorical_cols])
                encoded_df = pd.DataFrame(
                    encoded_array,
                    columns=encoder.get_feature_names_out(categorical_cols),
                    index=df.index
                )
            except Exception as e:
                logger.error(f"Error encoding batch categorical features: {e}")
                raise ValueError(f"Failed to encode batch categorical features: {e}")
        else:
            encoded_df = pd.DataFrame(index=df.index)
        
        # Prepare numerical features
        if numerical_cols:
            numerical_df = df[numerical_cols].copy()
            # Handle any infinity or NaN
            numerical_df = numerical_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            numerical_df = pd.DataFrame(index=df.index)
        
        # Combine features
        if not numerical_df.empty and not encoded_df.empty:
            final_features = pd.concat([numerical_df, encoded_df], axis=1)
        elif not numerical_df.empty:
            final_features = numerical_df
        elif not encoded_df.empty:
            final_features = encoded_df
        else:
            raise ValueError("No features generated from batch data")
        
        # Ensure all expected features are present
        for feat in expected_features:
            if feat not in final_features.columns:
                logger.warning(f"Missing expected feature '{feat}' in batch, filling with 0")
                final_features[feat] = 0
        
        # Reorder columns to match training
        final_features = final_features[expected_features]
        
        # Final validation
        if final_features.isnull().any().any():
            logger.warning("NaN values found in batch features, filling with 0")
            final_features = final_features.fillna(0)
        
        logger.debug(f"Batch final features shape: {final_features.shape}")
        
        return final_features.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error in preprocess_batch: {e}")
        raise ValueError(f"Failed to preprocess batch: {e}")

def get_feature_summary(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get summary of features for debugging and monitoring
    
    Args:
        artifacts: Dictionary of loaded artifacts
        
    Returns:
        Dictionary with feature summary
    """
    summary = {
        'total_features': len(artifacts.get('feature_names', [])),
        'numerical_features': len(artifacts.get('numerical_cols', [])),
        'categorical_features': len(artifacts.get('categorical_cols', [])),
        'encoder_classes': {}
    }
    
    # Get encoder categories if available
    encoder = artifacts.get('encoder')
    if encoder and hasattr(encoder, 'categories_'):
        for i, col in enumerate(artifacts.get('categorical_cols', [])):
            if i < len(encoder.categories_):
                summary['encoder_classes'][col] = len(encoder.categories_[i])
    
    return summary

def validate_input_schema(data: Dict[str, Any], artifacts: Dict[str, Any]) -> bool:
    """
    Validate that input data has required columns
    
    Args:
        data: Input data dictionary
        artifacts: Loaded artifacts
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = set(artifacts.get('numerical_cols', []) + artifacts.get('categorical_cols', []))
    missing_cols = required_cols - set(data.keys())
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

# For testing
if __name__ == "__main__":
    # Test the utilities
    try:
        artifacts = load_artifacts()
        print(get_feature_summary(artifacts))
        print("✅ Utilities loaded successfully")
    except Exception as e:
        print(f"❌ Error loading utilities: {e}")