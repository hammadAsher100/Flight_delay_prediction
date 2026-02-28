import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def load_artifacts():
    """Load all saved artifacts"""
    artifacts = {
        'model': joblib.load('models/flight_delay_model.pkl'),
        'encoder': joblib.load('models/onehot_encoder.pkl'),
        'numerical_cols': joblib.load('models/numerical_columns.pkl'),
        'categorical_cols': joblib.load('models/categorical_columns.pkl'),
        'feature_names': joblib.load('models/feature_names.pkl')
    }
    return artifacts

def preprocess_input(data: dict, artifacts: dict):
    """Preprocess single input for prediction"""
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Extract categorical and numerical columns
    categorical_cols = artifacts['categorical_cols']
    numerical_cols = artifacts['numerical_cols']
    encoder = artifacts['encoder']
    
    # Handle datetime features if present
    if 'FLIGHT_DATE' in df.columns:
        df['FLIGHT_DATE'] = pd.to_datetime(df['FLIGHT_DATE'])
        df['day_of_week'] = df['FLIGHT_DATE'].dt.dayofweek
        df['month'] = df['FLIGHT_DATE'].dt.month
        df['day_of_month'] = df['FLIGHT_DATE'].dt.day
    
    # Create distance buckets if needed
    if 'DISTANCE' in df.columns and 'distance_bucket' not in df.columns:
        df['distance_bucket'] = pd.cut(df['DISTANCE'], 
                                       bins=[0, 500, 1000, 1500, 2000, 5000],
                                       labels=['short', 'medium', 'medium_long', 'long', 'very_long'])
    
    # Ensure all required categorical columns exist
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = 'UNKNOWN'
    
    # Encode categorical features
    if len(categorical_cols) > 0:
        encoded_array = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
    else:
        encoded_df = pd.DataFrame()
    
    # Prepare numerical features
    numerical_df = df[numerical_cols] if numerical_cols else pd.DataFrame()
    
    # Combine features
    if not numerical_df.empty and not encoded_df.empty:
        final_features = pd.concat([numerical_df, encoded_df], axis=1)
    elif not numerical_df.empty:
        final_features = numerical_df
    else:
        final_features = encoded_df
    
    # Ensure all expected features are present
    expected_features = artifacts['feature_names']
    for feat in expected_features:
        if feat not in final_features.columns:
            final_features[feat] = 0
    
    # Reorder columns to match training
    final_features = final_features[expected_features]
    
    return final_features

def preprocess_batch(flights_data: list, artifacts: dict):
    """Preprocess batch of inputs"""
    
    # Convert to DataFrame
    df = pd.DataFrame(flights_data)
    
    # Apply same preprocessing as single input
    categorical_cols = artifacts['categorical_cols']
    numerical_cols = artifacts['numerical_cols']
    encoder = artifacts['encoder']
    
    # Create derived features if needed
    if 'DISTANCE' in df.columns:
        df['distance_bucket'] = pd.cut(df['DISTANCE'], 
                                       bins=[0, 500, 1000, 1500, 2000, 5000],
                                       labels=['short', 'medium', 'medium_long', 'long', 'very_long'])
    
    # Encode categorical features
    if len(categorical_cols) > 0:
        encoded_array = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
    else:
        encoded_df = pd.DataFrame()
    
    # Prepare numerical features
    numerical_df = df[numerical_cols] if numerical_cols else pd.DataFrame()
    
    # Combine features
    if not numerical_df.empty and not encoded_df.empty:
        final_features = pd.concat([numerical_df, encoded_df], axis=1)
    elif not numerical_df.empty:
        final_features = numerical_df
    else:
        final_features = encoded_df
    
    # Ensure all expected features are present
    expected_features = artifacts['feature_names']
    for feat in expected_features:
        if feat not in final_features.columns:
            final_features[feat] = 0
    
    # Reorder columns to match training
    final_features = final_features[expected_features]
    
    return final_features