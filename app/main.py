from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from typing import List, Dict, Any

from .schemas import (
    FlightFeatures, BatchFeatures, 
    PredictionResponse, BatchPredictionResponse,
    HealthResponse
)
from .model import model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Flight Delay Prediction API",
    description="API for predicting flight delays using ML model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up... loading model")
    model.load()
    logger.info("Startup complete")

# Health check
@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model.loaded else "degraded",
        model_loaded=model.loaded,
        model_type=type(model.model).__name__ if model.model else "unknown"
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(features: FlightFeatures, background_tasks: BackgroundTasks):
    """
    Predict delay for a single flight
    """
    start_time = time.time()
    
    try:
        # Log request
        background_tasks.add_task(
            logger.info, 
            f"Prediction request: {features.dict()}"
        )
        
        # Make prediction
        result = model.predict(features.dict())
        
        # Log latency
        latency = time.time() - start_time
        background_tasks.add_task(
            logger.info, 
            f"Prediction completed in {latency:.3f}s"
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchFeatures, background_tasks: BackgroundTasks):
    """
    Predict delays for multiple flights
    """
    start_time = time.time()
    
    try:
        flights_data: List[Dict[str, Any]] = [f.dict() for f in batch.flights]
        
        # Log request
        background_tasks.add_task(
            logger.info, 
            f"Batch prediction request: {len(flights_data)} flights"
        )
        
        # Make predictions
        results = model.predict_batch(flights_data)
        
        # Log latency
        latency = time.time() - start_time
        background_tasks.add_task(
            logger.info, 
            f"Batch prediction completed in {latency:.3f}s for {len(flights_data)} flights"
        )
        
        return BatchPredictionResponse(**results)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model.loaded or model.artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model.model).__name__,
        "features": list(model.artifacts['feature_names']),
        "num_features": len(model.artifacts['feature_names']),
        "categorical_features": model.artifacts['categorical_cols'],
        "numerical_features": model.artifacts['numerical_cols']
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Flight Delay Prediction API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }