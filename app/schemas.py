from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import time

class FlightFeatures(BaseModel):
    # Add all your features here based on your model
    AIRLINE: str = Field(..., description="Airline code")
    ORIGIN_AIRPORT: str = Field(..., description="Origin airport code")
    DESTINATION_AIRPORT: str = Field(..., description="Destination airport code")
    DISTANCE: float = Field(..., description="Flight distance in miles")
    DEPARTURE_DELAY: Optional[float] = Field(0, description="Departure delay in minutes")
    SCHEDULED_DEPARTURE: Optional[int] = Field(None, description="Scheduled departure time")
    SCHEDULED_ARRIVAL: Optional[int] = Field(None, description="Scheduled arrival time")
    DAY_OF_WEEK: Optional[int] = Field(None, description="Day of week (0-6)")
    MONTH: Optional[int] = Field(None, description="Month (1-12)")
    
    class Config:
        schema_extra = {
            "example": {
                "AIRLINE": "AA",
                "ORIGIN_AIRPORT": "JFK",
                "DESTINATION_AIRPORT": "LAX",
                "DISTANCE": 2475,
                "DEPARTURE_DELAY": 5,
                "SCHEDULED_DEPARTURE": 800,
                "SCHEDULED_ARRIVAL": 1100,
                "DAY_OF_WEEK": 1,
                "MONTH": 6
            }
        }

class BatchFeatures(BaseModel):
    flights: List[FlightFeatures]

class PredictionResponse(BaseModel):
    flight_id: Optional[int] = None
    delay_probability: float
    prediction: str  # "Delayed" or "Not Delayed"
    delay_minutes: Optional[float] = None  # For regression task

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_flights: int
    delayed_count: int
    delayed_percentage: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    version: str = "1.0.0"