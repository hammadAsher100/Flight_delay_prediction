import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Test health
def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:", response.json())

# Test single prediction
def test_predict():
    url = f"{BASE_URL}/predict"
    
    # Sample flight data
    flight_data = {
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
    
    response = requests.post(url, json=flight_data)
    print("Single prediction:", json.dumps(response.json(), indent=2))

# Test batch prediction
def test_batch():
    url = f"{BASE_URL}/predict/batch"
    
    # Sample batch data
    batch_data = {
        "flights": [
            {
                "AIRLINE": "AA",
                "ORIGIN_AIRPORT": "JFK",
                "DESTINATION_AIRPORT": "LAX",
                "DISTANCE": 2475,
                "DEPARTURE_DELAY": 5,
                "DAY_OF_WEEK": 1,
                "MONTH": 6
            },
            {
                "AIRLINE": "UA",
                "ORIGIN_AIRPORT": "SFO",
                "DESTINATION_AIRPORT": "ORD",
                "DISTANCE": 1846,
                "DEPARTURE_DELAY": 20,
                "DAY_OF_WEEK": 3,
                "MONTH": 6
            }
        ]
    }
    
    response = requests.post(url, json=batch_data)
    print("Batch prediction:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing Flight Delay Prediction API...")
    test_health()
    print("\n" + "="*50)
    test_predict()
    print("\n" + "="*50)
    test_batch()