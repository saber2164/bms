import requests
import json
import time

def test_soc_api():
    base_url = "http://localhost:8080"
    
    print("1. Testing Initialization...")
    init_url = f"{base_url}/api/init_filter"
    init_payload = {
        "capacity_ah": 2.0,
        "initial_soc": 0.9,
        "initial_r0": 0.01,
        "dt": 1.0
    }
    
    try:
        response = requests.post(init_url, json=init_payload)
        if response.status_code == 200:
            print("Initialization Success!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Initialization Failed: {response.status_code}")
            print(response.text)
            return
    except requests.exceptions.ConnectionError:
        print("Connection Error: Is the server running on port 8080?")
        return

    print("\n2. Testing Prediction (Sequence)...")
    predict_url = f"{base_url}/api/predict_soc"
    
    # Sample data: [Voltage, Current, Temperature]
    payload = {
        "features": [
            [4.1, 1.0, 25.0],
            [4.09, 1.0, 25.1],
            [4.08, 1.0, 25.2]
        ]
    }
    
    response = requests.post(predict_url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("Prediction Success!")
        print(json.dumps(data, indent=2))
        
        results = data.get('results', [])
        if len(results) == 3:
            print("\nTEST PASSED: Received 3 SoC estimates.")
        else:
            print(f"\nTEST FAILED: Expected 3 results, got {len(results)}.")
    else:
        print(f"\nTEST FAILED: Status Code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Wait a bit for server to start if running in parallel (manual run)
    time.sleep(2) 
    test_soc_api()
