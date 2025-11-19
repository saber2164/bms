import requests
import json

url = 'http://127.0.0.1:5000/api/predict'
features = [[0.95, 25.0, 0.05, 0.02, 10.0, 3.3] for _ in range(15)]
data = {'features': features}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
