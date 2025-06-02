import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "date": [f"2025-04-{i}" for i in range(24, 31)] + [f"2025-05-{i}" for i in range(1, 8)],
    "sales": [
        900000, 950000, 1000000, 1050000, 1100000, 1150000, 1200000,
        1100000, 1500000, 2000000, 1750000, 2250000, 1900000, 2100000
    ]
}

response = requests.post(url, json=data)
print(response.json())