import requests

# Test different possible endpoints
base_url = "https://v6.db.transport.rest"
test_endpoints = [
    "/stops?query=Dortmund",
    "/locations?query=Dortmund", 
    "/stations?query=Dortmund",
    "/places?query=Dortmund"
]

for endpoint in test_endpoints:
    url = f"{base_url}{endpoint}"
    try:
        print(f"\n🔍 Testing: {url}")
        response = requests.get(url, timeout=5)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS! Found: {data[:2]}")
            break
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
