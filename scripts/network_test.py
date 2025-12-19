import requests
import time
import sys

def check_connectivity():
    url = "http://localhost:8001/metrics"
    print(f"Checking connectivity to {url}...")
    for i in range(30):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"SUCCESS: Connected! Status: {response.status_code}")
                # Also try the home page
                home = requests.get("http://localhost:8001/", timeout=2)
                print(f"Home Page Status: {home.status_code}")
                return True
        except Exception as e:
            print(f"Attempt {i+1}: Failed ({e})")
            time.sleep(2)
            
    print("FAILURE: Could not connect after 60 seconds.")
    return False

if __name__ == "__main__":
    if check_connectivity():
        sys.exit(0)
    else:
        sys.exit(1)


