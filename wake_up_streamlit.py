import time
import requests

def ping_app():
    app_url = "https://your-streamlit-app-url"  # Replace with your app's URL
    while True:
        try:
            response = requests.get(app_url)
            if response.status_code == 200:
                print(f"Successfully pinged {app_url}")
            else:
                print(f"Failed to ping {app_url}, status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error pinging {app_url}: {e}")
        time.sleep(21600)  # Wait for 6 hours before the next ping

if __name__ == "__main__":
    ping_app()
