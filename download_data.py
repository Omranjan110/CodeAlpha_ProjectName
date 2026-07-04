import os
import requests

def download_file(url, output_path):
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {output_path} successfully!")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    # 1. Download Unemployment Data (Task 2)
    urls_unemployment = {
        "Unemployment in India.csv": "https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno2/main/Unemployment%20in%20India.csv",
        "Unemployment_Rate_upto_11_2020.csv": "https://raw.githubusercontent.com/proteendas/OIBSIP-DS-03/main/Unemployment_Rate_upto_11_2020.csv"
    }
    
    os.makedirs("unemployment_data", exist_ok=True)
    
    print("--- Checking/Downloading Unemployment Datasets ---")
    for filename, url in urls_unemployment.items():
        dest = os.path.join("unemployment_data", filename)
        if not os.path.exists(dest) or os.path.getsize(dest) < 100:
            download_file(url, dest)
        else:
            print(f"{filename} already exists. Skipping.")

    # 2. Download Car Price Data (Task 3)
    urls_car_price = {
        "CarPrice_Assignment.csv": "https://raw.githubusercontent.com/Lakshya-Ag/Prediction-of-Car-prices/master/CarPrice_Assignment.csv"
    }
    
    os.makedirs("car_price_data", exist_ok=True)
    
    print("\n--- Checking/Downloading Car Price Datasets ---")
    for filename, url in urls_car_price.items():
        dest = os.path.join("car_price_data", filename)
        if not os.path.exists(dest) or os.path.getsize(dest) < 100:
            if not download_file(url, dest):
                # Fallback URL if first one fails
                fallback_url = "https://raw.githubusercontent.com/Sinchana-K/Car-Price-Prediction-with-Machine-Learning/main/CarPrice_Assignment.csv"
                print("Trying fallback URL...")
                download_file(fallback_url, dest)
        else:
            print(f"{filename} already exists. Skipping.")

if __name__ == "__main__":
    main()
