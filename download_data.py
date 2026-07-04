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
    urls = {
        "Unemployment in India.csv": "https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno2/main/Unemployment%20in%20India.csv",
        "Unemployment_Rate_upto_11_2020.csv": "https://raw.githubusercontent.com/proteendas/OIBSIP-DS-03/main/Unemployment_Rate_upto_11_2020.csv"
    }
    
    os.makedirs("unemployment_data", exist_ok=True)
    
    success = True
    for filename, url in urls.items():
        dest = os.path.join("unemployment_data", filename)
        # Only download if not already exists or if it failed previously
        if not os.path.exists(dest) or os.path.getsize(dest) < 100:
            if not download_file(url, dest):
                success = False
        else:
            print(f"{filename} already exists and is not empty. Skipping.")
                
    if success:
        print("\nAll datasets downloaded successfully and saved under 'unemployment_data/'!")
    else:
        print("\nSome downloads failed. Please verify URLs.")

if __name__ == "__main__":
    main()
