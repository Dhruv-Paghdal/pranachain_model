import requests
from bs4 import BeautifulSoup
import os
import zipfile
from urllib.parse import urljoin

# Base URL for the NHANES website. It's crucial for building absolute URLs.
BASE_URL = "https://wwwn.cdc.gov"
LAB_DATA_URL = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory"

# Define the directory where files will be saved and the name of the zip file
DOWNLOAD_DIR = "nhanes_data"
ZIP_FILE_NAME = "nhanes_lab_data.zip"

def main():
    """
    Main function to orchestrate the downloading and zipping of .xpt files.
    """
    # Create the download directory if it doesn't exist
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Creating directory: {DOWNLOAD_DIR}")
        os.makedirs(DOWNLOAD_DIR)

    # Fetch the main page content
    print(f"Fetching data from: {LAB_DATA_URL}")
    try:
        response = requests.get(LAB_DATA_URL, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links that end with '.xpt'
    xpt_links = soup.find_all('a', href=lambda href: href and href.endswith('.xpt'))

    if not xpt_links:
        print("No .xpt files found on the page.")
        return

    print(f"Found {len(xpt_links)} .xpt files to download.")
    
    # List to keep track of successfully downloaded files for zipping
    downloaded_files = []

    for link in xpt_links:
        relative_url = link['href']
        # Construct the absolute URL
        absolute_url = urljoin(BASE_URL, relative_url)
        
        # Get the filename from the URL
        file_name = os.path.basename(absolute_url)
        save_path = os.path.join(DOWNLOAD_DIR, file_name)

        print(f"Downloading {file_name}...")

        try:
            # Use stream=True for large files
            file_response = requests.get(absolute_url, stream=True, timeout=10)
            file_response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded {file_name}")
            downloaded_files.append(save_path)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_name}: {e}")
            continue

    # Create a zip archive of the downloaded files
    if downloaded_files:
        print(f"\nCreating zip archive: {ZIP_FILE_NAME}")
        try:
            with zipfile.ZipFile(ZIP_FILE_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in downloaded_files:
                    # The second argument to write() is the name of the file inside the zip.
                    # We only want the filename, not the full path.
                    zipf.write(file_path, arcname=os.path.basename(file_path))
            print(f"Successfully created {ZIP_FILE_NAME}")
        except Exception as e:
            print(f"Error creating zip file: {e}")
    else:
        print("No files were downloaded, so no zip file was created.")

if __name__ == "__main__":
    main()
