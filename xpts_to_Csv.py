import pandas as pd
import os

# Define the source and destination directories.
# In Google Colab, this would be a path to your mounted Google Drive
# or the directory where you've uploaded your files.
SOURCE_DIR = 'nhanes_data'
DESTINATION_DIR = 'nhanes_csv_data'

def convert_all_xpt_to_csv(source_dir, dest_dir):
    """
    Converts all .xpt files in a source directory to .csv files
    in a destination directory.

    Args:
        source_dir (str): The path to the directory containing .xpt files.
        dest_dir (str): The path to the directory where .csv files will be saved.
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        print(f"Creating directory: {dest_dir}")
        os.makedirs(dest_dir)
    
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory not found: {source_dir}. Please ensure your .xpt files are in this folder.")
        return

    # List all files in the source directory
    all_files = os.listdir(source_dir)
    
    xpt_files = [f for f in all_files if f.endswith('.xpt')]
    
    if not xpt_files:
        print(f"No .xpt files found in '{source_dir}'.")
        return
        
    print(f"Found {len(xpt_files)} .xpt files. Starting conversion...")

    for file_name in xpt_files:
        source_path = os.path.join(source_dir, file_name)
        csv_file_name = file_name.replace('.xpt', '.csv')
        dest_path = os.path.join(dest_dir, csv_file_name)
        
        try:
            # Read the XPT file
            df = pd.read_sas(source_path)
            
            # Write the DataFrame to a CSV file
            df.to_csv(dest_path, index=False)
            print(f"Successfully converted '{file_name}' to '{csv_file_name}'.")
        except Exception as e:
            print(f"Error converting '{file_name}': {e}")
            continue

    print("\nConversion complete.")

if __name__ == "__main__":
    convert_all_xpt_to_csv(SOURCE_DIR, DESTINATION_DIR)
