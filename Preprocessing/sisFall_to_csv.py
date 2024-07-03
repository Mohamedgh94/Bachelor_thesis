import os
import pandas as pd

def add_headers_and_save(input_file, output_file, headers):
    try:
        df = pd.read_csv(input_file, header=None, encoding='utf-8')
        df.columns = headers[:df.shape[1]]  # Only use as many headers as there are columns
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Failed to process {input_file}. Error: {e}")

def process_directory(input_dir, output_dir, headers):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for name in files:
            if name == 'Readme.txt' or not name.endswith('.txt'):
                continue
            
            # Determine the full input file path
            input_file_path = os.path.join(root, name)
            
            # Determine the relative output directory
            relative_dir = os.path.relpath(root, input_dir)
            output_folder_path = os.path.join(output_dir, relative_dir)
            
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            
            # Determine the full output file path
            output_file_name = f"{os.path.splitext(name)[0]}.csv"
            output_file_path = os.path.join(output_folder_path, output_file_name)
            
            # Add headers and save as CSV
            add_headers_and_save(input_file_path, output_file_path, headers)

# Define the headers for the CSV file
headers = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']

# Define the input and output directories
input_dir = '/Users/mohamadghajar/Downloads/SisFallDatasetAnnotation-master/SisFall_dataset'
output_dir = "/Users/mohamadghajar/Downloads/SisFallDatasetAnnotation-master/sis_csv"

# Process the directory
process_directory(input_dir, output_dir, headers)
