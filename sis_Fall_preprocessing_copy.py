import re
import os
import gc
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from multiprocessing import Pool



# Configuration Parameters
DATA_DIR = '/data/malghaja/SisFall_csv' 
README_FILE_PATH = os.path.join(DATA_DIR, 'Readme.txt')
SUBJECT_IDS = ['SA01', 'SA02', 'SA03', 'SA04', 'SA05', 'SA06', 'SA07', 'SA08', 'SA09', 'SA10', 'SA11', 'SA12', 'SA13', 'SA14', 'SA15', 'SA16', 'SA17', 'SA18', 'SA19', 'SA20', 'SA21', 'SA22', 'SA23', 'SE01', 'SE02']
WINDOW_SIZE = 200
STRIDE = 50
SOFT_BIOMETRICS = ['age', 'height', 'weight', 'gender']

def get_person_info(subject_id):
    # Open the readme file and read the lines
    file_path = '/data/malghaja/SisFall_csv/Readme.txt'
    with open(file_path, 'r', encoding='latin1') as file:
        strings = file.readlines()

    # Parse the person information for the given person ID
    person_list = []
    for s in strings:
        if re.match(f'^\| {subject_id}', s):   # The line starts with '| {subject_id}'
            temp = s.split('|')
            temp = [x.strip() for x in temp]
            if len(temp) == 7:
                person_list.append(temp[1:-1])
               # If person_list is empty, return an empty DataFrame
            if not person_list:
                return pd.DataFrame(columns=['subject', 'age', 'height', 'weight', 'gender'])

            # Create a DataFrame with the person information
            columns = ['subject', 'age', 'height', 'weight', 'gender']
            person_info = pd.DataFrame(person_list, columns=columns)

            # Convert the age, height, and weight columns to numeric values
            person_info[['age', 'height', 'weight']] = person_info[['age', 'height', 'weight']].apply(pd.to_numeric)

            # Encode the gender column as a categorical variable
            person_info['gender'] = pd.Categorical(person_info['gender'], categories=['M', 'F'])

    return person_info  # Returning the DataFrame directly

def process_file(file_path, subject_id):
    segments = []  # Define a list for the segments
    try:
        # Read the data in chunks and process each chunk individually
        for chunk in pd.read_csv(file_path, header=0, chunksize=1000):  # Use chunksize to reduce memory usage
# Apply sliding window segmentation
            for i in range(0, len(chunk) - WINDOW_SIZE + 1, STRIDE):
                segment = chunk.iloc[i:i + WINDOW_SIZE].copy()  # Create a segment
                segments.append(segment)  # Add the segment to the list
            person_info = get_person_info(subject_id)  # Correct call to get_person_info
        for segment in segments:
            segment['person_id'] = subject_id
            for label in SOFT_BIOMETRICS:
                segment[label] = person_info[label].values[0]  # Correct usage of person_info

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    return segments

def process_subject(subject_id):
    print(f'Processing subject {subject_id}...')
    subject_dir = os.path.join(DATA_DIR, subject_id)
    file_list = os.listdir(subject_dir)
    all_segments = []
    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(subject_dir, file_name)
            segments = process_file(file_path, subject_id)
            if segments is not None:
                all_segments.extend(segments)
    subject_dir = os.path.join(DATA_DIR, subject_id)  # Use defined constant
    return all_segments
def normalize_and_encode(all_data):
    try:
        scaler = StandardScaler()
        sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
        all_data['MMA8451Q_x'] = all_data['MMA8451Q_x'].str.split(';').apply(lambda x: [float(i) for i in x])
        all_data[sensor_cols] = all_data[sensor_cols].astype(float)

        all_data[sensor_cols] = scaler.fit_transform(all_data[sensor_cols])

        # Encode the person IDs and soft biometric labels
        print('Encoding labels...')
        label_encoders = {}  
        for col in ['person_id'] + SOFT_BIOMETRICS:
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col])
            label_encoders[col] = le  # Store the encoder
            joblib.dump(scaler, 'scaler.pkl')
    except Exception as e:
        logging.error(f"Error in normalize_and_encode: {e}")
        return None

def extract_features(segment):
    features = []
    sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
    for col in segment.columns:
        if col in  sensor_cols:
            features.append(segment[col].mean())
            features.append(segment[col].std())
            features.append(segment[col].max())
            features.append(segment[col].min())
            features.append(np.sqrt(np.mean(segment[col]**2)))
        feature_names = [
        f'{col}_{stat}' for col in sensor_cols 
        for stat in ['min', 'max','mean', 'std','rms']
    ]
    return pd.Series(features, index=feature_names).astype('float32')  

def remove_original_sensor_data(df):
    sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
    return df.drop(columns=sensor_cols)

def rearrange_columns(df):
    cols = list(df.columns)
    cols = [col for col in cols if col not in ['person_id', 'age', 'height', 'weight', 'gender']]
    cols.extend(['person_id', 'age', 'height', 'weight', 'gender'])
    return df[cols]

def split_and_save_data(X, y):
    try:
        # Concatenate the labels into a single string for stratification
        # y_stratify = y.apply(lambda x: '_'.join(x.map(str)), axis=1)
        print('Splitting data...')
        # Split the data into training and validation sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y['person_id'])

        # Split the temp data into validation and test sets
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp['person_id'])
        print('Splitting complete.')
        # Now, you can concatenate the X and y DataFrames for each split and save them to CSV
        train_data = pd.concat([X_train, y_train], axis=1)
        valid_data = pd.concat([X_valid, y_valid], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv('Sis_train_data.csv', index=False)
        valid_data.to_csv('Sis_valid_data.csv', index=False)
        test_data.to_csv('Sis_test_data.csv', index=False)
    except Exception as e:
        print(f"Error in split_and_save_data: {e}")


def main():
    sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
    all_segments = []
    for subject_id in SUBJECT_IDS:
        subject_segments = process_subject(subject_id)
        
        if subject_segments:
            
            all_segments.extend(subject_segments)

    try:
        all_data = pd.concat(all_segments, axis=0, ignore_index=True)
        float_cols = all_data.select_dtypes(include=['float64']).columns
        all_data[float_cols] = all_data[float_cols].astype('float32')
        cat_cols = all_data.select_dtypes(include=['category']).columns
        all_data[cat_cols] = all_data[cat_cols].astype('category')
        print(all_data.head())
        all_data['MMA8451Q_z'] = all_data['MMA8451Q_z'].str.replace(';', '').astype(float)
        print('Extracting features...')
        feature_df = all_data.groupby('person_id').apply(lambda segment: extract_features(segment))
        feature_df.reset_index(inplace=True)
        print('Feature extraction complete.')
        all_data = pd.merge(all_data, feature_df, on='person_id', how='left')
        all_data = remove_original_sensor_data(all_data) 
        all_data = rearrange_columns(all_data)
        print('Normalizing and encoding...')
        normalize_and_encode(all_data)
        print('Normalization and encoding complete.')
        #print(all_data.head())
        X = all_data.iloc[:, :-5]
        y = all_data[['person_id', 'age', 'height', 'weight', 'gender']]
      
        split_and_save_data(X, y)

    except ValueError as e:
        print(f"Error: {e}")  # Print the error message
if __name__ == "__main__":
    main()
