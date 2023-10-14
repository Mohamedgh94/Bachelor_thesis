import scipy.io
import pandas as pd
import numpy as np
import os
from pandas import read_csv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

""" # Pfad zur .mat-Datei
mat_file_path = "/Users/mohamadghajar/Downloads/UniMiB-SHAR/data/full_data.mat"
output_directory = '/Users/mohamadghajar/Downloads/UniMiB-SHAR/untitled folder 2'
# Laden Sie die .mat-Datei
import scipy.io

# Load the dataset
mat = scipy.io.loadmat(mat_file_path)
print(mat.keys())
full_data = mat['full_data']
# Revising the function to save sensor data for each subject and each activity
def save_all_sensor_data_v2(mat_file_path, output_directory):
    # Load the .mat file
    mat = scipy.io.loadmat(mat_file_path)
    full_data = mat['full_data']
    
    # Get the list of activity names from dtype names
    activity_names = full_data[0][0][0].dtype.names
    print(f"Activity names: {activity_names}")
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through each subject (0 to 29 for 30 subjects)
    for subject_idx in range(30):
        subject_data = full_data[subject_idx][0][0]
        subject_folder = os.path.join(output_directory, f"S{str(subject_idx + 1).zfill(2)}")
        
        # Create a directory for the subject if it doesn't exist
        if not os.path.exists(subject_folder):
            os.makedirs(subject_folder)
        
        # Loop through each activity for the subject
        for activity_name in activity_names:
            print(f"Processing subject {subject_idx + 1}, activity {activity_name}...")
            #print(subject_data[activity_name].shape)
            activity_data = subject_data[0][activity_name][0][0]
            print(f"Shape of activity_data for {activity_name}: {activity_data.shape}")
            # Debug: Print the content of subject_data[activity_name]
            #print(f"Content of subject_data[{activity_name}]: {[activity_data]}")

            print(f"Shape of activity_data for {activity_name}: {activity_data.shape}")
            print(f"Type of activity_data for {activity_name}: {type(activity_data)}")
            column_names = ['accel_x', 'accel_y', 'accel_z', 'timestamp', 'time_instants', 'signal_magnitude']
            #activity_data = pd.DataFrame(activity_data.T, columns=column_names)


            # Skip if the activity data is empty for this subject
            if activity_data.size == 0:
                continue
            try:
                activity_data_df = pd.DataFrame(activity_data.T, columns=column_names)
            except ValueError as e:
                print(f"ValueError encountered for activity {activity_name}: {e}")
                continue
            # Save the data to a CSV file within the subject's folder
            activity_csv_path = os.path.join(subject_folder, f"{activity_name}.csv")
            pd.DataFrame(activity_data_df).to_csv(activity_csv_path, index=False)

# Run the function to save all sensor data
save_all_sensor_data_v2(mat_file_path, output_directory)

# Check if there are any empty folders
empty_folders_check = []
for subject_folder in os.listdir(output_directory):
    subject_path = os.path.join(output_directory, subject_folder)
    if os.path.isdir(subject_path):
        if not os.listdir(subject_path):
            empty_folders_check.append(subject_folder)

empty_folders_check """


DATA_DIR = '/Users/mohamadghajar/Downloads/UniMiB-SHAR/untitled folder 2'
SUBJECT_IDS = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','17','18','19','20','21','22','23','24','25','26','27','28','29','30']

WINDOW_SIZE = 200
STRIDE = 50
SOFT_BIOMETRICS= ['age', 'height', 'weight', 'gender']

def get_person_info(subject_id):
    file_path = '/Users/mohamadghajar/Downloads/UniMiB-SHAR/subjects_info.csv'  
    person_info = pd.read_csv(file_path)
    
    # Filter the DataFrame based on the Subject_ID
    filtered_info = person_info[person_info['Subject_ID'] == int(subject_id)]
    
    if filtered_info.empty:
        print(f"No information found for subject {subject_id}")
        return None
    
    person_list = []
    for _, row in filtered_info.iterrows():
        temp = [row['Subject_ID'], row['Gender'], row['Age'], row['Height'], row['Weight']]
        person_list.append(temp)
    
    # Create a DataFrame with the person information
    columns = ['subject', 'gender', 'age', 'height', 'weight']
    person_df = pd.DataFrame(person_list, columns=columns)
    
    return person_df  # Returning the DataFrame directly

def process_file(file_path, subject_id):
    segments = []  # Define a list for the segments
    try:
        # Read the data in chunks and process each chunk individually
        for chunk in pd.read_csv(file_path, header=0, chunksize=1000):  # Use chunksize to reduce memory usage
            

            # Apply sliding window segmentation
            for i in range(0, len(chunk) - WINDOW_SIZE + 1, STRIDE):
                segment = chunk.iloc[i:i + WINDOW_SIZE].copy()  # Create a segment
                segments.append(segment)  # Add the segment to the list
        
        # Add the person ID and soft biometric labels
        person_info = get_person_info(subject_id)  # Correct call to get_person_info
        for segment in segments:
            segment['person_id'] = subject_id
            for label in SOFT_BIOMETRICS:
                segment[label] = person_info[label].values[0]  # Correct usage of person_info

    except Exception as e:
        print(f"Error processing {file_path} : {e}")
        return None
    return segments

def process_subject(subject_id):
    print(f'Processing subject {subject_id}...')
    subject_dir = os.path.join(DATA_DIR, 'S' + subject_id)
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
        sensor_cols = all_data.iloc[:, :-5].columns
        #all_data['MMA8451Q_x'] = all_data['MMA8451Q_x'].str.split(';').apply(lambda x: [float(i) for i in x])
        all_data[sensor_cols] = all_data[sensor_cols].astype(float)

        all_data[sensor_cols] = scaler.fit_transform(all_data[sensor_cols])

        # Encode the person IDs and soft biometric labels
        print('Encoding labels...')
        label_encoders = {}  
        for col in ['person_id'] + SOFT_BIOMETRICS:
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col])
            label_encoders[col] = le  # Store the encoder
            
    except Exception as e:
        print(f"Error in normalize_and_encode: {e}")
        return None

def extract_features(segment):
    features = []
    sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'timestamp', 'time_instants', 'signal_magnitude']
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
    sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'timestamp', 'time_instants', 'signal_magnitude']
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
        # Now, concatenate the X and y DataFrames for each split and save them to CSV
        train_data = pd.concat([X_train, y_train], axis=1)
        valid_data = pd.concat([X_valid, y_valid], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv('Unimib_train_data.csv', index=False)
        valid_data.to_csv('Unimib_valid_data.csv', index=False)
        test_data.to_csv('Unimib_test_data.csv', index=False)
    except Exception as e:
        print(f"Error in split_and_save_data: {e}")


def main():
    all_segments = []
    for subject_id in SUBJECT_IDS:
        subject_segments = process_subject(subject_id)
        
        if subject_segments:
            
            all_segments.extend(subject_segments)

    try:
        all_data = pd.concat(all_segments, axis=0, ignore_index=True)
        
        print('Extracting features...')
        feature_df = all_data.groupby('person_id').apply(lambda segment: extract_features(segment))
        feature_df.reset_index(inplace=True)
        print('Feature extraction complete.')
        all_data = pd.merge(all_data, feature_df, on='person_id', how='left')
        all_data = remove_original_sensor_data(all_data) 
        all_data = rearrange_columns(all_data)
        print(all_data.head())
        X = all_data.iloc[:, :-5]
        y = all_data[['person_id', 'age', 'height', 'weight', 'gender']]
        print('Normalizing and encoding...')
        normalize_and_encode(all_data)
        print('Normalization and encoding complete.')
        print(all_data.head())
        
      
        split_and_save_data(X, y)

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()        