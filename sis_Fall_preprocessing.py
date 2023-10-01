import re  
import os
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib  # for saving the scaler object

# Initialize logging
logging.basicConfig(filename='preprocessing.log', level=logging.INFO)

# Configuration Parameters
data_dir = '/Users/mohamadghajar/Downloads/SisFallDatasetAnnotation-master/untitled folder/'
subject_ids = ['SA01', 'SA02', 'SA03', 'SA04', 'SA05', 'SA06', 'SA07', 'SA08', 'SA09', 'SA10', 'SA11', 'SA12', 'SA13', 'SA14', 'SA15', 'SA16', 'SA17', 'SA18', 'SA19', 'SA20', 'SA21', 'SA22', 'SA23', 'SE01', 'SE02', 'SE03', 'SE04', 'SE05', 'SE06', 'SE07', 'SE08', 'SE09', 'SE10', 'SE11', 'SE12', 'SE13', 'SE14', 'SE15']

window_size = 200
stride = 50
soft_biometrics = ['age', 'height', 'weight', 'gender']

def get_person_info(subject_id):
    """Get the person information from the Readme.txt file."""
    

    # Open the readme file and read the lines
    file_path = '/Users/mohamadghajar/Downloads/SisFallDatasetAnnotation-master/untitled folder/Readme.txt'
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
    try:
        # Define a list for the segments
        segments = []
        
        # Read the data in chunks and process each chunk individually
        for chunk in pd.read_csv(file_path, header=0, chunksize=1000):  # Use chunksize to reduce memory usage
            
            # Apply sliding window segmentation
            for i in range(0, len(chunk) - window_size + 1, stride):
                
                segment = chunk.iloc[i:i + window_size].copy()  # Create a segment
                segments.append(segment)  # Add the segment to the list
        
        # Add the person ID and soft biometric labels
        person_info = get_person_info(subject_id)
        for segment in segments:
            segment['person_id'] = subject_id
            for label in soft_biometrics:
                segment[label] = person_info[label].values[0]
       
          

        return segments
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_subject(subject_id):

    print(f'Processing subject {subject_id}...')
    subject_dir = os.path.join(data_dir, subject_id)
    file_list = os.listdir(subject_dir)
    all_segments = []
    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(subject_dir, file_name)
            segments = process_file(file_path, subject_id)
            if segments is not None:
                all_segments.extend(segments)
    
    return all_segments

def normalize_and_encode(all_data):
    """Normalize sensor data and encode labels."""
    try:
        scaler = StandardScaler()
        sensor_cols = ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']
        all_data['MMA8451Q_x'] = all_data['MMA8451Q_x'].str.split(';').apply(lambda x: [float(i) for i in x])
        all_data[sensor_cols] = all_data[sensor_cols].astype(float)

        all_data[sensor_cols] = scaler.fit_transform(all_data[sensor_cols])

        # Encode the person IDs and soft biometric labels
        print('Encoding labels...')
        label_encoders = {}  
        for col in ['person_id'] + soft_biometrics:
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col])
            label_encoders[col] = le  # Store the encoder
            joblib.dump(scaler, 'scaler.pkl')
    except Exception as e:
        logging.error(f"Error in normalize_and_encode: {e}")
        return None

def extract_features(segment):
    """Extract features from a single segment."""
    features = []
    for col in segment.columns:
        if col in ['ADXL345_x', 'ADXL345_y', 'ADXL345_z', 'ITG3200_x', 'ITG3200_y', 'ITG3200_z', 'MMA8451Q_x', 'MMA8451Q_y', 'MMA8451Q_z']:
            features.append(segment[col].mean())
            features.append(segment[col].std())
            features.append(segment[col].max())
            features.append(segment[col].min())
            #mean = np.mean(segment[col])
            #std = np.std(segment[col])
            #max = np.max(segment[col])
            #min = np.min(segment[col])
            #features =np.hstack([mean, std, max, min])
      
    return features



def split_and_save_data(X, y):
    """Split the data into train, validation, and test sets and save them."""
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

        train_data.to_csv('train_data.csv', index=False)
        valid_data.to_csv('valid_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)
    except Exception as e:
        print(f"Error in split_and_save_data: {e}")

def main():
    all_segments = []
     
    
    for subject_id in subject_ids:
        subject_segments = process_subject(subject_id)
        if subject_segments:
            all_segments.extend(subject_segments)
       
    all_data = pd.concat(all_segments, axis=0, ignore_index=True)
    normalize_and_encode(all_data)
   
    X = all_data.iloc[:, :-5]
    y = all_data[['person_id', 'age', 'height', 'weight', 'gender']]
    
    split_and_save_data(X, y)
    
if __name__ == "__main__":
    main()
