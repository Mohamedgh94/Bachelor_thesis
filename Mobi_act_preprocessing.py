import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import gc 
from multiprocessing import Queue, Process, Pool
import time
# Configuration parameters
DATA_DIR = '/data/datasets/act_datasets/Annotated Data'
WINDOW_SIZE = 200
STRIDE = 50
SUBJECT_INFO_FILE = '/data/datasets/act_datasets/Annotated Data/Readme.txt'

def read_subject_info(file_path):
    """
    Reads subject information from a file and returns a pandas DataFrame.

    Args:
        file_path (str): The path to the file containing the subject information.

    Returns:
        pandas.DataFrame: A DataFrame containing the subject information, with columns for subject ID, age, height, weight, and gender.
    """
    with open(file_path, 'r', encoding='latin1') as file:
        strings = file.readlines()
    file.close()
    person_list = []
    for s in strings:
        if 'sub' in s and '|' in s:
            temp = s.split('|')
            temp = [x.strip() for x in temp]
            if len(temp) == 9:
                person_list.append(temp[3:-1])
    columns = ['subject', 'age', 'height', 'weight', 'gender']
    person_info = pd.DataFrame(person_list, columns=columns)
    person_info[['age', 'height', 'weight']] = person_info[['age', 'height', 'weight']].apply(pd.to_numeric)
    person_info['gender'] = pd.Categorical(person_info['gender'], categories=['M', 'F','Ì '])
    return person_info


def process_file(act, data_dir):
    """
    Process the data files for a given activity and return the segmented data.
    Modified to read files in chunks to save memory.
    """
    segments = []
    #file_list = os.listdir(data_dir + act + '/')
    file_list = os.listdir(os.path.join(data_dir, act))
    for idx in range(0, len(file_list), 10):  # process 10 files at a time
        sub_list = file_list[idx: idx + 10]
        for file in sub_list:
            file_path = os.path.join(data_dir, act, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Failed to process {file_path}. Error: {e}")
                continue
            file_names = file.split('_')
            subject_id = int(file_names[1])
            df['subject_id'] = subject_id

            new_segments = segment_data(df, WINDOW_SIZE, STRIDE)
            segments.extend(new_segments)
        del sub_list  # Free up memory
        gc.collect()  # Explicit garbage collection
    return segments

def segment_data(df, window_size, stride):
    segments = []
    for i in range(0, len(df) - window_size, stride):
        segment = df.iloc[i:i + window_size]
        segments.append(segment)
    return segments


def extract_features(segment, sensor_cols):
    """
    Extracts statistical features from a given segment of sensor data.

    Args:
    segment of sensor data.
    sensor_cols (list): A list of column names to extract features from.

    Returns:
    pandas.Series: A series of statistical features extracted from the segment.
    """
    features = []
    for col in sensor_cols:
        features.append(segment[col].min())
        features.append(segment[col].max())
        features.append(segment[col].mean())
        features.append(segment[col].std())
        features.append(np.sqrt(np.mean(segment[col]**2)))
    feature_names = [
        f'{col}_{stat}' for col in sensor_cols 
        for stat in ['min', 'max','mean', 'std','rms']
    ]
    
    return pd.Series(features, index=feature_names).astype('float32')
# Remove the original sensor columns
def remove_original_sensor_data(df):
    sensor_cols = ['timestamp', 'rel_time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll', 'label']
    return df.drop(columns=sensor_cols)

def normalize_data_in_batches(df, batch_size=1000):
    exclude_cols = ['subject_id', 'age', 'height', 'weight', 'gender']
    cols_to_normalize = df.drop(columns=exclude_cols).columns
    
    scaler = StandardScaler()
    
    # Aufteilen des DataFrames in eine Liste von kleineren DataFrames
    dfs = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
    
    normalized_dfs = []  # Liste zum Speichern der normalisierten DataFrames
    
    for small_df in dfs:
        # Erstellen einer Kopie des kleinen DataFrames, um Warnungen zu vermeiden
        small_df_copy = small_df.copy()
        small_df_copy.loc[:, cols_to_normalize] = scaler.fit_transform(small_df[cols_to_normalize])
        normalized_dfs.append(small_df_copy)
         # Zusammenf  hren der normalisierten DataFrames zur  ck in einen gro ^=en DataFrame
    normalized_df = pd.concat(normalized_dfs, ignore_index=True)
    return normalized_df  # R  ckgabe des normalisierten DataFrames
def label_encode(df, subject_info):
    """
    Encodes categorical columns in the given dataframe using LabelEncoder.

    Args:
        df The dataframe to be encoded.
        subject_info The dataframe containing subject information.

    Returns:
        None
    """
    if df is None:
        print("DataFrame is None in label_encode.")
        return
    le = LabelEncoder()
    for col in ['subject', 'age', 'height', 'weight', 'gender']:
        if col in df.columns and col in subject_info.columns:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
def rearrange_columns(df):
    cols = list(df.columns)
    cols = [col for col in cols if col not in ['subject_id', 'age', 'height', 'weight', 'gender']]
    cols.extend(['subject_id', 'age', 'height', 'weight', 'gender'])
    return df[cols]
def split_and_save(df):
    X = df.iloc[:,:-5]
    y = df[['subject_id', 'age', 'height', 'weight', 'gender']]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    train_data = pd.concat([X_train, y_train], axis=1)
    valid_data = pd.concat([X_valid, y_valid], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv('mobiact_train.csv', index=False)
    valid_data.to_csv('mobiact_valid.csv', index=False)
    test_data.to_csv('mobiact_test.csv', index=False)
""" 
    train_shuffled  = train_data.sample(frac=1, random_state=1).reset_index(drop=True)
    valid_shuffled  = valid_data.sample(frac=1, random_state=1).reset_index(drop=True)
    test_shuffled  = test_data.sample(frac=1, random_state=1).reset_index(drop=True)
    train_shuffled.to_csv('mobiact_train.csv', index=False)
    valid_shuffled.to_csv('mobiact_valid.csv', index=False)
    test_shuffled.to_csv('mobiact_test.csv', index=False) 
"""
    
    
    

def process_file_parallel(act):
    """Ein Wrapper f  r die process_file-Funktion f  r Parallelisierung."""
    return process_file(act, DATA_DIR)

def main():
    print("Reading subject info...")
    start_time = time.time()
    subject_info = read_subject_info(SUBJECT_INFO_FILE)
    print(f"Subject info read in {time.time() - start_time:.2f} seconds.")

    # Verwenden von os.path.join f  r Dateipfade
    data_dir = os.path.join('/data/datasets/act_datasets', 'Annotated Data')
    act_list = [folder for folder in os.listdir(data_dir) if folder not in ['.DS_Store', 'MobiAct_data.csv', 'MobiAct_data.numbers', 'import pandas as pd.py', 'Readme.txt']]
    print("Processing files...")
    start_time = time.time()
   # Parallelisieren Sie den Dateiverarbeitungsprozess
    with Pool() as pool:
        all_segments = pool.map(process_file_parallel, act_list)

    print(f"Files processed in {time.time() - start_time:.2f} seconds.")
     # Optimierung des Speichers durch Verwenden einer List Comprehension
    all_data = pd.concat([segment for sublist in all_segments for segment in sublist], ignore_index=True)
    print("convert to float32")
    # Optimierung der Datentypen
    float_cols = all_data.select_dtypes(include=['float64']).columns
    all_data[float_cols] = all_data[float_cols].astype('float32', errors='raise')
    print(all_data[float_cols].dtypes)
    #all_data[float_cols] = all_data[float_cols].astype('float32')
    #all_data = all_data.astype({col: 'float32' for col in float_cols})
    cat_cols = all_data.select_dtypes(include=['category']).columns
    all_data[cat_cols] = all_data[cat_cols].astype('category')
    print("convert to float32 done")
    # L  schen und Garbage Collection
    del all_segments
    gc.collect()
    all_data = pd.merge(all_data, subject_info[['age', 'height', 'weight', 'gender']], left_on='subject_id', right_on=subject_info.index, how='left')
    print(all_data.head())
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z','azimuth','pitch','roll']  
    feature_df = all_data.groupby('subject_id').apply(lambda segment: extract_features(segment, sensor_cols))
    all_data = pd.merge(all_data, feature_df, on='subject_id', how='left')
    print(all_data.head())
    all_data = remove_original_sensor_data(all_data) 
    all_data =  rearrange_columns(all_data)
    
    print(all_data.head())
    print(all_data.info())
    print(all_data.describe())
    if all_data is not None:
        all_data = normalize_data_in_batches(all_data)
        label_encode(all_data, subject_info)
        split_and_save(all_data)
    gc.collect()

if __name__ == "__main__":
    main()
