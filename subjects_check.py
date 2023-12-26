""" import pandas as pd
valid_data = pd.read_csv('/data/malghaja/Bachelor_thesis/mobiact_train.csv')
valid_shuffeld  = valid_data.sample(frac=1,random_state=1).reset_index(drop=True)
valid_shuffeld.to_csv('mobiact_train.csv ', index=False)
print(f'train data shuffeld complete')
test_data = pd.read_csv('/data/malghaja/Bachelor_thesis/mobiact_test.csv')
test_shuffeld  = test_data.sample(frac=1,random_state=1).reset_index(drop=True)
test_shuffeld.to_csv('mobiact_test.csv', index=False)
print(f'test data shuffeld complete')
train_data = pd.read_csv('/data/malghaja/Bachelor_thesis/mobiact_valid.csv')
train_shuffeld  = train_data.sample(frac=1,random_state=1).reset_index(drop=True)
train_shuffeld.to_csv('mobiact_valid.csv', index=False)
print(f'validation data shuffeld complete') """

import pandas as pd
import matplotlib.pyplot as plt
data_train = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_train_data.csv')
data_valid = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_valid_data.csv')
data_test = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_test_data.csv')
print(f'train Data subjects :')

for person_id in data_train['person_id'].unique():
    # Extract data for the current person ID
    person_data = data_train[data_train['person_id'] == person_id].iloc[0]
    age = person_data['age']
    height = person_data['height']
    weight = person_data['weight']
    gender = person_data['gender']
    print(f"Person ID: {person_id}, Age: {age}, Height: {height}, Weight: {weight}, Gender: {gender}")
print(f'valid Data subjects :')

for person_id in data_valid['person_id'].unique():
    # Extract data for the current person ID
    person_data = data_valid[data_valid['person_id'] == person_id].iloc[0]
    age = person_data['age']
    height = person_data['height']
    weight = person_data['weight']
    gender = person_data['gender']
    print(f"Person ID: {person_id}, Age: {age}, Height: {height}, Weight: {weight}, Gender: {gender}")
print(f'Test Data subjects :')
for person_id in data_test['person_id'].unique():
    # Extract data for the current person ID
    person_data = data_test[data_test['person_id'] == person_id].iloc[0]
    age = person_data['age']
    height = person_data['height']
    weight = person_data['weight']
    gender = person_data['gender']

    # Print or process the data as needed
    print(f" Person ID: {person_id}, Age: {age}, Height: {height}, Weight: {weight}, Gender: {gender}")
"""
import pandas as pd


dft = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_train_data.csv')
print(f'training subjects: {dft["person_id"].unique()}')
dftest = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_valid_data.csv')
print(f'test subjects: {dftest["person_id"].unique()}')
dfv = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_test_data.csv')
print(f'validation subjects: {dfv["person_id"].unique()}')
"""
"""
from PIL import Image

image = Image.open('/data/malghaja/Bachelor_thesis/GatedTransformer/confusion_matrix.png')
image.show()
"""
"""
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_test_data.csv')
print(f'validation subjects: {df["person_id"].unique()}')
unique_person_ids = df['person_id'].unique()
X = df.iloc[:,:-5]
y = df.iloc[:,-5:]
# Randomly split unique_person_ids into train, valid, and test
train_ids, test_ids = train_test_split(unique_person_ids, test_size=0.3, random_state=42)
valid_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

X_train = X[y['person_id'].isin(train_ids)]
y_train = y[y['person_id'].isin(train_ids)]

X_valid = X[y['person_id'].isin(valid_ids)]
y_valid = y[y['person_id'].isin(valid_ids)]

X_test = X[y['person_id'].isin(test_ids)]
y_test = y[y['person_id'].isin(test_ids)]
#print(f'X_train subjects: {X_train["person_id"].unique()}')
print(f'y_train subjects: {y_train["person_id"].unique()}')
print(f'y_valid subjects: {y_valid["person_id"].unique()}')
print(f'y_test subjects: {y_test["person_id"].unique()}')
#df['person_id'].value_counts().plot(kind='bar')
train_data = pd.concat([X_train, y_train], axis=1)
print(len(train_data.columns.to_list()))
valid_data = pd.concat([X_valid, y_valid], axis=1)
print(len(valid_data.columns.to_list()))
test_data = pd.concat([X_test, y_test], axis=1)
print(len(test_data.columns.to_list()))
train_data.to_csv('v_data.csv')
test_data.to_csv('vv_data.csv')
valid_data.to_csv('vvv_data.csv')
"""
