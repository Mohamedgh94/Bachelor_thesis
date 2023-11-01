"""
import pandas as pd


dft = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_train_data.csv')
print(f'training subjects: {dft["person_id"].unique()}')
dftest = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_valid_data.csv')
print(f'test subjects: {dftest["person_id"].unique()}')
dfv = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_test_data.csv')
print(f'validation subjects: {dfv["person_id"].unique()}')
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
