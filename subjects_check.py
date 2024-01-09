import pandas as pd

data = pd.read_csv('/data/malghaja/Bachelor_thesis/mobiact_test.csv')
test_class_counts = data['person_id'].unique()
print(test_class_counts)
""" # Read the datasets
train_data = pd.read_csv('/Users/mohamadghajar/Desktop/Bachelor_version2/Bac/Bachelor_thesis/Unimib_train_data.csv')
valid_data = pd.read_csv('/Users/mohamadghajar/Desktop/Bachelor_version2/Bac/Bachelor_thesis/Unimib_valid_data.csv')
test_data = pd.read_csv('/Users/mohamadghajar/Desktop/Bachelor_version2/Bac/Bachelor_thesis/Unimib_test_data.csv')

# Define the categorization function
def categorize_age(age):
    return '0' if age < 40 else '1'

def categorize_height(height):
    return '0' if height < 165 else '1'

def categorize_weight(weight):
    return '0' if weight < 65 else '1'

# Apply the categorization
train_data['age'] = train_data['age'].apply(categorize_age)
train_data['height'] = train_data['height'].apply(categorize_height)
train_data['weight'] = train_data['weight'].apply(categorize_weight)

valid_data['age'] = valid_data['age'].apply(categorize_age)
valid_data['height'] = valid_data['height'].apply(categorize_height)
valid_data['weight'] = valid_data['weight'].apply(categorize_weight)

test_data['age'] = test_data['age'].apply(categorize_age)
test_data['height'] = test_data['height'].apply(categorize_height)
test_data['weight'] = test_data['weight'].apply(categorize_weight)

# Optionally, save the modified datasets
train_data.to_csv('UniCat_train_data.csv', index=False)
valid_data.to_csv('UniCat_valid_data.csv', index=False)
test_data.to_csv('UniCat_test_data.csv', index=False) """


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

""" import pandas as pd

def print_dataset_info(dataset, dataset_name):
    print(f'{dataset_name} Data subjects:')
    for person_id in dataset['person_id'].unique():
        person_data = dataset[dataset['person_id'] == person_id].iloc[0]
        #print(f"Person ID: {person_id}, Age: {person_data['age']}, Height: {person_data['height']}, Weight: {person_data['weight']}, Gender: {person_data['gender']}")
    
    print(f"{dataset_name} Age Distribution:\n{dataset['age'].value_counts()}")
    print(f"{dataset_name} Height Distribution:\n{dataset['height'].value_counts()}")
    print(f"{dataset_name} Weight Distribution:\n{dataset['weight'].value_counts()}")
    print(f"{dataset_name} Gender Distribution:\n{dataset['gender'].value_counts()}\n")

# Load data
data_train = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_train_data.csv')
data_valid = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_valid_data.csv')
data_test = pd.read_csv('/data/malghaja/Bachelor_thesis/Sis_test_data.csv')

# Print information
print_dataset_info(data_train, 'Train')
print_dataset_info(data_valid, 'Valid')
print_dataset_info(data_test, 'Test') """

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
