import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class IMUDataset(Dataset):
    def __init__(self, csv_file):
        # Read the CSV file
        self.dataframe = pd.read_csv(csv_file)
        
        # Assuming the last 5 columns are labels
        self.labels = self.dataframe.iloc[:, -5:].values
        # Assuming all other columns are features
        self.features = self.dataframe.iloc[:, :-5].values
        self.label_categories = {}
        for column in self.dataframe.columns[-5:]:
            self.label_categories[column] = self.dataframe[column].unique()
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        label_vector = self.labels[idx]
        label_dict = {
            'person_id': label_vector[0],
            'age': label_vector[1],
            'height': label_vector[2],
            'weight': label_vector[3],
            'gender': label_vector[4],
        }
        
        return torch.tensor(feature_vector, dtype=torch.float32), label_dict #torch.tensor(label_vector, dtype=torch.float32)
    @staticmethod
    def get_combined_categories(*datasets):
        combined_categories = {}
        for dataset in datasets:
            for key, value in dataset.label_categories.items():
                if key not in combined_categories:
                    combined_categories[key] = set(value)
                else:
                    combined_categories[key].update(value)    
                #combined_categories[key].update(dataset.label_categories[key])
            # Convert sets back to lists
        combined_categories = {key: list(values) for key, values in combined_categories.items()}
        return combined_categories
    @staticmethod    
    def calculate_class_weights(class_counts):
        total_samples = sum(class_counts)
        num_classes = len(class_counts)
        weights = torch.zeros(num_classes)
        for class_idx, count in enumerate(class_counts):
            weights[class_idx] = total_samples / (num_classes * count)
        return weights
# Create instances of the IMUDataset class for each dataset
train_dataset = IMUDataset("/data/malghaja/Bachelor_thesis/UniCat_train_data.csv")
valid_dataset = IMUDataset("/data/malghaja/Bachelor_thesis/UniCat_valid_data.csv")
test_dataset = IMUDataset("/data/malghaja/Bachelor_thesis/UniCat_test_data.csv")
# train_dataset = IMUDataset("/Users/mohamadghajar/Documents/BAC/SisCat_train_data.csv")
# valid_dataset = IMUDataset("/Users/mohamadghajar/Documents/BAC/SisCat_valid_data.csv")
# test_dataset= IMUDataset("/Users/mohamadghajar/Documents/BAC/SisCat_test_data.csv")

global person_id_weights
global gender_weights
person_id_weights = IMUDataset.calculate_class_weights(train_dataset['person_id'].value_counts())
gender_weights = IMUDataset.calculate_class_weights(train_dataset['gender'].value_counts())
combined_categories = IMUDataset.get_combined_categories(train_dataset,valid_dataset,test_dataset)
global num_person_ids 
global num_ages
global num_heights
global num_weights
global num_genders
num_person_ids = len(combined_categories['person_id'])
num_ages = len (combined_categories['age'])
num_heights = len(combined_categories['height'])
num_weights = len(combined_categories['weight'])
num_genders = len(combined_categories['gender'])

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
