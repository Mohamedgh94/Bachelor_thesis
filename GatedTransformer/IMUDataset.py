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
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        label_vector = self.labels[idx]
        
        return torch.tensor(feature_vector, dtype=torch.float32), torch.tensor(label_vector, dtype=torch.float32)


# Create instances of the IMUDataset class for each dataset
train_dataset = IMUDataset("/data/malghaja/Bachelor_thesis/Sis_train_data.csv")
valid_dataset = IMUDataset("data/malghaja/Bachelor_thesis/Sis_valid_data.csv")
test_dataset = IMUDataset("/data/malghaja/Bachelor_thesis/Sis_test_data.csv")

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
