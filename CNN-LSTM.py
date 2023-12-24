import torch
import torch.nn as nn
import torch.nn.functional as F

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
        feature_vector = feature_vector.reshape(1, -1)
        #print("Feature vector shape:", feature_vector.shape)
        return torch.tensor(feature_vector, dtype=torch.float32), label_dict

    @staticmethod
    def get_combined_categories(*datasets):
        combined_categories = {}
        for dataset in datasets:
            for key, value in dataset.label_categories.items():
                combined_categories.setdefault(key, set()).update(value)
        combined_categories = {key: list(values) for key, values in combined_categories.items()}
        return combined_categories



train_dataset = IMUDataset("/Users/mohamadghajar/Desktop/Bachelor_version2/Bac/Bachelor_thesis/Unimib_train_data.csv")
valid_dataset = IMUDataset("/Users/mohamadghajar/Desktop/Bachelor_version2/Bac/Bachelor_thesis/Unimib_valid_data.csv")
test_dataset = IMUDataset("/Users/mohamadghajar/Desktop/Bachelor_version2/Bac/Bachelor_thesis/Unimib_test_data.csv")


# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=1028, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1028, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1028, shuffle=False)


##############
##############

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=2, batch_first=True)

        # Output heads
        self.fc_age = nn.Linear(hidden_size, num_classes['age'])
        self.fc_height = nn.Linear(hidden_size, num_classes['height'])
        self.fc_weight = nn.Linear(hidden_size, num_classes['weight'])
        self.fc_gender = nn.Linear(hidden_size, num_classes['gender'])

        # Activation function for gender
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # Convolutional layers
       # print(f'x.shape before',x.shape)
        x = self.relu(self.conv1(x))
        #print(f'x.shape after',x.shape)
        x = self.relu(self.conv2(x))

        x = F.max_pool1d(x, kernel_size=x.size(2))  # Global max pooling
        x = x.permute(0, 2, 1)
        # LSTM layer
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        # Output heads with activation functions
        age = self.fc_age(x)
        height = self.fc_height(x)
        weight = self.fc_weight(x)
        gender = self.softmax(self.fc_gender(x))

        return age, height, weight, gender
########################################################################
    import torch.nn.functional as F

def combined_loss(predictions, targets):
    # Unpack predictions
    age_pred, height_pred, weight_pred, gender_pred = predictions

    # Unpack targets
    age_target, height_target, weight_target, gender_target = targets

    # Squeeze the predictions to match the target shape
    age_pred = age_pred.squeeze()
    height_pred = height_pred.squeeze()
    weight_pred = weight_pred.squeeze()

    # Compute regression losses (MSE)
    loss_age = F.mse_loss(age_pred, age_target)
    loss_height = F.mse_loss(height_pred, height_target)
    loss_weight = F.mse_loss(weight_pred, weight_target)

    # Compute classification loss (Cross-Entropy)
    loss_gender = F.cross_entropy(gender_pred, gender_target)

    # Combine losses
    total_loss = loss_age + loss_height + loss_weight + loss_gender
    return total_loss

##################################################
def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        # Convert regression targets to Float
        labels['age'] = labels['age'].float()
        labels['height'] = labels['height'].float()
        labels['weight'] = labels['weight'].float()

        # Move data to the appropriate device (CPU or GPU)
        features, labels = features.to(device), {k: v.to(device) for k, v in labels.items()}

        # Forward pass
        predictions = model(features)
        targets = (labels['age'], labels['height'], labels['weight'], labels['gender'])

        # Compute loss
        loss = combined_loss(predictions, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

##############################################
def validate(model, valid_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in valid_loader:
            features, labels = features.to(device), {k: v.to(device) for k, v in labels.items()}

            predictions = model(features)
            targets = (labels['age'], labels['height'], labels['weight'], labels['gender'])
            loss = combined_loss(predictions, targets)

            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    return avg_loss

#################################################
def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), {k: v.to(device) for k, v in labels.items()}

            predictions = model(features)
            targets = (labels['age'], labels['height'], labels['weight'], labels['gender'])
            loss = combined_loss(predictions, targets)

            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f'Using device: {device}')
    hidden_size = 128  # Example hidden size, this can be tuned
    input_size = 15
    num_classes = {
        'age': 1,  # Regression (assuming age is a continuous value)
        'height': 1,  # Regression
        'weight': 1,  # Regression
        'gender': 2  # Classification (assuming gender is binary)
    }
    model = CNNLSTM(input_size, hidden_size, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training and Validation Loop
    num_epochs = 10
    for epoch in range(num_epochs):
       print(f'training ', epoch)
       train_loss = train(model, train_loader, optimizer, device)
       print(f'Epoch, {epoch+1}/{num_epochs},  Training Loss: {train_loss}')
       valid_loss = validate(model, valid_loader, device)
       print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')

    # Test the model
    test_loss = test(model, test_loader, device)
    print(f'Test Loss: {test_loss}')
if __name__ == "__main__":
    main()