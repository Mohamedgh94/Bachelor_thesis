import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging


logging.basicConfig(filename='cnn_lstm_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
            #'person_id': torch.tensor(label_vector[0], dtype=torch.long),
            'age': torch.tensor(label_vector[1], dtype=torch.long),
            'height': torch.tensor(label_vector[2], dtype=torch.long),
            'weight': torch.tensor(label_vector[3], dtype=torch.long),
            'gender': torch.tensor(label_vector[4], dtype=torch.long),
        }
        feature_vector = feature_vector.reshape(1, -1)
        return torch.tensor(feature_vector, dtype=torch.float32), label_dict
        #feature_vector = feature_vector.reshape(1, -1)
        #print("Feature vector shape:", feature_vector.shape)
        #return torch.tensor(feature_vector, dtype=torch.float32), label_dict

    @staticmethod
    def get_combined_categories(*datasets):
        combined_categories = {}
        for dataset in datasets:
            for key, value in dataset.label_categories.items():
                combined_categories.setdefault(key, set()).update(value)
        combined_categories = {key: list(values) for key, values in combined_categories.items()}
        return combined_categories



train_dataset = IMUDataset("/data/malghaja/Bachelor_thesis/SisCat_train_data.csv")
valid_dataset = IMUDataset("/data/malghaja/Bachelor_thesis/SisCat_valid_data.csv")
test_dataset = IMUDataset("/data/malghaja/Bachelor_thesis/SisCat_test_data.csv")
# train_dataset = IMUDataset("/Users/mohamadghajar/Documents/BAC/Sis_train_data.csv")
# valid_dataset = IMUDataset("/Users/mohamadghajar/Documents/BAC/Sis_valid_data.csv")
# test_dataset= IMUDataset("/Users/mohamadghajar/Documents/BAC/Sis_test_data.csv")



# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size= 256, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


##############
##############

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNNLSTM, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        #new
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        #new
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(in_channels= 128 , out_channels= 256 , kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        #self.fc_intermediate = nn.Linear(256, 128)
        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size= 128 ,hidden_size = hidden_size, num_layers = 2,batch_first = True)

        #
        self.fc1 = nn.Linear(hidden_size,256)
        self.fc2 = nn.Linear(256,128)
        #new
       
        # Output heads
        """ self.fc_age = nn.Linear(hidden_size, num_classes['age'])
        self.fc_height = nn.Linear(hidden_size, num_classes['height'])
        self.fc_weight = nn.Linear(hidden_size, num_classes['weight'])
        self.fc_gender = nn.Linear(hidden_size, num_classes['gender'])
 """
        
        
        self.fc_age = nn.Linear(hidden_size, 2)  # 2 classes for age
        self.fc_height = nn.Linear(hidden_size, 2)  # 2 classes for height
        self.fc_weight = nn.Linear(hidden_size, 2)  # 2 classes for weight
        self.fc_gender = nn.Linear(hidden_size, 2)  # 2 classes for gender
        # Activation function for gender
        

        logging.info(f"Initialized CNN-LSTM model with architecture: {self}")
    def forward(self, x):

        # print(f"Original shape: {x.shape}")
        x = x.permute(0, 2, 1)
        # print(f"Shape after permute: {x.shape}")
        # Convolutional layers
        x = self.conv1(x)  # First convolution
        x = self.relu(x)   # Apply ReLU
        x = self.dropout1(x)  # Apply dropout

        x = self.conv2(x)  # Second convolution
        x = self.relu2(x)  # Apply ReLU
        x = self.dropout2(x)  # Apply dropout

        x = self.conv3(x)  # Third convolution
        x = self.relu3(x)  # Apply ReLU
        x = self.dropout3(x)  # Apply dropout

        # Global Max Pooling
        x = F.max_pool1d(x, kernel_size=x.size(2))  # Global max pooling
        x = x.permute(0, 2, 1)  # Rearrange dimensions for LSTM input

        #x = self.fc_intermediate(x)
        # LSTM layers
        x, _ = self.lstm1(x)
        #x = self.fc_intermediate(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Get the last time step's output

        # Dense layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # Output layers
        age = self.fc_age(x)
        height = self.fc_height(x)
        weight = self.fc_weight(x)
        gender = self.fc_gender(x)  

        return age, height, weight, gender
########################################################################
    
def infer(model, input):
    model.eval()
    with torch.no_grad():
        age_logits, height_logits, weight_logits, gender_logits = model(input)
        
        age_probs = F.softmax(age_logits, dim=1)
        height_probs = F.softmax(height_logits, dim=1)
        weight_probs = F.softmax(weight_logits, dim=1)
        gender_probs = F.softmax(gender_logits, dim=1)

        # Now age_probs, height_probs, weight_probs, and gender_probs contain
        # the probabilities for each class
        return age_probs, height_probs, weight_probs, gender_probs
    
    
def __repr__(self):
        # String representation of your model
        representation = "CNNLSTM(\n"
        # Add details about each layer, hyperparameters, etc.
        representation += f"\tInput Size: {self.input_size}\n"
        representation += f"\tHidden Size: {self.hidden_size}\n"
        representation += f"\tNumber of Classes: {self.num_classes}\n"
        # Add details for each layer
        representation += f"\tConv1: {self.conv1}\n"
        representation += f"\tConv2: {self.conv2}\n"
        # Continue for other layers...
        representation += ")"
        return representation

    
    

def combined_loss(predictions, targets):
    # Unpack predictions
    """ age_pred, height_pred, weight_pred, gender_pred = predictions

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

    print(f"age_pred shape: {age_pred.shape}, age_target shape: {age_target.shape}")
    print(f"height_pred shape: {height_pred.shape}, age_target shape: {age_target.shape}")
    print(f"weight_pred shape: {weight_pred.shape}, age_target shape: {age_target.shape}") """
    
    age_pred, height_pred, weight_pred, gender_pred = predictions
    age_target, height_target, weight_target, gender_target = targets

    # Compute classification loss (Cross-Entropy) for all tasks
    loss_age = F.cross_entropy(age_pred, age_target)
    loss_height = F.cross_entropy(height_pred, height_target)
    loss_weight = F.cross_entropy(weight_pred, weight_target)
    loss_gender = F.cross_entropy(gender_pred, gender_target)

    # Combine losses
    total_loss = loss_age + loss_height + loss_weight + loss_gender
    return total_loss
    """ # Compute classification loss (Cross-Entropy)
    loss_age = F.cross_entropy(age_pred, age_target)
    loss_height = F.cross_entropy(height_pred, height_target)
    loss_weight = F.cross_entropy(weight_pred, weight_target)
    loss_gender = F.cross_entropy(gender_pred, gender_target)

    # Combine losses
    total_loss = loss_age + loss_height + loss_weight + loss_gender
    return total_loss """

##################################################
""" def train(model, train_loader, optimizer, device):
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
    logging.info(f"Training - Epoch Loss: {avg_loss}")
    return avg_loss
 """

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features = features.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

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
    logging.info(f"Validation - Epoch Loss: {avg_loss}")
    
    return avg_loss

#################################################
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_recall_fscore_support

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def test(model, test_loader, device):
    model.eval()
    age_preds, age_targets = [], []
    height_preds, height_targets = [], []
    weight_preds, weight_targets = [], []
    gender_preds, gender_targets = [], []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), {k: v.to(device) for k, v in labels.items()}

            age_pred, height_pred, weight_pred, gender_pred = model(features)
            age_targets += labels['age'].tolist()
            height_targets += labels['height'].tolist()
            weight_targets += labels['weight'].tolist()
            gender_targets += labels['gender'].tolist()

            age_preds += age_pred.argmax(dim=1).tolist()
            height_preds += height_pred.argmax(dim=1).tolist()
            weight_preds += weight_pred.argmax(dim=1).tolist()
            gender_preds += gender_pred.argmax(dim=1).tolist()

    # Classification Metrics
    accuracy_age = accuracy_score(age_targets, age_preds)
    accuracy_height = accuracy_score(height_targets, height_preds)
    accuracy_weight = accuracy_score(weight_targets, weight_preds)
    accuracy_gender = accuracy_score(gender_targets, gender_preds)

    precision_age, recall_age, f1_age, _ = precision_recall_fscore_support(age_targets, age_preds, average='binary')
    precision_height, recall_height, f1_height, _ = precision_recall_fscore_support(height_targets, height_preds, average='binary')
    precision_weight, recall_weight, f1_weight, _ = precision_recall_fscore_support(weight_targets, weight_preds, average='binary')
    precision_gender, recall_gender, f1_gender, _ = precision_recall_fscore_support(gender_targets, gender_preds, average='binary')

    metrics = {
        'accuracy_age': accuracy_age, 'precision_age': precision_age, 'recall_age': recall_age, 'f1_age': f1_age,
        'accuracy_height': accuracy_height, 'precision_height': precision_height, 'recall_height': recall_height, 'f1_height': f1_height,
        'accuracy_weight': accuracy_weight, 'precision_weight': precision_weight, 'recall_weight': recall_weight, 'f1_weight': f1_weight,
        'accuracy_gender': accuracy_gender, 'precision_gender': precision_gender, 'recall_gender': recall_gender, 'f1_gender': f1_gender
    }
    
    return metrics


def check_gender_predictions(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels['gender'].to(device)
            predictions = model(features)
            gender_pred = predictions[3]  # Assuming gender is the fourth output
            gender_pred = torch.argmax(gender_pred, dim=1)
            all_preds.extend(gender_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Model loaded from {self.model_path}")

import time

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    hidden_size = 256
    input_size = 45
    num_classes = {
        'age': 2, 
        'height': 2,  
        'weight': 2,  
        'gender': 2  
    }

    # Ask user for the operation mode
    mode = input("Enter mode (train/test/both): ").strip().lower()
    if mode not in ['train', 'test', 'both']:
        print("Invalid mode selected. Exiting.")
        return

    model = CNNLSTM(input_size, hidden_size, num_classes).to(device)

    if mode in ['train', 'both']:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        num_epochs = 3
        start_time = time.time()

        # Training and Validation Loop
        for epoch in range(num_epochs):
            print(f'Training Epoch {epoch+1}/{num_epochs}')
            train_loss = train(model, train_loader, optimizer, device)
            valid_loss = validate(model, valid_loader, device)
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {valid_loss}')
            trainng_time  = (time.time() - start_time)/60
            print(f'Epoch {epoch+1} trainng time {trainng_time}')
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Save the trained model
        model_save_path = 'saved_model.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    if mode in ['test', 'both']:
        if mode == 'test':
            # Load the pretrained model
            model_load_path = input("Enter the path to the saved model: ").strip()
            model.load_state_dict(torch.load(model_load_path))
            model.eval()

        # Test the model
        test_metrics = test(model, test_loader, device)
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
trainng_timetrainng_time