import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging


logging.basicConfig(filename='{dataset_name}}cnn_lstm.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

""" def get_dataset_paths(dataset_name):
        if dataset_name == "SisFall":
            return "/data/malghaja/Bachelor_thesis/SisCat_train_data.csv", "/data/malghaja/Bachelor_thesis/SisCat_valid_data.csv", "/data/malghaja/Bachelor_thesis/SisCat_test_data.csv"
        elif dataset_name == "MobiAct":
            return "path_to_MobiAct_train", "path_to_MobiAct_valid", "path_to_MobiAct_test"
        elif dataset_name == "Unimib":
            return "/data/malghaja/Bachelor_thesis/UniCat_train_data.csv", "/data/malghaja/Bachelor_thesis/UniCat_valid_data.csv", "/data/malghaja/Bachelor_thesis/UniCat_test_data.csv" 
 """
def get_dataset_paths(dataset_name):
        if dataset_name == "SisFall":
            return "/data/malghaja/Bachelor_thesis/SisCat_train_data.csv", "/data/malghaja/Bachelor_thesis/SisCat_valid_data.csv", "/data/malghaja/Bachelor_thesis/SisCat_test_data.csv"
        elif dataset_name == "MobiAct":
            return "path_to_MobiAct_train", "path_to_MobiAct_valid", "path_to_MobiAct_test"
        elif dataset_name == "Unimib":
            return "/Users/mohamadghajar/Desktop/py_exampels/UniCat_train_data.csv", "/Users/mohamadghajar/Desktop/py_exampels/UniCat_valid_data.csv", "/Users/mohamadghajar/Desktop/py_exampels/UniCat_test_data.csv"

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
        print("Feature vector shape:", feature_vector.shape) 
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
    
    
    


dataset_name = input("Enter dataset name: ")
train_path, valid_path, test_path = get_dataset_paths(dataset_name)
train_dataset = IMUDataset(train_path)
valid_dataset = IMUDataset(valid_path)
test_dataset = IMUDataset(test_path)
# train_dataset = IMUDataset("/Users/mohamadghajar/Documents/BAC/Sis_train_data.csv")
# valid_dataset = IMUDataset("/Users/mohamadghajar/Documents/BAC/Sis_valid_data.csv")
# test_dataset= IMUDataset("/Users/mohamadghajar/Documents/BAC/Sis_test_data.csv")



# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size= 256, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


##############
##############

import torch.nn as nn
import torch.nn.functional as F
import logging

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,num_classes,dropout_rate, kernel_size):
        super(CNNLSTM, self).__init__()

        # Convolutional layers 
        print(input_size)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=kernel_size, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(64)  # Layer norm after conv1
        self.relu = nn.ReLU()
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=1)
        self.ln2 = nn.LayerNorm(128)  # Layer norm after conv2
        
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, stride=1, padding=1)
        self.ln3 = nn.LayerNorm(256)  # Layer norm after conv3
        
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=kernel_size, stride=1, padding=1)
        self.ln4 = nn.LayerNorm(512)  # Layer norm after conv4
        
        self.relu4 = nn.ReLU()

        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=4, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=4, batch_first=True)

        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Final classification layers
        self.fc_age = nn.Linear(hidden_size, 2)  # 2 classes for age
        self.fc_height = nn.Linear(hidden_size, 2)  # 2 classes for height
        self.fc_weight = nn.Linear(hidden_size, 2)  # 2 classes for weight
        self.fc_gender = nn.Linear(hidden_size, 2)  # 2 classes for gender 

        self.sigmoid = nn.Sigmoid()

        logging.info(f"Initialized CNN-LSTM model with architecture: {self}")

    def forward(self, x):
        x = x.permute(0, 2, 1)

        # Convolutional layers with layer normalization
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.ln1(x)
        x = x.view(x.size(0), 64, -1)
        x = self.relu(x)
        x = self.dropout1(x)

        
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.ln2(x)
        x = x.view(x.size(0), 128, -1)
        x = self.relu2(x)
        x = self.dropout2(x)

        
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.ln3(x)
        x = x.view(x.size(0), 256, -1)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        x = self.ln4(x)
        x = x.view(x.size(0), 512, -1)
        x = self.relu4(x)

        # Global Max Pooling
        x = F.max_pool1d(x, kernel_size=x.size(2))
        x = x.permute(0, 2, 1)

        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Get the last time step's output

        # Dense layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # Final classification layers with sigmoid activation
        age_output = self.sigmoid(self.fc_age(x))
        height_output = self.sigmoid(self.fc_height(x))
        weight_output = self.sigmoid(self.fc_weight(x))
        gender_output = self.sigmoid(self.fc_gender(x))

        return age_output, height_output, weight_output, gender_output

        """ age_logits = self.fc_age(x)
        height_logits = self.fc_height(x)
        weight_logits = self.fc_weight(x)
        gender_logits = self.fc_gender(x)  

        age = F.softmax(age_logits, dim=1)
        height = F.softmax(height_logits, dim=1)
        weight = F.softmax(weight_logits, dim=1)
        gender = F.softmax(gender_logits, dim=1)
        return age, height, weight, gender """

        """
        return age, height, weight, gender
        # Output layers
        age = self.fc_age(x)
        height = self.fc_height(x)
        weight = self.fc_weight(x)
        gender = self.fc_gender(x)  

        return age, height, weight, gender """
########################################################################
    
def infer(model, input):
    model.eval()
    with torch.no_grad():
        age_logits, height_logits, weight_logits, gender_logits = model(input)
        
        age_probs = F.softmax(age_logits, dim=1)
        height_probs = F.softmax(height_logits, dim=1)
        weight_probs = F.softmax(weight_logits, dim=1)
        gender_probs = F.softmax(gender_logits, dim=1)

        
        return age_probs, height_probs, weight_probs, gender_probs
    
    
def __repr__(self):
        
        representation = "CNNLSTM(\n"
        
        representation += f"\tInput Size: {self.input_size}\n"
        representation += f"\tHidden Size: {self.hidden_size}\n"
        representation += f"\tNumber of Classes: {self.num_classes}\n"
        
        representation += f"\tConv1: {self.conv1}\n"
        representation += f"\tConv2: {self.conv2}\n"
        
        representation += ")"
        return representation

    
    

def combined_loss(predictions, targets):
    
    
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
   

##################################################


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
from sklearn.metrics import  accuracy_score, precision_recall_fscore_support


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
            gender_pred = predictions[3]  
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

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

def objective(params):
    lr = params['lr']
    batch_size = int(params['batch_size'])
    hidden_size = int(params['hidden_size'])
    dropout_rate = params['dropout_rate']
    kernel_size = int(params['kernel_size'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_classes = {
        'age': 2, 
        'height': 2,  
        'weight': 2,  
        'gender': 2  
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNNLSTM(15, hidden_size, num_classes, dropout_rate, kernel_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 10
    total_valid_loss = 0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        valid_loss = validate(model, valid_loader, device)
        total_valid_loss += valid_loss

    avg_valid_loss = total_valid_loss / num_epochs
    return {'loss': avg_valid_loss, 'status': STATUS_OK}
def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset_name = input("Enter dataset name: ")
    train_path, valid_path, test_path = get_dataset_paths(dataset_name)
    hidden_size = 128
    #input_size = 45
    num_classes = {
        'age': 2, 
        'height': 2,  
        'weight': 2,  
        'gender': 2  
    }
    # Dataset-specific configurations
    if dataset_name == "Unimib":
        input_size = 15
        #learning_rates = [0.0001]
        #batch_sizes = [200]
    else:
        input_size = 45
        #learning_rates = [0.0001, 0.00001, 0.000001]
        #batch_sizes = [64, 128, 256]

    train_dataset = IMUDataset(train_path)
    valid_dataset = IMUDataset(valid_path)
    test_dataset = IMUDataset(test_path)
    space = {
        'lr': hp.loguniform('lr', np.log(0.00001), np.log(0.001)),
        'batch_size': hp.choice('batch_size', [50, 100, 200]),
        'hidden_size': hp.choice('hidden_size', [64, 128, 256]),
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
        'kernel_size': hp.choice('kernel_size', [1,2,3])
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

    print("Best hyperparameters:", best)

    """ for lr, batch_size in zip(learning_rates, batch_sizes):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = CNNLSTM(input_size, hidden_size, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        logging.info(f"Dataset: {dataset_name}, Learning Rate: {lr}, Batch Size: {batch_size}, Model: {model}")
    

   
    mode = input("Enter mode (train/test/both): ").strip().lower()
    if mode not in ['train', 'test', 'both','feat']:
        print("Invalid mode selected. Exiting.")
        return

    model = CNNLSTM(input_size, hidden_size, num_classes).to(device)

    if mode in ['train', 'both']:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        num_epochs = 10
        start_time = time.time()

        
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
        model_save_path = f" CNN-LSTM,_{dataset_name}_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    if mode in ['test', 'both']:
        if mode == 'test':
            model_load_path = f" CNN-LSTM,_{dataset_name}_model.pth"
            model.load_state_dict(torch.load(model_load_path))
            model.eval()

        
        test_metrics = test(model, test_loader, device)
        logging.info(f"Test Results for {dataset_name} with LR: {lr}, Batch Size: {batch_size}: {test_metrics}")

        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}") """
    """ if mode == 'feat':        
        model_load_path = f" CNN-LSTM,_{dataset_name}_model.pth"
        model.load_state_dict(torch.load(model_load_path))
        model.eval()
        df = pd.read_csv('/data/malghaja/Bachelor_thesis/UniCat_valid_data.csv')


        X_val = df.iloc[:, :-5].values  # Features (all columns except the last 5)
        y_val = df.iloc[:, -4:].values 

        def predict(model, X):
    
            X_tensor = torch.tensor(X, dtype=torch.float32)

            # Forward pass and get predictions
            with torch.no_grad():
                model_output = model(X_tensor)
                # Assuming you need to apply a softmax or another activation function based on your model's output
                predictions = torch.softmax(model_output, dim=1).numpy()

            # Return the class with the highest probability (you may need to adjust this based on your exact requirements)
            return predictions.argmax(axis=1)
        
        from sklearn.inspection import permutation_importance


        result = permutation_importance(lambda X: predict(model, X), X_val, y_val[:, 0], n_repeats=10, random_state=42)


        for i in range(len(result.importances_mean)):
            print('Feature %d: %f' % (i, result.importances_mean[i]))
 """



if __name__ == "__main__":
    main()
