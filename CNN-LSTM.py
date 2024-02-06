import os
from networkx import configuration_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import seaborn as sns

import datetime

#logging.basicConfig(filename=' cnn_lstm.log', level=logging.INFO,
#                    format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(filename=f"{configuration['daset_name']}cnn_lstm.log', level=logging.INFO,
#                    format='%(asctime)s - %(levelname)s - %(message)s")




class IMUDataset(Dataset):
    def __init__(self, csv_file):

        # Read the CSV file
        self.dataframe = pd.read_csv(csv_file)
        # Assuming the last 5 columns are labels
        self.labels = self.dataframe.iloc[:, -6:-1].values
        # Assuming all other columns are features
        self.features = self.dataframe.iloc[:, :-6].values
        self.label_categories = {}
        for column in self.dataframe.columns[-6:]:
            self.label_categories[column] = self.dataframe[column].unique()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        feature_vector = self.features[idx]
        label_vector = self.labels[idx]
        label_dict = {
            'person_id': torch.tensor(label_vector[0], dtype=torch.long),
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
    
    
    


##############
##############

class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,config):
        super(CNNLSTM, self).__init__()

        self.config = config 
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
       # self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.relu2 = nn.ReLU()
        #self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(in_channels= 128 , out_channels= 256 , kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        #self.dropout3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3)
        #self.fc_intermediate = nn.Linear(256, 128)
        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size= 128 ,hidden_size = hidden_size, num_layers = 2,batch_first = True)
        self.lstm2 = nn.LSTM(input_size= hidden_size ,hidden_size = hidden_size, num_layers =1,batch_first = True)
        self.dropout5 = nn.Dropout(0.3)
        #NEW check if that increase the acc without Dense Layers
        #
        #self.fc1 = nn.Linear(hidden_size,256)
        #self.fc2 = nn.Linear(256,128)
        
        self.fc_person_id = nn.Linear(hidden_size, num_classes)
        

        self.fc_age = nn.Linear(hidden_size, 2)  
        self.fc_height = nn.Linear(hidden_size, 2)
        self.fc_weight = nn.Linear(hidden_size, 2)  
        self.fc_gender = nn.Linear(hidden_size, 2)  
        

        

        logging.info(f"Initialized CNN-LSTM model with architecture: {self}")
    def forward(self, x):

        # print(f"Original shape: {x.shape}")
        x = x.permute(0, 2, 1)
        # print(f"Shape after permute: {x.shape}")
        # Convolutional layers
        x = self.conv1(x)  # First convolution
        x = self.relu(x)   # Apply ReLU
        #x = self.dropout1(x)  # Apply dropout

        x = self.conv2(x)  # Second convolution
        x = self.relu2(x)  # Apply ReLU
        #x = self.dropout2(x)  # Apply dropout

        x = self.conv3(x)  # Third convolution
        x = self.relu3(x)  # Apply ReLU
        #x = self.dropout3(x)  # Apply dropout

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout4(x) #

        # Global Max Pooling
        #x = F.max_pool1d(x, kernel_size=x.size(2))  # Global max pooling
        x = x.permute(0, 2, 1)  # Rearrange dimensions for LSTM input

        
        # LSTM layers
        x, _ = self.lstm1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout5(x) # Apply dropout
        x = x[:, -1, :]  # Get the last time step's output

        # Dense layers
        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        if  self.config['output_type'] == 'softmax':
            person_id_output = torch.softmax(self.fc_person_id(x),dim=1)
            return person_id_output
        
        elif self.config['output_type'] == 'attribute':
            age = torch.sigmoid(self.fc_age(x))
            height = torch.sigmoid(self.fc_height(x))
            weight = torch.sigmoid(self.fc_weight(x))
            gender = torch.sigmoid(self.fc_gender(x))
            return age, height, weight, gender
        
########################################################################
    

def __repr__(self):
        
        representation = "CNNLSTM(\n"
        
        representation += f"\tInput Size: {self.input_size}\n"
        representation += f"\tHidden Size: {self.hidden_size}\n"
        representation += f"\tNumber of Classes: {self.num_classes}\n"
        
        representation += f"\tConv1: {self.conv1}\n"
        representation += f"\tConv2: {self.conv2}\n"
        
        representation += ")"
        return representation

    
    

def combined_loss(predictions, targets, config):
    output_type = config['output_type']
    if output_type == 'softmax':
        # Assuming the first element in predictions is for person_id
        person_id_pred = predictions[0]
        person_id_target = targets['person_id']
        loss = F.cross_entropy(person_id_pred, person_id_target)
    elif output_type == 'attribute':
        # Assuming the predictions are ordered as age, height, weight, gender
        age_pred, height_pred, weight_pred, gender_pred = predictions[0:]
        age_target, height_target, weight_target, gender_target = targets['age'], targets['height'], targets['weight'], targets['gender']

        loss_age = F.cross_entropy(age_pred, age_target)
        loss_height = F.cross_entropy(height_pred, height_target)
        loss_weight = F.cross_entropy(weight_pred, weight_target)
        loss_gender = F.cross_entropy(gender_pred, gender_target)

        # Combine losses for attributes
        loss = loss_age + loss_height + loss_weight + loss_gender

    return loss

   

##################################################


def train(model, train_loader, optimizer, device,config):
    
    output_type = config['output_type']
    model.train()
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), {k: v.to(device) for k, v in labels.items()}
        optimizer.zero_grad()
        predictions = model(features)

        if output_type == 'softmax':
            loss = F.cross_entropy(predictions, labels['person_id'])
        elif output_type == 'attribute':
            loss = combined_loss(predictions, labels,config)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss= total_loss / len(train_loader)
    return train_loss


##############################################
def validate(model, valid_loader, device,config):
    output_type = config['output_type']
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in valid_loader:
            features, labels = features.to(device), {k: v.to(device) for k, v in labels.items()}
            predictions = model(features)

            if output_type == 'softmax':
                loss = F.cross_entropy(predictions, labels['person_id'])
            elif output_type == 'attribute':
                loss = combined_loss(predictions, labels,config)

            total_loss += loss.item()
        val_loss =  total_loss / len(valid_loader)
    return val_loss

#################################################


from sklearn.metrics import  accuracy_score, precision_recall_fscore_support


from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def test(model, test_loader, device, config):
    try:
        output_type = config['output_type']
        print(f"Testing model with output type: {output_type}")
        model.eval()
        metrics = {}

        if output_type == 'softmax':
            person_id_preds, person_id_targets = [], []
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels['person_id'].to(device)
                    predictions = model(features)
                    person_id_preds.extend(predictions.argmax(dim=1).tolist())
                    person_id_targets.extend(labels.tolist())
            
            # Calculating metrics
            try:
                accuracy_person_id = accuracy_score(person_id_targets, person_id_preds)
                precision_person_id, recall_person_id, f1_person_id, _ = precision_recall_fscore_support(person_id_targets, person_id_preds, average='weighted')
                #cm_person_id = confusion_matrix(person_id_targets, person_id_preds)

                metrics = {
                    'accuracy_person_id': accuracy_person_id,
                    'precision_person_id': precision_person_id,
                    'recall_person_id': recall_person_id,
                    'f1_person_id': f1_person_id,
                    #'confusion_matrix_person_id': cm_person_id
                }
                """  if 'confusion_matrix_person_id' in metrics:
                    cm_person_id = metrics['confusion_matrix_person_id']

                    class_labels = [f'Class {i}' for i in range(cm_person_id.shape[0])]
                    #save_confusion_matrix(cm_person_id, class_labels, 'confusion_matrix.png')
                print("Successfully calculated softmax output metrics.")
                """
            except Exception as e:
                print(f"Error calculating metrics for softmax output: {e}") 

        elif output_type == 'attribute':
            # Initialize prediction and target lists for each attribute
            age_preds, age_targets = [], []
            height_preds, height_targets = [], []
            weight_preds, weight_targets = [], []
            gender_preds, gender_targets = [], []

            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), {k: v.to(device) for k, v in labels.items()}
                    try:
                        age_pred, height_pred, weight_pred, gender_pred = model(features)
                        age_targets.extend(labels['age'].tolist())
                        height_targets.extend(labels['height'].tolist())
                        weight_targets.extend(labels['weight'].tolist())
                        gender_targets.extend(labels['gender'].tolist())

                        age_preds.extend(age_pred.argmax(dim=1).tolist())
                        height_preds.extend(height_pred.argmax(dim=1).tolist())
                        weight_preds.extend(weight_pred.argmax(dim=1).tolist())
                        gender_preds.extend(gender_pred.argmax(dim=1).tolist())
                    except Exception as e:
                        print(f"Error processing batch in attributes output: {e}")

            # Calculating metrics for each attribute
            try:
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
                
                print("Successfully calculated attributes output metrics.")
            except Exception as e:
                print(f"Error calculating metrics for attributes output: {e}")

        else:
            print(f"Unsupported output type: {output_type}")

        return metrics

    except Exception as e:
        print(f"Unexpected error during test function execution: {e}")
        return {}

    

def save_confusion_matrix(cm, class_labels, filename):
    """
    Saves a confusion matrix as a PNG file using Seaborn's heatmap.

    Args:
    cm (ndarray): Confusion matrix to save.
    class_labels (list): List of class labels to use in the plot.
    filename (str): The filename to save the image to.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Model loaded from {self.model_path}")
now = datetime.datetime.now()
def configuration(dataset_idx,dataset_paths,output_idx, usage_mod_idx,learning_rates_idx, batch_size_idx,input_size_idx,gpudevice_idx,epochs):
    dataset = {0 : 'Unimib', 1 : 'SisFall', 2 : 'MobiAct' }
    num_classes = {'Unimib': 30, 'SisFall': 38, 'MobiAct': 67}  
    dataset_paths = {
        'Unimib': ("/data/malghaja/Bachelor_thesis/UniCat_train_data.csv",
                   "/data/malghaja/Bachelor_thesis/UniCat_valid_data.csv",
                   "/data/malghaja/Bachelor_thesis/UniCat_test_data.csv"),
        'SisFall': ("/data/malghaja/Bachelor_thesis/SisCat_train_data.csv",
                    "/data/malghaja/Bachelor_thesis/SisCat_valid_data.csv",
                    "/data/malghaja/Bachelor_thesis/SisCat_test_data.csv"),
        'MobiAct': ("/data/malghaja/Bachelor_thesis/MobiCat_train_data.csv",
                    "/data/malghaja/Bachelor_thesis/MobiCat_valid_data.csv",
                    #"/data/malghaja/Bachelor_thesis/MobiCat_test_data.csv"
                    "/data/malghaja/Bachelor_thesis/SisCat_test_data.csv")
    }
    folder_exp = 'data/malghaja/Bachelor_thesis/folder_exp'
    output = {0 : 'softmax', 1 : 'attribute'}
    learning_rate = [0.001,0.0001, 0.00001, 0.000001]
    batch_sizes = [50, 100 ,200] 
    input_size = [24,45]
    # gpudevice = [0,1,2]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpudevice_idx)
    GPU = 0
    usage_mod = { 0 : 'tarin', 1: 'train and test', 2 : 'test' }
    epochs = epochs
    train_path, valid_path, test_path = dataset_paths[dataset[dataset_idx]]
    config= {
        "dataset": dataset[dataset_idx],
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path": test_path,
        "folder_exp": folder_exp,
        "learning_rate": learning_rate[learning_rates_idx],
        "usage_mod" : usage_mod[usage_mod_idx],
        "input_size" : input_size[input_size_idx],
        #"gpudevice" : gpudevice[gpudevice_idx],
        'GPU' : GPU,
        "output_type": output[output_idx],
        "batch_size": batch_sizes[batch_size_idx],
        "epochs": epochs,
        'file_suffix': 'results_yy{}mm{}dd{:02d}.xml'.format(now.year,
                                                                                          now.month,
                                                                                          now.day,
                                                                                          now.hour,
                                                                                          now.minute),

        'hidden_size' : 128,
        'num_classes' : num_classes[dataset[dataset_idx]]
                                                                                  
        #"input_size": input_size,
        #"num_classes": num_classes[dataset[dataset_idx]],
        #"num_attributes": num_attributes,
        
    }
    return config     

import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

def save_results(config, metrics):
    """
    Save the results of training and testing in XML format, adjusted for the output type.
    """
    xml_file_path = config['dataset']+ str(config['learning_rate'])+ str(config['batch_size'])+ config['file_suffix']

    xml_root = ET.Element("Experiment")
    child_network = ET.SubElement(xml_root, "network", name="CNN-LSTM")
    child_dataset = ET.SubElement(child_network, "dataset", name=str(config['dataset']))

    # Add more elements based on your configuration...
    child = ET.SubElement(child_dataset, "learning_rate", value=str(config['learning_rate']))
    child = ET.SubElement(child_dataset, "epochs", value=str(config['epochs']))

    # Adding metrics based on output type
    if config['output_type'] == 'softmax':
        ET.SubElement(child_dataset, "person_id_metrics",
                      accuracy=str(metrics['accuracy_person_id']),
                      precision=str(metrics['precision_person_id']),
                      recall=str(metrics['recall_person_id']),
                      f1_score=str(metrics['f1_person_id']))
    elif config['output_type'] == 'attribute':
        # Age metrics
        ET.SubElement(child_dataset, "age_metrics",
                      accuracy=str(metrics['accuracy_age']),
                      precision=str(metrics['precision_age']),
                      recall=str(metrics['recall_age']),
                      f1_score=str(metrics['f1_age']))
        # Height metrics
        ET.SubElement(child_dataset, "height_metrics",
                      accuracy=str(metrics['accuracy_height']),
                      precision=str(metrics['precision_height']),
                      recall=str(metrics['recall_height']),
                      f1_score=str(metrics['f1_height']))
        # Weight metrics
        ET.SubElement(child_dataset, "weight_metrics",
                      accuracy=str(metrics['accuracy_weight']),
                      precision=str(metrics['precision_weight']),
                      recall=str(metrics['recall_weight']),
                      f1_score=str(metrics['f1_weight']))
        # Gender metrics
        ET.SubElement(child_dataset, "gender_metrics",
                      accuracy=str(metrics['accuracy_gender']),
                      precision=str(metrics['precision_gender']),
                      recall=str(metrics['recall_gender']),
                      f1_score=str(metrics['f1_gender']))

    xmlstr = minidom.parseString(ET.tostring(xml_root)).toprettyxml(indent="   ")
    with open(xml_file_path, "a") as f:
        f.write(xmlstr)

    print("Results saved to XML:", xmlstr)


""" def setup_experiment_logger(logging_level=logging.DEBUG, filename=None):
    
    # set up the logging
    logging_format = '[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s'
    if filename != None:
        logging.basicConfig(filename=filename,level=logging.DEBUG,
                            format=logging_format,
                            filemode='w')
    else:
        logging.basicConfig(level=logging_level,
                            format=logging_format,
                            filemode='w')
        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)   


    return """

import logging
import os

def setup_experiment_logger(logging_level=logging.DEBUG, log_dir='logs', experiment_name='experiment'):
    """
    Set up a custom logger for each experiment.

    Parameters:
    - logging_level: The logging level for the logger.
    - log_dir: The directory where log files will be stored.
    - experiment_name: A unique name for the experiment, which will be used to name the log file.
    """
    
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique logger for the experiment
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging_level)

    # Prevent adding multiple handlers to the logger if function is called multiple times
    if not logger.handlers:
        # Create file handler
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('[%(asctime)-19s, %(name)s, %(levelname)s] %(message)s')
        file_handler.setFormatter(file_format)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console_handler.setFormatter(console_format)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger, log_file


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
        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
def plot_learning_curve(train_losses, val_losses, title='Learning Curve'):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.show()
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_network(configuration,logger):
    #print(configuration)
    # Initialize datasets and data loaders
    train_dataset = IMUDataset(configuration["train_path"])
    valid_dataset = IMUDataset(configuration["valid_path"])
    test_dataset = IMUDataset(configuration["test_path"])
    train_loader = DataLoader(train_dataset, batch_size=configuration["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=configuration["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configuration["batch_size"], shuffle=False)

    # Initialize model and optimizer
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = configuration['gpudevice']
    if torch.cuda.is_available() and configuration['GPU'] >= 0:
        device = torch.device(f'cuda:{configuration["GPU"]}')
        print(f'Using GPU: {configuration["GPU"]}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    print(device)
    model = CNNLSTM(configuration["input_size"], configuration["hidden_size"], configuration["num_classes"],configuration).to(device)
    print(f"Total trainable parameters: {count_parameters(model)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration["learning_rate"])
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    logger.info(f"Dataset: {configuration['dataset']}, Learning Rate: {configuration['learning_rate']}, Batch Size: {configuration['batch_size']}, Model: {model}")
    #print(train_dataset.features)
    def execute_training():
        print(f'start training')
        start_time = time.time()
        train_losses, val_losses = [], []
        for epoch in range(configuration["epochs"]):
            print(f'Training Epoch {epoch+1}/{configuration["epochs"]}')
            train_loss = train(model, train_loader, optimizer, device,configuration)
            val_loss = validate(model, valid_loader, device,configuration)
            print(f'Epoch {epoch+1}/{configuration["epochs"]}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            trainng_time  = (time.time() - start_time)/60
            print(f'Epoch {epoch+1} trainng time {trainng_time}')
            logger.info(f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        plot_learning_curve(train_losses, val_losses)
        model_save_path = f"CNN-LSTM_{configuration['dataset']}_lr{configuration['learning_rate']}_bs{configuration['batch_size']}_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    def execute_testing():
        model_load_path= f"CNN-LSTM_{configuration['dataset']}_lr{configuration['learning_rate']}_bs{configuration['batch_size']}_model.pth"
        if configuration["usage_mod"] == 'test':
            model.load_state_dict(torch.load(model_load_path))
            model.eval()

        test_metrics = test(model, test_loader, device,configuration)
        logger.info(f"Test Results for {configuration['dataset']} with LR: {configuration['learning_rate']}, Batch Size: {configuration['batch_size']}: {test_metrics}")

        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")
        save_results(configuration, test_metrics)
    # Execution based on usage_mod
    if configuration["usage_mod"] in ['train', 'train and test']:
        execute_training()

    if configuration["usage_mod"] in ['test', 'train and test']:
        execute_testing()
 

def uniMib_main():
    """
    Run experiment for UniMib dataset with predefined parameters.
    """

    config = configuration(dataset_idx=0, dataset_paths = 'Unimib',output_idx=1, 
                           gpudevice_idx=0,usage_mod_idx= 1 , learning_rates_idx=0,batch_size_idx=1 ,input_size_idx= 0,
                            epochs=15)
    #print(config)
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #log_filename = f"{config['folder_exp']}logger_{timestamp}.txt"
    import os

    dir_name = os.path.dirname(log_filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    #setup_experiment_logger(logging_level=logging.DEBUG, filename=log_filename)
    #setup_experiment_logger(logging_level=logging.DEBUG, filename=config['folder_exp'] + "logger.txt")
    experiment_logger, log_filename  = setup_experiment_logger(experiment_name='Unimib_identification_experiment')    
    experiment_logger.info('Finished UniMib experiment setup')

    run_network(config)

    return

def sisFall_main():
    """
    Run experiment for SisFall dataset with predefined parameters.
    """

    config = configuration(dataset_idx=1, dataset_paths = 'SisFall',output_idx=1, 
                           usage_mod_idx= 1 , learning_rates_idx=1,batch_size_idx=1 ,input_size_idx= 1,
                            gpudevice_idx= 1,epochs=15)
    #print(config)
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #log_filename = f"{config['folder_exp']}logger_{timestamp}.txt"
    import os

    # dir_name = os.path.dirname(log_filename)
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)

    experiment_logger, log_filename = setup_experiment_logger(experiment_name='SisFall_identification_experiment')   
    experiment_logger.info('Finished UniMib experiment setup')
    # setup_experiment_logger(logging_level=logging.DEBUG, filename=log_filename)
    # #setup_experiment_logger(logging_level=logging.DEBUG, filename=config['folder_exp'] + "logger.txt")
    # logging.info('Finished SisFall experiment setup')

    run_network(config,experiment_logger)

    return
def mobiact_main():
    
    config = configuration(dataset_idx=2, dataset_paths = 'MobiAct',output_idx=0, 
                           usage_mod_idx= 2 , learning_rates_idx=1,batch_size_idx=1 ,input_size_idx= 1,
                            gpudevice_idx= 1,epochs=15)
     
    experiment_logger, log_filename = setup_experiment_logger(experiment_name='Mobiact_identification_on_sisFall_testdata')   
    experiment_logger.info('Finished Mobiact experiment setup')

    run_network(config,experiment_logger)

    return
if __name__ == "__main__":

    #main()
    uniMib_main()

    #sisFall_main()
    #mobiact_main()