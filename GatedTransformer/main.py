import logging
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from IMUDataset import IMUDataset
from saveAndLoad import SaveAndLoadModel
from loss import MultiTaskLossFunction 
from GatedTransformer import GatedTransformer
import pandas as pd
import os
from datetime import datetime
now = datetime.now()
# New configuration function

        

 
def configuration(dataset_idx,dataset_paths,output_idx, usage_mod_idx,learning_rates_idx, batch_size_idx,input_size_idx,gpudevice_idx,epochs):
    dataset = {0 : 'Unimib', 1 : 'SisFall', 2 : 'MobiAct' }
    num_classes = {'Unimib': 30, 'SisFall': 38, 'MobiAct': 67}  
    dataset_paths = {
        'Unimib': ("/data/malghaja/Bachelor_thesis/UniMib/UniAtt_train_data.csv",
                    "/data/malghaja/Bachelor_thesis/UniMib/UniAtt_valid_data.csv",
                    "/data/malghaja/Bachelor_thesis/UniMib/UniAtt_test_data.csv"),
        # 'Unimib' : ("/Users/mohamadghajar/Documents/BAC/Bachelor_thesis/test_data.csv",
        #             "/Users/mohamadghajar/Documents/BAC/Bachelor_thesis/test_data.csv",
        #             "/Users/mohamadghajar/Documents/BAC/Bachelor_thesis/test_data.csv"),
        'SisFall': ("/data/malghaja/Bachelor_thesis/SisFall/SisAtt_train_data.csv",
                    "/data/malghaja/Bachelor_thesis/SisFall/SisAtt_valid_data.csv",
                    "/data/malghaja/Bachelor_thesis/SisFall/SisAtt_test_data.csv"),
        'MobiAct': ("/data/malghaja/Bachelor_thesis/MobiAct/MobiAtt_train_data.csv",
                    "/data/malghaja/Bachelor_thesis/MobiAct/Mobiatt_valid_data.csv",
                    #"/data/malghaja/Bachelor_thesis/SisFall/SisAtt_test_data.csv"
                    "/data/malghaja/Bachelor_thesis/MobiAct/MobiAtt_test_data.csv"
                    
                    )
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

        'hidden_size' : 256,
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
    child_network = ET.SubElement(xml_root, "network", name="GTN")
    child_dataset = ET.SubElement(child_network, "dataset", name=str(config['dataset']))

    # Add more elements based on your configuration...
    child = ET.SubElement(child_dataset, "learning_rate", value=str(config['learning_rate']))
    child = ET.SubElement(child_dataset, "Batchsize", value=str(config['batch_size']))
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

class Main:
    def __init__(self, model, loss_fn, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.config = config

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        self.model_saver = SaveAndLoadModel(self.model,optimizer_class=optim.AdamW, epochs=self.config['epochs'], config = config,model_path= f"GTN_{config['dataset']}_lr{config['learning_rate']}_bs{config['batch_size']}_model.pth" , device=self.device)
        # self.train_loader = DataLoader(IMUDataset(self.config['train_path']), batch_size=self.config['batch_size'], shuffle=True)
        # self.valid_loader = DataLoader(IMUDataset(self.config['valid_path']), batch_size=self.config['batch_size'], shuffle=False)
        # self.test_loader = DataLoader(IMUDataset(self.config['test_path']), batch_size=self.config['batch_size'], shuffle=False)
        
        print(f'Total number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    """ def run(self):
        print("Running on device:", self.device)
        if self.config['usage_mod'] == 'train':
            self.model_saver.train(self.train_loader)
        elif self.config['usage_mod'] == 'train and test':
            self.model_saver.train_and_validate(self.train_loader, self.valid_loader)
            self.model_saver.test(self.test_loader)
        elif self.config['usage_mod'] == 'test':
            self.model_saver.load_model()
            self.model_saver.test(self.test_loader)
        else:
            print("Invalid configuration for usage mode.") """
    
    def run_network(self,configuration,logger):
        train_dataset = IMUDataset(configuration["train_path"])
        valid_dataset = IMUDataset(configuration["valid_path"])
        test_dataset = IMUDataset(configuration["test_path"])
        train_loader = DataLoader(train_dataset, batch_size=configuration["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=configuration["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=configuration["batch_size"], shuffle=False)

        # Initialize the model
        model = GatedTransformer(input_dim=configuration["input_size"],
                                d_model=512,
                                num_heads=16,
                                d_ff=1024,
                                num_layers=6,
                                config=configuration,  
                                num_classes=configuration['num_classes'],
                                dropout_rate=0.1)  

        loss_fn = MultiTaskLossFunction(configuration)

        logger.info(f"Dataset: {configuration['dataset']}, Learning Rate: {configuration['learning_rate']}, Batch Size: {configuration['batch_size']}, Model: {model}")

        def execute_training():
            print(f'start training')
            start_time = time.time()
            train_losses, val_losses = [], []
            for epoch in range(configuration["epochs"]):
                print(f'Training Epoch {epoch+1}/{configuration["epochs"]}')
                # Assuming `train` and `validate` functions are defined elsewhere and accessible here
                train_loss = self.model_saver.train (train_loader)
                val_loss = self.model_saver.validate(valid_loader)
                print(f'Epoch {epoch+1}/{configuration["epochs"]}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                training_time = (time.time() - start_time) / 60
                print(f'Epoch {epoch+1} training time {training_time} minutes')
                logger.info(f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

            model_save_path = f"GTN_{configuration['dataset']}_lr{configuration['learning_rate']}_bs{configuration['batch_size']}_model.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        def execute_testing():
            model_load_path = f"GTN_{configuration['dataset']}_lr{configuration['learning_rate']}_bs{configuration['batch_size']}_model.pth"
            if configuration["usage_mod"] == 'test':
                model.load_state_dict(torch.load(model_load_path))
                model.eval()

            # Assuming `test` function is defined elsewhere and accessible here
            test_metrics = self.model_saver.test(test_loader)
            logger.info(f"Test Results for {configuration['dataset']} with LR: {configuration['learning_rate']}, Batch Size: {configuration['batch_size']}: {test_metrics}")

            print("Test Metrics:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value}")
            save_results(configuration, test_metrics)    
        if configuration["usage_mod"] in ['train', 'train and test']:
            execute_training()

        if configuration["usage_mod"] in ['test', 'train and test']:
            execute_testing()
            
def uniMib_main():
    config = configuration(dataset_idx=0, dataset_paths = 'Unimib',output_idx=1, 
                        gpudevice_idx=1,usage_mod_idx= 1 , learning_rates_idx=1,batch_size_idx=1 ,input_size_idx= 0,
                            epochs=5)
    experiment_logger, log_filename  = setup_experiment_logger(experiment_name='GTN_Unimib_IDs')    
    experiment_logger.info('Finished UniMib experiment setup')
    #model = GatedTransformer(input_dim=config["input_size"],d_model=512, num_heads=16, d_ff=1024, num_layers=6,config = config,num_classes=config['num_classes'] ,dropout_rate=0.1)
    model = GatedTransformer(input_dim=config["input_size"],d_model=128, num_heads=4, d_ff=256, num_layers=6,config = config,num_classes=config['num_classes'] ,dropout_rate=0.1)

    loss_fn = MultiTaskLossFunction(config)
    controller = Main(model, loss_fn, config)
    controller.run_network(config,experiment_logger)
    

def sisFall_main():
    config = configuration(dataset_idx=1, dataset_paths = 'SisFall',output_idx=1, 
                        gpudevice_idx=0,usage_mod_idx= 1 , learning_rates_idx=1,batch_size_idx=1 ,input_size_idx= 1,
                            epochs=5)
    experiment_logger, log_filename  = setup_experiment_logger(experiment_name='GTN_SisFall_identification2')    
    experiment_logger.info('Finished SisFall experiment setup')
    #model = GatedTransformer(input_dim=config["input_size"],d_model=512, num_heads=16, d_ff=1024, num_layers=6,config = config,num_classes=config['num_classes'] ,dropout_rate=0.1)
    model = GatedTransformer(input_dim=config["input_size"],d_model=64, num_heads=4, d_ff=256, num_layers=4,config = config,num_classes=config['num_classes'] ,dropout_rate=0.3)

    loss_fn = MultiTaskLossFunction(config)
    controller = Main(model, loss_fn, config)
    controller.run_network(config,experiment_logger)

def mobiact_main():
    config = configuration(dataset_idx=2, dataset_paths = 'MobiAct',output_idx=0, 
                        gpudevice_idx=2,usage_mod_idx= 1 , learning_rates_idx=1,batch_size_idx=2 ,input_size_idx= 1,
                            epochs=5)
    experiment_logger, log_filename  = setup_experiment_logger(experiment_name='GTN_MobiAct_identification')    
    experiment_logger.info('Finished MobiAct experiment setup')
    #model = GatedTransformer(input_dim=config["input_size"],d_model=512, num_heads=8, d_ff=1024, num_layers=4,config = config,num_classes=config['num_classes'] ,dropout_rate=0.3)
    #model = GatedTransformer(input_dim=config["input_size"],d_model=256, num_heads=8, d_ff=512, num_layers=4,config = config,num_classes=config['num_classes'] ,dropout_rate=0.1)
    model = GatedTransformer(input_dim=config["input_size"],d_model=128, num_heads=4, d_ff=256, num_layers=6,config = config,num_classes=config['num_classes'] ,dropout_rate=0.2)
    loss_fn = MultiTaskLossFunction(config)
    controller = Main(model, loss_fn, config)
    controller.run_network(config,experiment_logger)    
   
if __name__ == "__main__":
    
    uniMib_main()
    #sisFall_main()
    #mobiact_main()




""" import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from IMUDataset import IMUDataset,num_person_ids , num_ages,num_heights,num_weights,num_genders
from saveAndLoad import SaveAndLoadModel
from loss import MultiTaskLossFunction 
from GatedTransformer import GatedTransformer
from torchsummary import summary
import pandas as pd

class Main:
    def __init__(self, model, loss_fn):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.get_file_paths()
        self.get_hyperparameters()
        #summary(self.model, input_size=(self.batch_size, 45)) 
        
        print(f'Total number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

       
        

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model_saver = SaveAndLoadModel(self.model,optimizer_class= torch.optim.Adam,epochs=self.epochs ,model_path= self.model_path, device=self.device)
        #self.model_saver = SaveAndLoadModel(self.model,ptimizer_class=torch.optim.Adam,epochs=self.epochs,model_path=self.model_path,device=self.device,)
        self.train_loader = DataLoader(IMUDataset(self.train_csv), batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(IMUDataset(self.valid_csv), batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(IMUDataset(self.test_csv), batch_size=self.batch_size, shuffle=False)

        
    def get_file_paths(self):
        self.train_csv = input("Enter the path to the training CSV file: ")
        self.valid_csv = input("Enter the path to the validation CSV file: ")
        self.test_csv = input("Enter the path to the test CSV file: ")
        self.model_path = input("Enter the path where you want to save/load the model: ")
    def get_hyperparameters(self):
        self.learning_rate = float(input("Enter the learning rate: "))
        self.batch_size = int(input("Enter the batch size: "))
        self.epochs = int(input("Enter the number of epochs: "))

    def run(self):
        print("Running on device:", self.device)
        print("What would you like to do?")
        print("1: Train and Validate")
        print("2: Train, Validate, and Test")
        print("3: Test using a saved pre-trained model")
        
        choice = input("Enter the number corresponding to your choice: ")
        if choice == '0':
            self.model_saver.train(self.train_loader)
        if choice == '1':
            epochs = int(input("Enter the number of epochs for training: "))
            print("Debug:", type(epochs), epochs)
            self.model_saver.train_and_validate(self.train_loader, self.valid_loader)
            
        elif choice == '2':
            epochs = int(input("Enter the number of epochs for training: "))
            self.model_saver.train_validate_and_test(self.train_loader, self.valid_loader, self.test_loader )
            self.model_saver.test(self.test_loader)
            
        elif choice == '3':
            self.model_saver.load_model()
            self.model_saver.test(self.test_loader)
            
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    input_dim= 15 
    d_model = 512      # Size of embeddings and model dimensionality
    num_heads = 16      # Number of attention heads
    d_ff = 2048        # Dimensionality of feed-forward layer
    num_layers = 6     # Number of layers in the encoder
    
    model = GatedTransformer(input_dim, d_model, num_heads, d_ff, num_layers)
    
    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = MultiTaskLossFunction()
    print("CUDA available:", torch.cuda.is_available())

    # Initialize the MainController
    controller = Main(model, loss_fn)
    
    # Run the MainController
    controller.run()
 """