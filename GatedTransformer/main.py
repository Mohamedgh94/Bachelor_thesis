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
        'Unimib': ("/Users/mohamadghajar/Documents/BAC/Bachelor_thesis/test_data.csv",
                   "/Users/mohamadghajar/Documents/BAC/Bachelor_thesis/test_data.csv",
                   "/Users/mohamadghajar/Documents/BAC/Bachelor_thesis/test_data.csv"),
        'SisFall': ("/data/malghaja/Bachelor_thesis/SisFall/SisCat_train_data.csv",
                    "/data/malghaja/Bachelor_thesis/SisFall/SisCat_valid_data.csv",
                    "/data/malghaja/Bachelor_thesis/SisFall/SisCat_test_data.csv"),
        'MobiAct': ("/data/malghaja/Bachelor_thesis/MobiAct/MobiCat_train_data.csv",
                    "/data/malghaja/Bachelor_thesis/MobiAct/MobiCat_valid_data.csv",
                    "/data/malghaja/Bachelor_thesis/MobiAct/MobiCat_test_data.csv"
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
class Main:
    def __init__(self, model, loss_fn, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.config = config

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.model_saver = SaveAndLoadModel(self.model, optimizer_class=optim.Adam, epochs=self.config['epochs'], model_path=self.config['folder_exp'], device=self.device)
        self.train_loader = DataLoader(IMUDataset(self.config['train_path']), batch_size=self.config['batch_size'], shuffle=True)
        self.valid_loader = DataLoader(IMUDataset(self.config['valid_path']), batch_size=self.config['batch_size'], shuffle=False)
        self.test_loader = DataLoader(IMUDataset(self.config['test_path']), batch_size=self.config['batch_size'], shuffle=False)

        print(f'Total number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    def run(self):
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
            print("Invalid configuration for usage mode.")

if __name__ == "__main__":
    config = configuration(dataset_idx=0, dataset_paths = 'Unimib',output_idx=0, 
                           gpudevice_idx=0,usage_mod_idx= 1 , learning_rates_idx=0,batch_size_idx=1 ,input_size_idx= 0,
                            epochs=15)
    
    model = GatedTransformer(input_dim=24, d_model=512, num_heads=16, d_ff=1024, num_layers=6)
    loss_fn = MultiTaskLossFunction()
    controller = Main(model, loss_fn, config)
    controller.run()




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