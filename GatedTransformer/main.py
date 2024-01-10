import torch
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
