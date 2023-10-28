
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from IMUDataset import IMUDataset
from saveAndLoad import SaveAndLoadModel
from loss import LossFunction as loss
from GatedTransformer import GatedTransformer

class Main:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.get_file_paths()
        self.get_hyperparameters()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model_saver = SaveAndLoadModel(self.model,loss_fn=self.loss_fn ,optimizer_class= torch.optim.Adam,epochs=self.epochs ,model_path= self.model_path)
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
            self.model_saver.train_and_validate(self.train_loader, self.valid_loader, epochs)
            
        elif choice == '2':
            epochs = int(input("Enter the number of epochs for training: "))
            self.model_saver.train_validate_and_test(self.train_loader, self.valid_loader, self.test_loader ,epochs)
            self.model_saver.test(self.test_loader)
            
        elif choice == '3':
            self.model_saver.load_model()
            self.model_saver.test(self.test_loader)
            
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    input_dim=30  # Replace with your actual vocabulary size
    d_model = 512      # Size of embeddings and model dimensionality
    num_heads = 8      # Number of attention heads
    d_ff = 2048        # Dimensionality of feed-forward layer
    num_layers = 6     # Number of layers in the encoder

    model = GatedTransformer(input_dim, d_model, num_heads, d_ff, num_layers)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize the MainController
    controller = Main(model, loss_fn)

    # Run the MainController
    controller.run()
