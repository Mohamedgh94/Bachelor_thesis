import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

class SaveAndLoadModel:
    def __init__(self, model, loss_fn, optimizer, epochs,model_path="model.pth"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.model_path = model_path

      
    def train(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  
        self.model.train()
        total_loss = 0
        for epoch in range(self.epochs):
            for batch in train_loader:
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, valid_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    def test(self, test_loader):
        self.model.eval()
        # Your testing logic here, maybe store metrics in a dictionary
        metrics = {"Accuracy": 0.9, "F1": 0.8, "Precision": 0.85, "Recall": 0.9}  # Example metrics
        return metrics

    def train_and_validate(self, train_loader, valid_loader, epochs=5):
        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            valid_loss = self.validate(valid_loader)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def train_validate_and_test(self, train_loader, valid_loader, test_loader, epochs=5):
        self.train_and_validate(train_loader, valid_loader, epochs)
        metrics = self.test(test_loader)
        
        # Save metrics to Excel
        df = pd.DataFrame([metrics], columns=metrics.keys())
        df.to_excel("results.xlsx", index=False)
        print("Results saved to results.xlsx")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Model loaded from {self.model_path}")
