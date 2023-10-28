import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
class SaveAndLoadModel:
    def __init__(self, model, loss_fn, optimizer_class, epochs,model_path="model.pth"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class(self.model.parameters())
        self.epochs = epochs
        self.model_path = model_path

      
    def train(self, train_loader, epochs = 10):
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  
        self.model.train()
        total_loss = 0
        corect = 0
        total = 0
        print(type(self.epochs))
        
        for epoch in range(epochs):
            for batch in train_loader:
                inputs, labels = batch
                #print(f'inputs.shape, labels.shape',inputs.shape, labels.shape)
                labels = torch.argmax(labels, dim=1)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                _, predicted = torch.max(outputs, dim=1)
                #print(f'predicted.shape, labels.shape',predicted.shape, labels.shape)  
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
                total += labels.size(0)
                corect += (predicted == labels).sum().item()
        train_accuracy = 100 * corect / total        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss}, train_accuracy: {train_accuracy}")
        return avg_loss, train_accuracy

    def validate(self, valid_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch
                labels = torch.argmax(labels, dim=1)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                corect += (predicted == labels).sum().item()
        valid_accuracy = 100 * corect / total
        avg_loss = total_loss / len(valid_loader)
        return avg_loss, valid_accuracy

    def test(self, test_loader):
        self.model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                all_outputs.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_outputs)
        f1 = f1_score(all_labels, all_outputs, average="weighted")   
        precision = precision_score(all_labels, all_outputs, average="weighted")
        recall = recall_score(all_labels, all_outputs, average="weighted")
        metrics = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
        return metrics

    def train_and_validate(self, train_loader, valid_loader, epochs):
        for epoch in range(epochs):
            train_loss , train_accuracy = self.train(train_loader)
            valid_loss , valid_accuracy = self.validate(valid_loader)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, train_accuracy: {train_accuracy}, Valid Loss: {valid_loss} , Valid Accuracy: {valid_accuracy} ")

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def train_validate_and_test(self, train_loader, valid_loader, test_loader, epochs):
        self.train_and_validate(train_loader, valid_loader, epochs)
        metrics = self.test(test_loader)
        
        # Save metrics to Excel
        df = pd.DataFrame([metrics], columns=metrics.keys())
        df.to_excel("results.xlsx", index=False)
        print("Results saved to results.xlsx")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Model loaded from {self.model_path}")