import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time

class SaveAndLoadModel:
    def __init__(self, model, loss_fn, optimizer_class, epochs,model_path="model.pth"):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class(self.model.parameters())
        self.epochs = epochs
        self.model_path = model_path

      
    def train(self, train_loader, epochs = 10):
        trainings_start_time = time.time()
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  
        self.model.train()
        total_loss = 0
        corect = 0
        total = 0
        print(type(self.epochs))
        
        
        print(f'batch type','train loader type',type(train_loader))
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
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
        trainigend_time = time.time()
        print(f'training time',trainigend_time-trainings_start_time)
        return avg_loss, train_accuracy

    def validate(self, valid_loader):
        validation_start_time = time.time()
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = torch.argmax(labels, dim=1)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                corect += (predicted == labels).sum().item()
        valid_accuracy = 100 * corect / total
        avg_loss = total_loss / len(valid_loader)
        validationend_time = time.time()
        print(f'validation time',validationend_time-validation_start_time)
        return avg_loss, valid_accuracy

    def test(self, test_loader):
        teststart_time = time.time()
        self.model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                all_outputs.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_outputs)
        f1 = f1_score(all_labels, all_outputs, average="weighted")   
        precision = precision_score(all_labels, all_outputs, average="weighted")
        recall = recall_score(all_labels, all_outputs, average="weighted")
        metrics = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
        testend_time = time.time()
        print(f'test time',testend_time-teststart_time)
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
