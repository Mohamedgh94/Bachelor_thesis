""" import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

class SaveAndLoadModel:
    def __init__(self, model, loss_fn, optimizer_class, epochs,model_path="model.pth",  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class(self.model.parameters())
        self.epochs = epochs
        self.model_path = model_path
        self.device = device
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")

      
    def train(self, train_loader, epochs = 10):
        trainings_start_time = time.time()
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  
        self.model.train()
        total_loss = 0
        corect = 0
        total = 0
        
        
        
        print(f'train loader type',type(train_loader))
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
        corect = 0 
        total = 0
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

    def test(self, test_loader,debug= False):
        teststart_time = time.time()
        self.model.eval()
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if debug:
                    
                    print("Batch Input Shape:", inputs.shape)
                    print("Batch Labels Shape:", labels.shape)
                outputs = self.model(inputs)
                if debug:
                    
                    print("Batch Output Shape:", outputs.shape)
                predicted = outputs.argmax(dim=1)
                flat_labels = labels.argmax(dim=1)
                if debug:
                    print("predicted shape:", predicted.shape)
                    print("flat_labels shape:", flat_labels.shape)
                    #flat_predicted = predicted.view(-1)
                    #print("Before: ", len(all_outputs), len(all_labels))
                all_outputs.extend(predicted.cpu().numpy())
                all_labels.extend(flat_labels.cpu().numpy())
                if debug:
                    print("After: ", len(all_outputs), len(all_labels))
            
        all_labels = np.array(all_labels).reshape(-1)
        all_outputs = np.array(all_outputs).reshape(-1)
        all_labels = all_labels.astype(int)
        all_outputs = all_outputs.astype(int)
        if debug:
            
            print("all_outputs shape:", np.array(all_outputs).shape, " type:", type(all_outputs))
            print("all_labels shape:", np.array(all_labels).shape, " type:", type(all_labels))   
        accuracy = accuracy_score(all_labels, all_outputs)
        f1 = f1_score(all_labels, all_outputs, average="weighted")   
        precision = precision_score(all_labels, all_outputs, average="weighted")
        recall = recall_score(all_labels, all_outputs, average="weighted")
        metrics = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
        print(f"Test Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        testend_time = time.time()
        print(f'test time',testend_time-teststart_time)
        return metrics, all_labels, all_outputs

    def train_and_validate(self, train_loader, valid_loader, epochs):
        for epoch in range(epochs):
            train_loss , train_accuracy = self.train(train_loader)
            valid_loss , valid_accuracy = self.validate(valid_loader)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, train_accuracy: {train_accuracy}, Valid Loss: {valid_loss} , Valid Accuracy: {valid_accuracy} ")

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def train_validate_and_test(self, train_loader, valid_loader, test_loader, epochs):
        self.train_and_validate(train_loader, valid_loader, epochs)
        metrics, y_true, y_pred = self.test(test_loader)
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in range(len(cm))], columns=[f'Predicted {i}' for i in range(len(cm))])
        with pd.ExcelWriter("results.xlsx") as writer:
            df_metrics = pd.DataFrame([metrics], columns=metrics.keys())
            df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            # Save the confusion matrix
            cm_df.to_excel(writer, sheet_name='Confusion Matrix')
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.savefig('confusion_matrix.png')
            plt.close()
            
            print("Results and confusion matrix saved to results.xlsx")
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Model loaded from {self.model_path}")
 """

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from loss import MultiTaskLossFunction 

class SaveAndLoadModel:
    def __init__(self, model, optimizer_class, epochs, model_path="model.pth", device=None):
        self.model = model
        self.loss_fn = MultiTaskLossFunction()
        self.optimizer = optimizer_class(self.model.parameters())
        self.epochs = epochs
        self.model_path = model_path
        self.device = device or (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")


    
   
    """
    def train(self, train_loader, epochs=10):
        multi_task_loss_fn = MultiTaskLossFunction() 
        trainings_start_time = time.time()
        #for name, param in self.model.named_parameters():
        #    print(f"{name} requires_grad: {param.requires_grad}")
        self.model.train()
        total_loss = 0
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = {task: 0 for task in ['person_id', 'gender']}
            total_predictions = {task: 0 for task in ['person_id', 'gender']}
            for batch  in train_loader:
                inputs, labels_dict = batch
                inputs = inputs.to(self.device)
                labels_dict = {task: labels.to(self.device) for task, labels in labels_dict.items()}
                self.optimizer.zero_grad()
                outputs_dict = self.model(inputs)
                for task, labels in labels_dict.items():
                    if task in ['age', 'height', 'weight']:

                        labels_dict[task] = labels.float()
                    elif task in ['person_id', 'gender']:
                        labels_dict[task] = labels.long()    
                loss = multi_task_loss_fn.compute_loss(outputs_dict, labels_dict)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
        training_end_time = time.time()
        print(f'Training completed in {training_end_time - trainings_start_time}s')
     """
    
    def train(self, train_loader, epochs=10):
        multi_task_loss_fn = MultiTaskLossFunction()  # Ensure this is initialized correctly for classification tasks
        trainings_start_time = time.time()
        
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = {task: 0 for task in ['age', 'height', 'weight', 'gender']}
            total_predictions = {task: 0 for task in ['age', 'height', 'weight', 'gender']}

        for batch in train_loader:
            inputs, labels_dict = batch
            inputs = inputs.to(self.device)
            labels_dict = {task: labels.to(self.device) for task, labels in labels_dict.items()}

            self.optimizer.zero_grad()
            outputs_dict = self.model(inputs)

            loss = 0
            for task, output in outputs_dict.items():
                labels = labels_dict[task].long()  # Ensure labels are long type for classification
                loss += multi_task_loss_fn.loss_fns[task](output, labels)
                
                # Calculate the number of correct predictions
                _, predicted = torch.max(output, 1)
                correct_predictions[task] += (predicted == labels).sum().item()
                total_predictions[task] += labels.size(0)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")

        # Calculate and print the accuracy for each classification task
        for task in correct_predictions:
            accuracy = correct_predictions[task] / total_predictions[task]
            print(f"Epoch {epoch+1}, {task} Training Accuracy: {accuracy:.4f}")

        training_end_time = time.time()
        print(f'Training completed in {training_end_time - trainings_start_time}s')

    def validate(self, valid_loader):
        multi_task_loss_fn = MultiTaskLossFunction() 
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels_dict in valid_loader:
                inputs = inputs.to(self.device)
                labels_dict = {task: labels.to(self.device) for task, labels in labels_dict.items()}
                outputs_dict = self.model(inputs)
                loss =multi_task_loss_fn.compute_loss(outputs_dict, labels_dict)
                total_loss += loss.item()

        avg_loss = total_loss / len(valid_loader)
        print(f"Validation Avg Loss: {avg_loss}")
        return avg_loss

    def test(self, test_loader):
        self.model.eval()
        all_outputs = {}
        all_labels = {}
        with torch.no_grad():
            for inputs, labels_dict in test_loader:
                inputs = inputs.to(self.device)
                outputs_dict = self.model(inputs)
                for task, output in outputs_dict.items():
                    predicted = output.argmax(dim=1)
                    labels = labels_dict[task]
                    all_outputs.setdefault(task, []).extend(predicted.cpu().numpy())
                    all_labels.setdefault(task, []).extend(labels.cpu().numpy())

        # Calculate metrics for each classification task
        metrics = {}
        for task in all_labels:
            true = np.array(all_labels[task]).reshape(-1)
            pred = np.array(all_outputs[task]).reshape(-1)
            accuracy = accuracy_score(true, pred)
            f1 = f1_score(true, pred, average="weighted")
            precision = precision_score(true, pred, average="weighted")
            recall = recall_score(true, pred, average="weighted")
            cm = confusion_matrix(true, pred)
            metrics[task] = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "confusion_matrix": cm}
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
            plt.title(f"Confusion Matrix for {task}")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()    
            
        
        print(metrics)
        return metrics , all_labels, all_outputs
    def train_and_validate(self, train_loader, valid_loader):
        self.train(train_loader, self.epochs)
        self.validate(valid_loader)
        
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def train_validate_and_test(self, train_loader, valid_loader, test_loader):
        self.train_and_validate(train_loader, valid_loader)
        self.test(test_loader)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Model loaded from {self.model_path}")
