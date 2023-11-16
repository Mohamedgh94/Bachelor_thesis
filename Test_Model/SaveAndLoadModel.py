import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from MultiTaskLossFunction import MultiTaskLossFunction 
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SaveAndLoadModel:
    def __init__(self, model, loss_fn, learning_rate,optimizer_class, epochs, model_path="model.pth", device=None):
        self.model = model
        self.loss_fn = MultiTaskLossFunction() # This should be an instance of MultiTaskLossFunction
        self.learning_rate = learning_rate
        self.optimizer = optimizer_class(self.model.parameters())
        self.epochs = epochs
        self.model_path = model_path
        self.device = device or (torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")

    
   
    """   
    def train(self, train_loader, valid_loader, epochs=10, patience=3):
        multi_task_loss_fn = MultiTaskLossFunction() 
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
        best_val_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = {task: 0 for task in ['person_id', 'gender']}
            total_predictions = {task: 0 for task in ['person_id', 'gender']}
            regression_outputs = {task: [] for task in ['age', 'height', 'weight']}
            regression_labels = {task: [] for task in ['age', 'height', 'weight']}
            for batch_idx, (inputs, labels_dict) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels_dict = {task: labels.to(self.device) for task, labels in labels_dict.items()}

                optimizer.zero_grad()
                outputs_dict = self.model(inputs)

                loss = 0
                for task, output in outputs_dict.items():
                    if task == 'gated':
                        continue
                    labels = labels_dict[task]
                    if task in ['age', 'height', 'weight']:
                        loss += multi_task_loss_fn.loss_fns[task](output.squeeze(), labels.float())
                        regression_outputs[task].append(outputs_dict[task].detach().cpu().numpy())
                        regression_labels[task].append(labels_dict[task].detach().cpu().numpy())
                    elif task in ['person_id', 'gender']:
                        loss += multi_task_loss_fn.loss_fns[task](output, labels.long())
                        _, predicted = torch.max(output, 1)
                        correct_predictions[task] += (predicted == labels).sum().item()
                        total_predictions[task] += labels.size(0)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Print batch loss and accuracy
                if (batch_idx + 1) % 10 == 0:  # Print every 10 batches.
                    print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')

            # Average loss and accuracy for the epoch
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Avg Loss: {avg_loss}')
            for task in correct_predictions:
                if total_predictions[task] > 0:
                    accuracy = correct_predictions[task] / total_predictions[task]
                    print(f'Epoch {epoch+1}, {task} Training Accuracy: {accuracy:.4f}')

            for task in ['age', 'height', 'weight']:
                if regression_outputs[task]:
                    true_labels = np.concatenate(regression_labels[task])
                    pred_outputs = np.concatenate(regression_outputs[task])
                    mse = mean_squared_error(true_labels, pred_outputs)
                    print(f'Epoch {epoch+1}, {task} MSE: {mse}')

            # Validation after each epoch
            val_loss = self.validate(valid_loader)
            scheduler.step(val_loss)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f'Model saved to {self.model_path} (new best validation loss).')
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs.')
                    break
        print(f'Training completed.')        
        return accuracy        
        """ 
    def train(self, train_loader, valid_loader, epochs=10, patience=3):
        multi_task_loss_fn = MultiTaskLossFunction() 
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
        best_val_loss = float('inf')
        no_improvement_count = 0

        # Initialize lists to store losses for monitoring
        train_losses = []
        valid_losses = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            # ... (rest of your training loop code) ...
            epoch_loss = 0
            correct_predictions = {task: 0 for task in ['person_id', 'gender']}
            total_predictions = {task: 0 for task in ['person_id', 'gender']}
            regression_outputs = {task: [] for task in ['age', 'height', 'weight']}
            regression_labels = {task: [] for task in ['age', 'height', 'weight']}
            for batch_idx, (inputs, labels_dict) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels_dict = {task: labels.to(self.device) for task, labels in labels_dict.items()}

                optimizer.zero_grad()
                outputs_dict = self.model(inputs)

                loss = 0
                for task, output in outputs_dict.items():
                    if task == 'gated':
                        continue
                    labels = labels_dict[task]
                    if task in ['age', 'height', 'weight']:
                        loss += multi_task_loss_fn.loss_fns[task](output.squeeze(), labels.float())
                        regression_outputs[task].append(outputs_dict[task].detach().cpu().numpy())
                        regression_labels[task].append(labels_dict[task].detach().cpu().numpy())
                    elif task in ['person_id', 'gender']:
                        loss += multi_task_loss_fn.loss_fns[task](output, labels.long())
                        _, predicted = torch.max(output, 1)
                        correct_predictions[task] += (predicted == labels).sum().item()
                        total_predictions[task] += labels.size(0)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Print batch loss and accuracy
                if (batch_idx + 1) % 10 == 0:  # Print every 10 batches.
                    print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')


            # Calculate and store the average training loss for the epoch
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

        # Validation after each epoch
        val_loss = self.validate(valid_loader)
        valid_losses.append(val_loss)  # Store the validation loss for monitoring
        scheduler.step(val_loss)

        # Print out the losses to monitor them
        print(f'Epoch {epoch+1}/{epochs} \t Training Loss: {avg_train_loss} \t Validation Loss: {val_loss}')

        # Check for early stopping and overfitting
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(self.model.state_dict(), self.model_path)
            print(f'Model saved to {self.model_path} (new best validation loss).')
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                

        # Check for signs of overfitting
        if train_losses[-1] < valid_losses[-1]:
            print("Warning: Potential overfitting detected. Training loss is lower than validation loss.")

        print(f'Training completed.')
        return best_val_loss  # It might be better to return best_val_loss instead of accuracy


        
    

    def validate(self, valid_loader):
        multi_task_loss_fn = MultiTaskLossFunction() 
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels_dict in valid_loader:
                inputs = inputs.to(self.device)
                labels_dict = {task: labels.to(self.device) for task, labels in labels_dict.items()}
                outputs_dict = self.model(inputs)
                for task, output in outputs_dict.items():
                    if task == 'gated':
                        continue
                    labels = labels_dict[task]
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
                    predicted = output.argmax(dim=1) if task in ['person_id', 'gender'] else output.squeeze()
                    labels = labels_dict[task]
                    all_outputs.setdefault(task, []).extend(predicted.cpu().numpy())
                    all_labels.setdefault(task, []).extend(labels.cpu().numpy())

        # Calculate metrics for each task
        metrics = {}
        for task in all_labels:
            true = np.array(all_labels[task]).reshape(-1)
            pred = np.array(all_outputs[task]).reshape(-1)
            if task in ['person_id', 'gender']:
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
                
            else:
                # Metrics for regression tasks
                mse = mean_squared_error(true, pred)
                rmse = mean_squared_error(true, pred, squared=False)
                mae = mean_absolute_error(true, pred)
                r2 = r2_score(true, pred)
                
                metrics[task] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
        
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

    