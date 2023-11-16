""" import torch
import torch.nn as nn
import torch.nn.functional as F
from IMUDataset import num_person_ids,num_ages,num_heights,num_weights,num_genders
class MultiTaskLossFunction:
    def __init__(self):
        self.loss_fns = {
            'person_id': nn.CrossEntropyLoss(),
            'age': nn.MSELoss(),
            'height': nn.MSELoss(),
            'weight': nn.MSELoss(),
            'gender': nn.CrossEntropyLoss(),
        }

    def compute_loss(self, outputs_dict, labels_dict):
        total_loss = 0
        for task, output in outputs_dict.items():
            # Skip the 'gated' output if it does not have corresponding labels
            if task == 'gated':  
                continue
            
            labels = labels_dict[task].float()  # Ensure labels are float if it's regression
            loss_fn = self.loss_fns[task]  # Access the correct loss function

            if task in ['age', 'height', 'weight']:
                # Ensure output is also float for regression tasks
                output = output.float()  
                # Remove unnecessary squeeze, or make sure it's not changing the tensor's rank incorrectly
                # output = output.squeeze(-1)
            else:
                labels = labels.long()    
            # Compute the loss using the correct function
            loss = loss_fn(output, labels)
            total_loss += loss
        return total_loss

 """
import torch
import torch.nn as nn
import torch.nn.functional as F
from IMUDataset import num_person_ids,num_ages,num_heights,num_weights,num_genders
class MultiTaskLossFunction:
    def __init__(self):
        self.loss_fns = {
            'person_id': nn.CrossEntropyLoss(),
            'age': nn.MSELoss(),
            'height': nn.MSELoss(),
            'weight': nn.MSELoss(),
            'gender': nn.CrossEntropyLoss(),
        }
    
    def compute_loss(self, outputs_dict, labels_dict):
       
        total_loss = 0
        for task, output in outputs_dict.items():
            # Skip the 'gated' output if it does not have corresponding labels
            if task == 'gated':  
                continue
            labels = labels_dict[task]
            loss_fn = self.loss_fns[task]
            
            if task in ['age', 'height', 'weight']:
                #loss_fn = nn.MSELoss()
                #labels = labels.float().view(-1, 1)
                output = output.squeeze(-1)
            loss = loss_fn(output, labels)
            total_loss += loss
        return total_loss