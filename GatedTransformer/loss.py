""" import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction:
    def __init__(self, loss_type="CrossEntropy"):
        self.loss_type = loss_type
        self.loss_fn = self.get_loss_function()

    def get_loss_function(self):
        if self.loss_type == "CrossEntropy":
            return nn.CrossEntropyLoss()
        elif self.loss_type == "MSE":
            return nn.MSELoss()
        elif self.loss_type == "Custom":
            return self.custom_loss
        else:
            raise ValueError("Invalid loss type")

    def custom_loss(self, y_pred, y_true):
        loss_person = F.cross_entropy(y_pred[:, 0:1], y_true[:, 0:1])
        loss_age = F.mse_loss(y_pred[:, 1:2], y_true[:, 1:2])
        loss_height = F.mse_loss(y_pred[:, 2:3], y_true[:, 2:3])
        loss_weight = F.mse_loss(y_pred[:, 3:4], y_true[:, 3:4])
        loss_gender = F.cross_entropy(y_pred[:, 4:5], y_true[:, 4:5])

        total_loss = loss_person + loss_age + loss_height + loss_weight + loss_gender
        return total_loss

    def compute_loss(self, outputs, labels):
        return self.loss_fn(outputs, labels)
 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from IMUDataset import num_person_ids,num_ages,num_heights,num_weights,num_genders
class MultiTaskLossFunction:
    def __init__(self):
        self.loss_fns = {
            'age': nn.CrossEntropyLoss(),
            'height': nn.CrossEntropyLoss(),
            'weight': nn.CrossEntropyLoss(),
            'gender': nn.CrossEntropyLoss(),
        }
    
    def compute_loss(self, outputs_dict, labels_dict):
        total_loss = 0
        for task, output in outputs_dict.items():
            labels = labels_dict[task].long()  # Ensure labels are long type for CrossEntropyLoss
            loss = self.loss_fns[task](output, labels)
            total_loss += loss
        return total_loss