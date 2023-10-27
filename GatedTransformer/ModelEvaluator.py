from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


class ModelEvaluator:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def add_batch(self, y_true_batch, y_pred_batch):
        # Assuming y_true_batch and y_pred_batch are NumPy arrays
        # You can add conversion from tensor to NumPy array here if needed
        self.y_true.extend(y_true_batch)
        self.y_pred.extend(y_pred_batch)

    def evaluate(self):
        # Convert lists to NumPy arrays for evaluation
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')  # Using weighted average for multi-class/multi-label
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        # Reset for next evaluation
        self.y_true = []
        self.y_pred = []

        return accuracy, f1, precision, recall
