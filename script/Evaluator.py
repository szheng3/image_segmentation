# Import necessary libraries and modules
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluator:
    def __init__(self, trained_models, val_loader, threshold, device):
        """
        Initialize the Evaluator object.

        Parameters:
        trained_models (dict): A dictionary containing the trained models with their names as keys.
        val_loader (DataLoader): A PyTorch DataLoader object containing the validation dataset.
        threshold (float): A threshold value to convert predicted probabilities into binary values.
        device (str): The device to perform the computation (e.g., 'cpu', 'cuda').
        """
        self.trained_models = trained_models
        self.val_loader = val_loader
        self.threshold = threshold
        self.device = device

    def iou_score(self, y_true, y_pred):
        """
        Calculate the Intersection over Union (IoU) score.

        Parameters:
        y_true (array-like): Ground truth binary segmentation mask.
        y_pred (array-like): Predicted binary segmentation mask.

        Returns:
        float: The IoU score.
        """
        # Compute the intersection of the ground truth and predicted masks
        intersection = np.logical_and(y_true, y_pred).sum()
        # Compute the union of the ground truth and predicted masks
        union = np.logical_or(y_true, y_pred).sum()
        # Compute the IoU score by dividing the intersection by the union
        return intersection / union

    def dice_coefficient(self, y_true, y_pred):
        """
        Calculate the Dice Coefficient.

        Parameters:
        y_true (array-like): Ground truth binary segmentation mask.
        y_pred (array-like): Predicted binary segmentation mask.

        Returns:
        float: The Dice Coefficient.
        """
        # Compute the intersection of the ground truth and predicted masks
        intersection = np.logical_and(y_true, y_pred).sum()
        # Compute the Dice Coefficient by using the formula: (2 * intersection) / (sum(y_true) + sum(y_pred))
        return (2 * intersection) / (y_true.sum() + y_pred.sum())

    def compute_scores(self, y_true, y_pred):
        """
        Calculate various evaluation metrics for the given ground truth and predicted masks.

        Parameters:
        y_true (array-like): Ground truth binary segmentation mask.
        y_pred (array-like): Predicted binary segmentation mask.

        Returns:
        tuple: A tuple containing accuracy, precision, recall, F1-score, IoU, and Dice Coefficient.
        """
        # Convert the ground truth and predicted masks to binary values using the threshold
        y_true = (y_true > self.threshold).astype(np.uint8).flatten()
        y_pred = (y_pred > self.threshold).astype(np.uint8).flatten()

        # Calculate the evaluation metrics for the ground truth and predicted masks
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        iou = self.iou_score(y_true, y_pred)
        dice = self.dice_coefficient(y_true, y_pred)

        return accuracy, precision, recall, f1, iou, dice


    def evaluate_models(self):
        """
        Evaluate all the trained models on the validation dataset and compute evaluation metrics.

        Returns:
        dict: A dictionary containing the evaluation metrics for each trained model.
        """
        # Create an empty dictionary to store evaluation results
        evaluation_results = {}

        # Iterate over each trained model
        for model_name, model in self.trained_models.items():
            # Set the model to evaluation mode
            model.eval()
            # Create empty lists to store evaluation metrics for each model
            accuracies, precisions, recalls, f1_scores, ious, dice_coeffs = [], [], [], [], [], []

            # Disable gradient computation to save memory and speed up evaluation
            with torch.no_grad():
                # Iterate over the validation dataset
                for images, masks in self.val_loader:
                    # Move images and masks to the specified device
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    # Predict segmentation masks using the model
                    preds = torch.sigmoid(model(images))

                    # Convert the predicted probabilities to binary values using the threshold
                    preds = (preds > self.threshold).float()
                    masks = (masks > self.threshold).float()

                    # Calculate evaluation metrics for the current batch
                    accuracy, precision, recall, f1, iou, dice = self.compute_scores(masks.cpu().numpy(),
                                                                                     preds.cpu().numpy())

                    # Append the evaluation metrics to their respective lists
                    accuracies.append(accuracy)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                    ious.append(iou)
                    dice_coeffs.append(dice)
            # Calculate mean evaluation metrics for the current model
            evaluation_results[model_name] = {
                "Accuracy": np.mean(accuracies),
                "Precision": np.mean(precisions),
                "Recall": np.mean(recalls),
                "F1-score": np.mean(f1_scores),
                "IoU": np.mean(ious),
                "Dice": np.mean(dice_coeffs)
            }
        # Return the dictionary containing the evaluation metrics for all models
        return evaluation_results
