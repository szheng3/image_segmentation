import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

class Evaluator:
    def __init__(self, trained_models, val_loader, threshold,device):
        self.trained_models = trained_models
        self.val_loader = val_loader
        self.threshold = threshold
        self.device = device
    def iou_score(self, y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / union

    def dice_coefficient(self, y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum()
        return (2 * intersection) / (y_true.sum() + y_pred.sum())

    def compute_scores(self, y_true, y_pred):
        y_true = (y_true > self.threshold).astype(np.uint8).flatten()
        y_pred = (y_pred > self.threshold).astype(np.uint8).flatten()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        iou = self.iou_score(y_true, y_pred)
        dice = self.dice_coefficient(y_true, y_pred)

        return accuracy, precision, recall, f1, iou, dice

    def evaluate_models(self):
        evaluation_results = {}

        for model_name, model in self.trained_models.items():
            model.eval()
            accuracies, precisions, recalls, f1_scores, ious, dice_coeffs = [], [], [], [], [], []

            with torch.no_grad():
                for images, masks in self.val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    preds = torch.sigmoid(model(images))

                    preds = (preds > self.threshold).float()
                    masks = (masks > self.threshold).float()

                    accuracy, precision, recall, f1, iou, dice = self.compute_scores(masks.cpu().numpy(),
                                                                                     preds.cpu().numpy())

                    accuracies.append(accuracy)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                    ious.append(iou)
                    dice_coeffs.append(dice)

            evaluation_results[model_name] = {
                "Accuracy": np.mean(accuracies),
                "Precision": np.mean(precisions),
                "Recall": np.mean(recalls),
                "F1-score": np.mean(f1_scores),
                "IoU": np.mean(ious),
                "Dice": np.mean(dice_coeffs)
            }

        return evaluation_results
