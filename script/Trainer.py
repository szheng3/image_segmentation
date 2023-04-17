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

from script.SegmentationDataset import SegmentationDataset

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, train_loader, val_loader, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_model(self, model, num_epochs, criterion, optimizer, model_save_path):
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            model.train()
            for images, masks in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for images, masks in self.val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                val_loss /= len(self.val_loader)
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}")

                if val_loss < best_val_loss:
                    print(f"Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f}, saving model...")
                    torch.save(model.state_dict(), model_save_path)
                    best_val_loss = val_loss

        model.load_state_dict(torch.load(model_save_path))
        return model

    def train_models(self, encoder_names, model_nets, num_epochs):
        trained_models = {}
        for encoder_name in encoder_names:
            for model_net in model_nets:
                print(f"Training {model_net} model with encoder {encoder_name}...")
                if model_net == "DeepLabV3" and "vgg" not in encoder_name:
                    model = smp.DeepLabV3(
                        encoder_name=encoder_name,
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=1
                    )
                elif model_net == "UNET":
                    model = smp.Unet(
                        encoder_name=encoder_name,
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=1
                    )
                elif model_net == "UNETplus":
                    model = smp.UnetPlusPlus(
                        encoder_name=encoder_name,
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=1
                    )
                else:
                    continue

                for param in model.encoder.parameters():
                    param.requires_grad = False

                model = model.to(self.device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                model_save_path = f"{model_net}_{encoder_name}.pth"
                trained_model = self.train_model(model, num_epochs, criterion, optimizer, model_save_path)
                trained_models[model_net + encoder_name] = trained_model

        return trained_models


