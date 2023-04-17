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

from script.Evaluator import Evaluator
from script.SegmentationDataset import SegmentationDataset
from script.Trainer import Trainer

warnings.filterwarnings("ignore")


def main():
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = SegmentationDataset("leaf_disease_images/train", "leaf_disease_masks/train",
                                        transform=data_transform)
    val_dataset = SegmentationDataset("leaf_disease_images/val", "leaf_disease_masks/val", transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(train_loader, val_loader, device)

    encoder_names = ["resnet34", "resnet50", "vgg16"]

    model_nets = ["DeepLabV3", "UNET", "UNETplus"]

    num_epochs = 10

    trained_models = trainer.train_models(encoder_names, model_nets, num_epochs)

    evaluator = Evaluator(trained_models, val_loader, threshold=0.5, device=device)
    evaluation_results = evaluator.evaluate_models()

    for model_name, metrics in evaluation_results.items():
        print(f"{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()


if __name__ == "__main__":
    main()
