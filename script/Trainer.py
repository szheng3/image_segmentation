import warnings

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, train_loader, val_loader, device):
        """
        Initialize the Trainer object.

        Parameters:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to move the models and data to.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_model(self, model, num_epochs, criterion, optimizer, model_save_path):
        """
        Train a single model and save the best weights based on validation loss.

        Parameters:
        model (nn.Module): The model to train.
        num_epochs (int): Number of epochs to train the model.
        criterion (nn.Module): Loss function used to train the model.
        optimizer (optim.Optimizer): Optimizer used to train the model.
        model_save_path (str): Path to save the best model weights.

        Returns:
        model (nn.Module): The trained model with the best weights.
        """
        # Initialize the best validation loss as infinity
        best_val_loss = float("inf")

        # Iterate through the epochs
        for epoch in range(num_epochs):
            # Set the model to training mode
            model.train()
            # Iterate through the training dataset
            for images, masks in self.train_loader:
                # Move images and masks to the specified device
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Reset the optimizer's gradients
                optimizer.zero_grad()
                # Perform a forward pass through the model
                outputs = model(images)
                # Calculate the loss
                loss = criterion(outputs, masks)
                # Perform backpropagation
                loss.backward()
                # Update the model's weights
                optimizer.step()

            # Set the model to evaluation mode
            model.eval()
            # Initialize the validation loss
            with torch.no_grad():
                val_loss = 0
                # Iterate through the validation dataset
                for images, masks in self.val_loader:
                    # Move images and masks to the specified device
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    # Perform a forward pass through the model
                    outputs = model(images)
                    # Calculate the loss
                    loss = criterion(outputs, masks)
                    # Accumulate the validation loss
                    val_loss += loss.item()

                # Calculate the average validation loss
                val_loss /= len(self.val_loader)
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss}")

                # If the validation loss has improved, save the model
                if val_loss < best_val_loss:
                    print(f"Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f}, saving model...")
                    torch.save(model.state_dict(), model_save_path)
                    best_val_loss = val_loss

        # Load the best model weights before returning the model
        model.load_state_dict(torch.load(model_save_path))
        return model

    def train_models(self, encoder_names, model_nets, num_epochs):
        """
        Train multiple models with different encoders and architectures.

        Parameters:
        encoder_names (list): List of encoder names to use in the models.
        model_nets (list): List of segmentation model architectures to train.
        num_epochs (int): Number of epochs to train the models.

        Returns:
        trained_models (dict): A dictionary containing the trained models with keys as a combination of the model architecture and encoder name.

        """
        trained_models = {}
        # Iterate through the encoder names
        for encoder_name in encoder_names:
            # Iterate through the model architectures
            for model_net in model_nets:
                print(f"Training {model_net} model with encoder {encoder_name}...")
                # Initialize the model based on the provided model architecture and encoder
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

                # Freeze the encoder's parameters
                for param in model.encoder.parameters():
                    param.requires_grad = False

                # Move the model to the specified device
                model = model.to(self.device)
                # Set the loss function and optimizer
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                # Set the path for saving the model weights
                model_save_path = f"{model_net}_{encoder_name}.pth"
                # Train the model and store it in the trained_models dictionary
                trained_model = self.train_model(model, num_epochs, criterion, optimizer, model_save_path)
                trained_models[model_net + encoder_name] = trained_model

        return trained_models
