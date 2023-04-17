import os
import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.decoders.unet.model import SegmentationHead
import segmentation_models_pytorch as smp


# Load the pre-trained models
def load_trained_models(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_architecture, encoder_name = model_name.split('_')

    if model_architecture == "UNET":
        model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    elif model_architecture == "DeepLabV3":
        model = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    elif model_architecture == "UNETplus":
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    else:
        raise ValueError("Invalid model architecture")

    model.load_state_dict(torch.load(f"./trained_models/{model_name}.pth",map_location=device))
    model = model.to(device)
    return model


# App
st.title("Image Segmentation")
st.write("Upload an image and select a pre-trained model for segmentation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

model_names = ["resnet34", "resnet101", "vgg16", "vgg19", "efficientnet-b0", "efficientnet-b7"]
model_nets = ["DeepLabV3", "UNET", "UNETplus"]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Select a pre-trained model:")
    model_name = st.selectbox("", [f"{net}_{encoder}" for net in model_nets for encoder in model_names])

    if st.button("Segment Image"):
        model = load_trained_models(model_name)

        data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Process the image
        input_image = data_transform(image)
        input_image = input_image.unsqueeze(0)

        # Perform segmentation
        with torch.no_grad():
            model.eval()
            preds = torch.sigmoid(model(input_image))
            threshold = 0.1
            preds = (preds > threshold).float()

        # Display the output image
        output_image = preds.squeeze().cpu().numpy()
        st.image(output_image, caption="Segmented Image", use_column_width=True)
