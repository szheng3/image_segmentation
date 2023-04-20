import os
import time
import cv2
import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import av
import requests
import smtplib
from email.message import EmailMessage
import streamlit as st
from camera_input_live import camera_input_live


# def send_email(subject, body, to):
#     msg = EmailMessage()
#     msg.set_content(body)
#     msg["Subject"] = subject
#     msg["From"] = "your-email@example.com"
#     msg["To"] = to
#
#     server = smtplib.SMTP_SSL("smtp.example.com", 465)
#     server.login("your-email@example.com", "your-email-password")
#     server.send_message(msg)
#     server.quit()


# Function to send email using API
def send_email_api(name, to_email, subject, body):
    url = "https://apiv1.sszzz.me/api/email/send"
    payload = {
        "name": name,
        "email": to_email,
        "subject": subject,
        "body": body
    }
    response = requests.post(url, json=payload)
    return response.status_code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained models
def load_trained_models(model_name):
    print(device)
    model_architecture, encoder_name = model_name.split('_')

    if model_architecture == "UNET":
        model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    elif model_architecture == "DeepLabV3":
        model = smp.DeepLabV3(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    elif model_architecture == "UNETplus":
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    else:
        raise ValueError("Invalid model architecture")

    model.load_state_dict(torch.load(f"./trained_models/{model_name}.pth", map_location=device))
    model = model.to(device)
    return model


def display_sent_email(uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        cols = st.columns(2)
        resized_image = image.resize((256, 256))
        cols[0].image(resized_image, caption="Uploaded Image", use_column_width=True)
        video_transformer = VideoTransformer(model)
        output_image = video_transformer.transform(image)
        cols[1].image(output_image, caption="Segmented Image", use_column_width=True)
        # Check if the segmented area is greater than the threshold percentage
        area_percentage = (np.count_nonzero(output_image) / (output_image.shape[0] * output_image.shape[1]*output_image.shape[2])) * 100

        if area_percentage > threshold_percentage:
            st.write(f"The predicted area percentage is {area_percentage:.2f}% which is greater than the threshold.")
            print(email)
            if email:
                # Send an email alert
                print("Sending email...")
                subject = "Image Segmentation Alert"
                body = f"The predicted area percentage is {area_percentage:.2f}% which is greater than the threshold of {threshold_percentage}%."
                try:
                    send_email_api("Leaf Image Segmentation", subject, body, email)
                    st.success("Alert email sent successfully.")
                except Exception as e:
                    st.error(f"Error sending email: {e}")
        else:
            st.write(f"The predicted area percentage is {area_percentage:.2f}% which is below the threshold.")


class VideoTransformer:
    def __init__(self, model):
        self.model = model
        self.data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def transform(self, image):
        input_image = self.data_transform(image).unsqueeze(0).to(device)

        # Perform segmentation
        with torch.no_grad():
            self.model.eval()
            preds = torch.sigmoid(self.model(input_image))
            threshold = 0.1
            preds = (preds > threshold).float()

        # Display the output image
        output_image = preds.squeeze().cpu().numpy() * 255
        segmented_frame = cv2.cvtColor(np.uint8(output_image), cv2.COLOR_GRAY2BGR)
        return segmented_frame

# App
st.set_page_config(page_title="Leaf Image Segmentation", layout="wide")
st.title("Leaf Image Segmentation")
st.write("Upload an image or use the camera to capture a photo and select a pre-trained model for segmentation.")

sidebar = st.sidebar
sidebar.title("Settings")
sidebar.write("Select a pre-trained model:")
model_names = ["resnet34", "resnet101", "vgg16", "vgg19", "efficientnet-b0", "efficientnet-b7"]
model_nets = ["DeepLabV3", "UNET", "UNETplus"]
model_name = sidebar.selectbox("", [f"{net}_{encoder}" for net in model_nets for encoder in model_names])
model = load_trained_models(model_name)

threshold_percentage = sidebar.slider("Alert Threshold Percentage", min_value=0, max_value=100, value=50, step=1)
sidebar.write("Email Subscription:")
email = sidebar.text_input("Enter your email", "")

nav = sidebar.radio("Navigation", ["Upload Image", "Use Camera"])

if nav == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        display_sent_email(uploaded_file)
else:
    st.write("Use the camera:")
    image = camera_input_live()
    display_sent_email(image)