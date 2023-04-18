import os
import time
import cv2
import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, VideoProcessorBase
import av



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

    model.load_state_dict(torch.load(f"./trained_models/{model_name}.pth", map_location=device))
    model = model.to(device)
    return model


class VideoTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model
        self.data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def transform(self, image):
        input_image = self.data_transform(image).unsqueeze(0)

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
st.set_page_config(page_title="Image Segmentation", layout="wide")
st.title("Image Segmentation")
st.write("Upload an image or use the camera to capture a photo and select a pre-trained model for segmentation.")

model_names = ["resnet34", "resnet101", "vgg16", "vgg19", "efficientnet-b0", "efficientnet-b7"]
model_nets = ["DeepLabV3", "UNET", "UNETplus"]

sidebar = st.sidebar
sidebar.write("Select a pre-trained model:")
model_name = sidebar.selectbox("", [f"{net}_{encoder}" for net in model_nets for encoder in model_names])
model = load_trained_models(model_name)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    cols = st.columns(2)
    cols[0].image(image, caption="Uploaded Image", use_column_width=True)
    video_transformer = VideoTransformer(model)
    output_image = video_transformer.transform(image)
    cols[1].image(output_image, caption="Segmented Image", use_column_width=True)
else:
    st.write("Or use the camera:")

# class CaptureEverySecond(VideoTransformerBase):
#     def __init__(self):
#         self.last_capture = time.time()
#
#     def transform(self, frame):
#         print("transform")
#         current_time = time.time()
#         if current_time - self.last_capture >= 1:
#             self.last_capture = current_time
#             # img = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
#             # image = Image.fromarray(img)
#             # cols = st.columns(2)
#             # cols[0].image(frame, caption="Captured Image", use_column_width=True)
#             # video_transformer = VideoTransformer(model)
#             # output_image = video_transformer.transform(image)
#             # cols[1].image(output_image, caption="Segmented Image", use_column_width=True)
#         return frame

# def video_frame_callback(frame):
#     print("video_frame_callback")
#     img = frame.to_ndarray(format="bgr24")
#
#     flipped = img[::-1,:,:]
#
#     return av.VideoFrame.from_ndarray(flipped, format="bgr24")
# webrtc_ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.style = 'color'


    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # image processing code here

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx=webrtc_streamer(key="vpf", video_processor_factory=VideoProcessor)
#
# if webrtc_ctx.video_transformer:
#     webrtc_ctx.video_transformer.model = model
