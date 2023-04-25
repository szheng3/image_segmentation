[![publish to Dockerhub](https://github.com/szheng3/leaf_image_segmentation/actions/workflows/publish.yml/badge.svg)](https://github.com/szheng3/leaf_image_segmentation/actions/workflows/publish.yml)

# Leaf Image Segmentation

> #### Shuai | Spring '23 | Duke AIPI 540

## Project Description

This project aims to enable leaf image segmentation, which would allow farmers or bio scientists to monitor withered
areas using a camera. By automatically alerting the farmer or bio scientist via email, this technology could
significantly improve the efficiency of plant monitoring. The project involves the comparison of various models, with
the aim of identifying the best model for achieving accurate leaf image segmentation.

## Project Demo with GPU enable

* [https://api.cloud.sszzz.me/](https://api.cloud.sszzz.me/)

![image](https://user-images.githubusercontent.com/16725501/233883932-4715b03d-fde3-451e-b4ad-6327f930db02.png)

## Data Source

This project includes a dataset consisting of 2500 leaf images and their corresponding masks for training purposes.
Additionally, there are 440 leaf images and masks provided for validation purposes, which will allow for testing the
performance of the developed models.
The data could be downloaded from the following link:
https://www.kaggle.com/datasets/sovitrath/leaf-disease-segmentation-with-trainvalid-split

## Pretrained Model

Pretrained model could be downloaded from the following
link: https://szdataset.s3.us-east-2.amazonaws.com/trained_models.zip

## Project Structure

```
|-- .devcontainer               ----codespaces configuration 
|-- .github                     ----github CI/CD actions      
|-- K8s                         ----kubernetes deployment files
|   |-- ingress.yaml
|   `-- pythonpod.yaml
|-- data                        ----store automatically fetched data
|   |-- train_images
|   |   `-- 00000_2.jpg
|   |-- train_masks
|   |   `-- 00000_2.png
|   |-- valid_images
|   |   `-- 00000_3.jpg
|   `-- valid_masks
|       `-- 00000_3.png
|-- notebooks
|   |-- ml.ipynb
|   `-- nondeep.ipynb
|-- script
|   |-- Evaluator.py            ----evaluation class
|   |-- SegmentationDataset.py  ----dataset class
|   |-- NonDeepLearning.py      ----Non Deep Learning models (RandomForest) in order to compared with deep learning model
|   `-- Trainer.py              ----trainer class
|-- trained_models              ----training model will be saved here
|   `-- UNETplus_efficientnet-b0.pth
|-- Dockerfile
|-- README.md
|-- demo.py                   ----web demo script
|-- main.py                   ----main function
`-- requirements.txt
```

## Kubernetes Deployment

* go to the directory `K8s`

```
cd K8s
```

* create the namespace `resume-prod`

```
kubectl create namespace resume-prod

```

* apply the yaml files

```
kubectl apply -f .
```

* Google Kubernetes Engine (GKE) in GPU mode is used in this project for deployment. The following is the screenshot of
  the deployment.

![image](https://user-images.githubusercontent.com/16725501/233886025-38abb94c-cbc5-4696-be34-396955bb14b0.png)

## Docker

* This repo main branch is automatically published to Dockerhub
  with [CI/CD](https://github.com/szheng3/leaf_image_segmentation/actions/runs/4781787422/jobs/8500549277), you can pull
  the image from [here](https://hub.docker.com/repository/docker/szheng3/sz-leaf-ml/general)

```
docker pull szheng3/sz-leaf-ml:latest
```

* Run the docker image with GPU.

```
docker run -d -p 8501:8501 szheng3/sz-leaf-ml:latest
```

## CI/CD

Github Actions configured in .github/workflows

## Requirements

See `requirements.txt`
> pip3 install -r requirements.txt

### Run the code

> python3 main.py

### Run the web demo

> streamlit run demo.py

open streamlit run demo.py

## Architecture Diagram

![model architecture diagram](https://user-images.githubusercontent.com/16725501/234078333-32a71887-ae9b-4f1a-8752-51a6ee4caeae.png)

## Segmentation Model

### Model Overview

This model employed various architectures, including DeepLabV3, UNET, and UNETplus, to segment leaf images.
Additionally, it utilized different encoders such as efficientnet-b7, efficientnet-b0, resnet34, resnet101, vgg16,
vgg19. The model's performance was assessed
using metrics such as IoU, Dice, F1-score, and accuracy. Finally, the most effective model was selected based on the
evaluation results.

### UNETplus vs UNET
![model architecture diagram](https://user-images.githubusercontent.com/16725501/233882811-1ed2155a-2d4a-4996-8a41-6fe1f1e0e9c2.png)

### DeepLabV3

![model architecture diagram](https://user-images.githubusercontent.com/16725501/233891246-12f80dcf-8b2b-4a68-9647-6d2cd450e3ad.png)

### Training and Evaluation

- train the model with Duke DCC on-demand GPU.
![image](https://user-images.githubusercontent.com/16725501/233892384-fa0ca1da-352f-4bef-a697-97888f2fd790.png)
- Use BCEWithLogitsLoss loss function and Adam optimizer.
- Train dataset : Validation dataset = 2500:440
- Number of Epochs = 20
- Model with best performance saved for final image segmentation

## Results

![image](https://user-images.githubusercontent.com/16725501/233882989-08bdf9d0-27db-4d0e-adda-a66fd430a3ac.png)
The best model used for leaf segmentation is UNETplus with the EfficientNet-B7 encoder. The evaluation metrics for the
best model
are as follows:

Accuracy: This metric represents the percentage of correctly predicted pixels in the segmentation mask, and in this
case, the model achieved an accuracy of 0.973.

Precision: Precision measures the percentage of true positive pixels out of all the positive predictions made by the
model. A high precision score indicates that the model makes fewer false positive predictions, and in this case, the
precision score is 0.929.

Recall: Recall is the percentage of true positive pixels that were correctly predicted out of all the ground truth
positive pixels. A high recall score indicates that the model has a lower tendency to miss positive pixels, and in this
case, the recall score is 0.896.

F1-score: The F1-score is the harmonic mean of precision and recall, and it is a good indicator of the overall
performance of the model. In this case, the F1-score is 0.912.

IoU (Intersection over Union): IoU is the ratio of the intersection between the predicted and ground truth masks to
their union. A high IoU score indicates that the model accurately predicts the object boundaries, and in this case, the
IoU score is 0.839.

Dice coefficient: The Dice coefficient is similar to IoU and measures the overlap between the predicted and ground truth
masks. In this case, the Dice coefficient is 0.912, which indicates that the model accurately predicts the object
boundaries and produces a high-quality segmentation mask.

## Non-Deep Learning Model

Although non-deep learning models such as random forests can achieve decent performance, their scalability can be
limited when dealing with large image datasets. In this case, training on eight images and validating on two, the model
achieved a high training IoU of 0.9116 and a validation IoU of 0.7790, which is a good result. However, as the
image size grows larger, the model's performance may suffer due to its reliance on CPU instead of GPU, causing the Duke
DCC on-demand platform to freeze under heavy CPU usage.

## Application

This is a Streamlit application that serves as a web-based demonstration for the leaf image segmentation model. Users
have the option to select a pre-trained model and upload a leaf image or use their camera to capture an image for
segmentation. Additionally, users can subscribe to the application to receive automatic alerts when the leaf is
withered, improving the efficiency of plant monitoring.