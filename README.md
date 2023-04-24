# Leaf Image Segmentation

> #### Shuai | Spring '23 | Duke AIPI 540

## Project Description

This project aims to enable leaf image segmentation, which would allow farmers or bio scientists to monitor withered
areas using a camera. By automatically alerting the farmer or bio scientist via email, this technology could
significantly improve the efficiency of plant monitoring. The project involves the comparison of various models, with
the aim of identifying the best model for achieving accurate leaf image segmentation.

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
|-- K8s
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
|   |-- ml-5.ipynb
|   |-- ml.ipynb
|   `-- nondeep.ipynb
|-- script
|   |-- Evaluator.py
|   |-- SegmentationDataset.py
|   `-- Trainer.py
|-- trained_models            ----best model will be saved here
|-- Dockerfile
|-- README.md
|-- demo.py                   ----web demo script
|-- main.py                   ----main function
`-- requirements.txt
```

## Requirements

See `requirements.txt`
> pip3 install -r requirements.txt

### Run the code

> python3 main.py

### Run the web demo

> streamlit run demo.py

open streamlit run demo.py

## Architecture Diagram

![model architecture diagram](https://user-images.githubusercontent.com/16725501/233882811-1ed2155a-2d4a-4996-8a41-6fe1f1e0e9c2.png)

## Segmentation Model

### Model Overview

This model employed various architectures, including DeepLabV3, UNET, and UNETplus, to segment leaf images.
Additionally, it utilized different encoders such as efficientnet-b7, efficientnet-b0, resnet34, resnet101, vgg16,
vgg19. The model's performance was assessed
using metrics such as IoU, Dice, F1-score, and accuracy. Finally, the most effective model was selected based on the
evaluation results.

### Training and Evaluation

- Use BCEWithLogitsLoss loss function and Adam optimizer.
- Train dataset : Validation dataset = 8:2
- Number of Epochs = 20
- Model with best performance saved for final image segmentation

## Results

![image](https://user-images.githubusercontent.com/16725501/233882989-08bdf9d0-27db-4d0e-adda-a66fd430a3ac.png)
This document provides a recommendation on the performance of a model based on its MAP@k (Mean Average Precision at k)
scores. The MAP@k scores were computed for k = 1, 3, 5, and 10, and the results are presented below:

* MAP@1: 0.1321
* MAP@3: 0.0865
* MAP@5: 0.0682
* MAP@10: 0.0485

The MAP@k is a popular evaluation metric for ranking models, and it measures the average precision at each cutoff k. In
this case, the model's performance decreases as the cutoff k increases, indicating that it is better at identifying the
top-ranked items than the lower-ranked ones.

The MAP@1 score of 0.1321 suggests that the model performs reasonably well in identifying the top-ranked item, but there
is still room for improvement. The MAP@3 score of 0.0865 indicates that the model's performance drops significantly
beyond the first item, suggesting that it may not be as effective in identifying the top three items. The MAP@5 score of
0.0682 and the MAP@10 score of 0.0485 indicate that the model's performance further decreases as more items are
considered.

## Application

This is a Streamlit application that serves as a web-based demonstration for the leaf image segmentation model. Users
have the option to select a pre-trained model and upload a leaf image or use their camera to capture an image for
segmentation. Additionally, users can subscribe to the application to receive automatic alerts when the leaf is
withered, improving the efficiency of plant monitoring.