import os

import numpy as np
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class ImageClassifier:
    def __init__(self):
        # Initialize the Random Forest Classifier model
        # n_estimators: Number of trees in the forest
        # random_state: Seed used by the random number generator
        # n_jobs: Number of CPU cores used for parallelism (-1 means using all processors)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    @staticmethod
    def flatten_image(img):
        # Flatten the input image by reshaping it into a 2D array
        # img: A 3D numpy array representing an image
        return img.reshape(-1, img.shape[-1])

    @staticmethod
    def flatten_mask(mask):
        # Flatten the input mask by reshaping it into a 1D array
        # mask: A 2D numpy array representing an image mask
        return mask.reshape(-1)

    def prepare_data(self, image_dir, mask_dir, image_files):
        # Prepare the data by reading images and masks, then flattening them
        # image_dir: Path to the directory containing the input images
        # mask_dir: Path to the directory containing the input masks
        # image_files: List of filenames for the input images
        X = []
        y = []
        for img_file in image_files:
            if img_file.endswith('jpg'):
                img_path = os.path.join(image_dir, img_file)
                mask_path = os.path.join(mask_dir, img_file.replace('.jpg', '.png'))

                # Read the image and mask files
                img = io.imread(img_path)
                mask = io.imread(mask_path)

                # Flatten the image and mask, then append them to X and y
                X.append(self.flatten_image(img))
                y.append(self.flatten_mask(mask))

        # Stack the flattened images and masks into single 2D and 1D arrays, respectively
        X = np.vstack(X)
        y = np.hstack(y)

        return X, y

    def train(self, X_train, y_train):
        # Train the Random Forest model using the provided training data
        # X_train: A 2D numpy array containing the flattened training images
        # y_train: A 1D numpy array containing the flattened training masks
        self.model.fit(X_train, y_train)

    def evaluate(self, X, y):
        # Evaluate the model by calculating its accuracy score for the given data
        # X: A 2D numpy array containing the flattened images
        # y: A 1D numpy array containing the flattened masks
        return self.model.score(X, y)


def main():
    # Read the list of image files from the input directory
    image_files = os.listdir("../data/train_images")

    # Initialize the ImageClassifier object
    classifier = ImageClassifier()

    # Prepare the data using the ImageClassifier object
    X, y = classifier.prepare_data("../data/train_images", "../data/train_masks", image_files)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model using the ImageClassifier object
    classifier.train(X_train, y_train)

    # Evaluate the model using the training and validation data
    train_accuracy = classifier.evaluate(X_train, y_train)
    val_accuracy = classifier.evaluate(X_val, y_val)

    # Print the training and validation accuracy scores
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()