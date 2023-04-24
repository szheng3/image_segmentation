import os
import warnings

from PIL import Image
from torch.utils.data import Dataset

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initialize the SegmentationDataset object.

        Parameters:
        image_dir (str): Path to the directory containing input images.
        mask_dir (str): Path to the directory containing segmentation masks.
        transform (callable, optional): Optional transform to be applied on both the images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get a list of image filenames from the image directory
        self.images = os.listdir(image_dir)
        # Get a list of mask filenames from the mask directory
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
        int: The total number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve the image and corresponding mask at the given index.

        Parameters:
        idx (int): Index of the image and mask pair to retrieve.

        Returns:
        tuple: A tuple containing the image and corresponding mask.
        """
        # Construct the file paths for the image and mask using the index
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        # Open and convert the image and mask files
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Apply the transform on the image and mask if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Return the image and mask pair
        return image, mask
