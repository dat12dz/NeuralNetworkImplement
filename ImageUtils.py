import os
import numpy as np
from PIL import Image

def load_handwritten_digits_dataset(folder_path, image_size=(32, 32), max_images=100):
    dataset = {}

    # Loop through each digit folder
    for digit in sorted(os.listdir(folder_path)):
        digit_path = os.path.join(folder_path, digit)

        if not os.path.isdir(digit_path) or not digit.isdigit():
            continue  # Skip non-digit or invalid folders

        images = []
        files = sorted(os.listdir(digit_path))
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for i, filename in enumerate(image_files[:max_images]):
            try:
                img_path = os.path.join(digit_path, filename)
                with Image.open(img_path).convert("L") as img:  # convert to grayscale
                    resized_img = img.resize(image_size)
                    img_array = np.array(resized_img)
                    images.append(img_array)
            except Exception as e:
                print(f"Failed to load {filename} in digit {digit}: {e}")

        dataset[int(digit)] = images

    return dataset


# Example usage:
# data = load_handwritten_digits_dataset("path/to/dataset")
# Access an image as a NumPy array: data[5][2]  # 3rd image of digit "5"
def flatten_dataset(dataset):
    """
    Flattens all 2D grayscale images in a nested dataset dict to 1D arrays.
    Input format: dataset[digit][index] -> 2D array
    Output format: dataset[digit][index] -> 1D array
    """
    flat_dataset = {}
    for digit, images in dataset.items():
        flat_dataset[digit] = [1.0 - (img.flatten() / 255.0) for img in images]
    return flat_dataset
def divBatch(numberOfBatch,dataset):
    batchSize = len(dataset) /  numberOfBatch
    batchSize = int(batchSize)
    res = []
    for i in range(0,numberOfBatch + 1):
        OneBatchArr = []
        print(min((i + 1) * batchSize,len(dataset)) - i * batchSize)
        for j in range(i * batchSize,min((i + 1) * batchSize,len(dataset))):
            if (i == 11):
                pass
            OneBatchArr.append(dataset[j])
        res.append(OneBatchArr)
    return res

import random
def DatasetToSuffledArray(dataset):
    datasetSuffled = []
    for digit in range(0,len( dataset)):
        for digitImg in dataset[digit]:
            datasetSuffled.append(ImageLable(digit,digitImg))
    random.shuffle(datasetSuffled)
    return datasetSuffled

def OpenIamgeAsFlatten(path):
    img = Image.open(path).convert('L')
    img = img.resize((32,32))
    ImagArr = np.array(img) 
    return ImagArr.flatten() / 255.0
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
class ImageLable:
    def __init__(self,lable,img):
        self.lable = lable
        self.img = img
    def Display(self):
        fig = plt.figure()

# Set the window title
        print(self.img)
        displayImg = np.stack((self.img,)*3, axis=-1)
        fig.canvas.manager.set_window_title(str(self.lable))
        plt.imshow(displayImg, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')  # Hide the axis
        plt.show()

    def Display(self):
        fig = plt.figure()

# Set the window title
        print(self.img)
        displayImg = np.stack((self.img,)*3, axis=-1)
        fig.canvas.manager.set_window_title(str(self.lable))
        plt.imshow(displayImg, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')  # Hide the axis
        plt.show()