import albumentations
import cv2
import os
import matplotlib.pyplot as plt
from random import shuffle

MASK_TRAIN_PATH = 'dataset/train/target/'
IMAGE_TRAIN_PATH = 'dataset/train/predictor/'

data_augmentor = albumentations.Compose(
    [
        albumentations.Resize(300, 300),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Rotate(limit=(-90, 90)),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ], p=1
)


def get_data_train_generator(new_num_of_images):
    filenames_list = os.listdir(MASK_TRAIN_PATH)
    shuffle(filenames_list)
    for filename in filenames_list:
        image_path = os.path.join(IMAGE_TRAIN_PATH, filename)
        mask_path = os.path.join(MASK_TRAIN_PATH, filename)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        for i in range(new_num_of_images):
            augmented_data = data_augmentor(image=image, mask=mask)
            yield augmented_data['image'], augmented_data['mask']


def plot_images(image, mask):
    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()


for image, mask in get_data_train_generator(10):
    plot_images(image, mask)
