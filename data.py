import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
import cv2
from skimage.transform import resize

MASK_TRAIN_PATH = 'dataset/train/mask'
IMAGE_TRAIN_PATH = 'dataset/train/image'
MASK_VALIDATION_PATH = 'dataset/validation/mask'
IMAGE_VALIDATION_PATH = 'dataset/validation/image'
MASK_TEST_PATH = 'dataset/test/mask'
IMAGE_TEST_PATH = 'dataset/test/image'

IMAGE_HEIGHT = 512
IMAGE_WEIGHT = 512
BATCH_SIZE = 16


def get_data_generator(datatype_image_path, datatype_mask_path):
    SEED = 1
    files_list = os.listdir(f'{datatype_image_path}')
    num_of_samples = len(files_list)
    list_of_images = np.zeros((num_of_samples, IMAGE_HEIGHT, IMAGE_WEIGHT, 3))
    list_of_masks = np.zeros((num_of_samples, IMAGE_HEIGHT, IMAGE_WEIGHT, 3))
    for i, file in enumerate(tqdm(files_list, 'Loading images')):
        image_file_path = f'{datatype_image_path}/{file}'
        image = cv2.imread(image_file_path)
        image = resize(image, (IMAGE_HEIGHT, IMAGE_WEIGHT))
        list_of_images[i] = image

        mask_file_path = f'{datatype_mask_path}/{file}'
        mask = cv2.imread(mask_file_path)
        mask = resize(mask, (IMAGE_HEIGHT, IMAGE_WEIGHT))
        list_of_masks[i] = mask

    image_datagen = ImageDataGenerator(
        height_shift_range=0.2,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.2,
        shear_range=0.2
    )
    image_datagen.fit(x=list_of_images, augment=True, seed=SEED)

    mask_datagen = ImageDataGenerator(
        height_shift_range=0.2,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.2,
        shear_range=0.2
    )
    mask_datagen.fit(x=list_of_masks, augment=True, seed=SEED)

    X = image_datagen.flow(list_of_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    y = mask_datagen.flow(list_of_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    return X, y


def get_test_data_generator():
    SEED = 1
    datatype_image_path = IMAGE_TEST_PATH
    datatype_mask_path = MASK_TEST_PATH
    files_list = os.listdir(f'{datatype_image_path}')
    num_of_samples = len(files_list)
    list_of_images = np.zeros((num_of_samples, IMAGE_HEIGHT, IMAGE_WEIGHT, 3))
    list_of_masks = np.zeros((num_of_samples, IMAGE_HEIGHT, IMAGE_WEIGHT, 3))
    for i, file in enumerate(tqdm(files_list, 'Loading images')):
        image_file_path = f'{datatype_image_path}/{file}'
        image = cv2.imread(image_file_path)
        image = resize(image, (IMAGE_HEIGHT, IMAGE_WEIGHT))
        list_of_images[i] = image

        mask_file_path = f'{datatype_mask_path}/{file}'
        mask = cv2.imread(mask_file_path)
        mask = resize(mask, (IMAGE_HEIGHT, IMAGE_WEIGHT))
        list_of_masks[i] = mask

    image_datagen = ImageDataGenerator()
    image_datagen.fit(x=list_of_images, augment=True, seed=SEED)

    mask_datagen = ImageDataGenerator(
        height_shift_range=0.2,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.2,
        shear_range=0.2
    )
    mask_datagen.fit(x=list_of_masks, augment=True, seed=SEED)

    X = image_datagen.flow(list_of_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    y = mask_datagen.flow(list_of_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    return X, y


def plot_images(image, mask):
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()


if __name__ == '__main__':
    image_datagen, mask_datagen = get_data_generator(IMAGE_VALIDATION_PATH, MASK_VALIDATION_PATH)
    for (images, masks) in zip(image_datagen, mask_datagen):
        for (image, mask) in zip(images, masks):
            plot_images(image, mask)
