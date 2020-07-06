import data
import tensorflow as tf
from model import build_model
import cv2
from split_datasets import Path
from skimage.io import imsave

WEIGHTS_PATH = 'unet_200_steps.hdf5'
TEST_RESULT_PATH = 'results'

model = build_model()
model.load_weights(WEIGHTS_PATH)

X_test, y_test = data.get_data_generator(data.IMAGE_TEST_PATH, data.MASK_TEST_PATH)
Path.create_directory(TEST_RESULT_PATH)

i = 0
for image_batch in X_test:
    for image in image_batch:
        mask = model.predict(tf.expand_dims(image, axis=0))
        filtered_image = cv2.bitwise_and(image, mask[0])
        data.plot_images(image, filtered_image)
        imsave(f'{TEST_RESULT_PATH}/{i}.png', filtered_image)
        i += 1
