import data
import tensorflow as tf
from model import build_model
from split_datasets import Path
import cv2
import numpy as np
from skimage.io import imsave

WEIGHTS_PATH = 'unet_200_steps.hdf5'
TEST_RESULT_PATH = 'results'

model = build_model()
model.load_weights(WEIGHTS_PATH)

X_test, y_test = data.get_data_generator(data.IMAGE_TEST_PATH, data.MASK_TEST_PATH)
Path.create_directory(TEST_RESULT_PATH)

i = 1
iou_score = 0
m = tf.keras.metrics.MeanIoU(num_classes=2)


def compute_iou(y_pred, y_true):
    m.update_state(y_true, y_pred)
    return m.result()


if __name__ == '__main__':
    for image_batch, mask_batch in zip(X_test, y_test):
        for image, mask in zip(image_batch, mask_batch):
            mask_predicted = model.predict(tf.expand_dims(image, axis=0))[0]
            mask_predicted[mask_predicted <= .5] = 0
            mask_predicted[mask_predicted > .5] = 1
            mask[mask <= .5] = 0
            mask[mask > .5] = 1
            iou_score += compute_iou(mask, mask_predicted)

            imsave(f'{TEST_RESULT_PATH}/{i}_predicted.png', mask_predicted)
            imsave(f'{TEST_RESULT_PATH}/{i}_true.png', mask)
            i += 1
            if i == 20:
                print(f'iou score: {float(iou_score/i)}')
                break
        if i == 20:
            break

