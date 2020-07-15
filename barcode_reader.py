import cv2
from pyzbar.pyzbar import decode
from model import get_trained_model
import os


model = get_trained_model()
if __name__ == '__main__':
    for image in os.listdir('results/'):
        image = cv2.imread(f'results/{image}')
        print(decode(image))



