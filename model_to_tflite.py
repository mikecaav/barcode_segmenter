from tensorflow import lite
from model import build_model

WEIGHTS_PATH = 'unet_200_steps.hdf5'

model = build_model()
model.load_weights(WEIGHTS_PATH)

MODEL_NAME = 'unet_200'

converter = lite.TFLiteConverter.from_keras_model(model)  # Your model's name
model = converter.convert()
file = open(f'{MODEL_NAME}.tflite', 'wb')
file.write(model)
