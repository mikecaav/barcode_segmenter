import data
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model



if __name__ == '__main__':
    model = build_model()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

    X_train, y_train = data.get_data_generator(data.IMAGE_TRAIN_PATH, data.MASK_TRAIN_PATH)
    X_val, y_val = data.get_data_generator(data.IMAGE_VALIDATION_PATH, data.IMAGE_VALIDATION_PATH)
    model.fit(
        zip(X_train, y_train),
        validation_data=zip(X_val, y_val),
        callbacks=[model_checkpoint, ]
    )
    X_test, y_test = data.get_data_generator(data.IMAGE_TEST_PATH, data.MASK_TEST_PATH)
    x = X_test.next()
    y = y_test.next()
    data.plot_images(y, model.predict(x)[0])
