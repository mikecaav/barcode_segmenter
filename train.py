import data
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model



if __name__ == '__main__':
    X_train, y_train = data.get_data_generator(data.IMAGE_TRAIN_PATH, data.MASK_TRAIN_PATH)
    X_val, y_val = data.get_data_generator(data.IMAGE_VALIDATION_PATH, data.IMAGE_VALIDATION_PATH)

    model = build_model()
    model_checkpoint = ModelCheckpoint('unet_200_steps.hdf5', monitor='loss', verbose=1, save_best_only=True)

    history = model.fit(
        zip(X_train, y_train),
        callbacks=[model_checkpoint, ],
        epochs=200,
        steps_per_epoch=234 // 16,
        batch_size=16,
        validation_data=(zip(X_val, y_val)),
        validation_steps=10
    )

    X_test, y_test = data.get_data_generator(data.IMAGE_TEST_PATH, data.MASK_TEST_PATH)
    x = X_test.next()
    y = y_test.next()
    data.plot_images(y, model.predict(x)[0])
