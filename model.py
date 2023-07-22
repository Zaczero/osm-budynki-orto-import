from datetime import datetime
from math import ceil
from typing import Sequence

import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV3Large
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from keras.losses import BinaryCrossentropy
from keras.metrics import Precision
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from config import DATA_DIR, ERROR_RATE, MODEL_PATH, SEED
from dataset import DatasetEntry, iter_dataset

_BATCH_SIZE = 32


def _split_x_y(dataset: Sequence[DatasetEntry]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack(tuple(map(lambda x: x.image, dataset)))
    y = np.array(tuple(map(lambda x: x.label, dataset)))
    return X, y


def create_model():
    # dataset_iterator = iter_dataset()
    # dataset = tuple(next(dataset_iterator) for _ in range(100))
    dataset = tuple(iter_dataset())

    labels = tuple(map(lambda x: x.label, dataset))
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))

    train, temp = train_test_split(dataset,
                                   test_size=0.3,
                                   random_state=SEED,
                                   stratify=tuple(map(lambda x: x.label, dataset)))

    holdout, test = train_test_split(temp,
                                     test_size=2/3,
                                     random_state=SEED,
                                     stratify=tuple(map(lambda x: x.label, temp)))

    X_train, y_train = _split_x_y(train)
    X_test, y_test = _split_x_y(test)
    X_holdout, y_holdout = _split_x_y(holdout)

    # train: 70%
    # test: 20%
    # val: 10%

    image_inputs = Input(dataset[0].image.shape)
    image_model = MobileNetV3Large(include_top=False,
                                   input_tensor=image_inputs,
                                   include_preprocessing=False)

    freeze_ratio = 0.6
    for layer in image_model.layers[:int(len(image_model.layers) * freeze_ratio)]:
        layer.trainable = False

    z = image_model(image_inputs)
    z = Flatten()(z)
    z = BatchNormalization()(z)
    z = Dropout(0.2)(z)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(1)(z)

    model = Model(inputs=image_inputs, outputs=z)
    model.compile(optimizer='adam',
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=[Precision(0.8)])

    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=15,
        zoom_range=0.2,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True,
    )

    callbacks = [
        EarlyStopping(patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), save_best_only=True, verbose=1),
        TensorBoard(str(DATA_DIR / 'tb' / datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=_BATCH_SIZE),
        epochs=1000,
        steps_per_epoch=ceil(len(train) / _BATCH_SIZE),
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights,
    )

    model: Model = load_model(str(MODEL_PATH))

    threshold = 1 - ERROR_RATE
    print(f'Threshold: {threshold}')

    y_pred_logit = model.predict(X_holdout)
    y_pred_proba = tf.sigmoid(y_pred_logit).numpy().flatten()
    y_pred = y_pred_proba >= threshold

    val_score = precision_score(y_holdout, y_pred)
    print(f'Validation score: {val_score:.3f}')
    print()

    tn, fp, fn, tp = confusion_matrix(y_holdout, y_pred).ravel()
    print(f'True Negatives: {tn}')
    print(f'[❗] False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print(f'[✅] True Positives: {tp}')
    print()

    for pred, proba, true, entry in sorted(zip(y_pred, y_pred_proba, y_holdout, holdout), key=lambda x: x[3].id.lower()):
        if pred != true and not true:
            print(f'FP: {entry.id!r} - {true} != {pred} [{proba:.3f}]')
