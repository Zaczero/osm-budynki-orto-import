from datetime import datetime
from math import ceil
from typing import Sequence

import numpy as np
from keras.applications import MobileNetV3Large
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.experimental import CosineDecay
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from config import CONFIDENCE, DATA_DIR, MODEL_PATH, SEED
from dataset import DatasetEntry, iter_dataset

_BATCH_SIZE = 24
_EPOCHS = 25


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

    steps_per_epoch = ceil(len(train) / _BATCH_SIZE)

    datagen = ImageDataGenerator(
        width_shift_range=0.15,
        height_shift_range=0.15,
        rotation_range=180,
        shear_range=20,
        zoom_range=0.25,
        channel_shift_range=0.1,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True,
    )

    image_inputs = Input(dataset[0].image.shape)
    image_model = MobileNetV3Large(include_top=False,
                                   input_tensor=image_inputs,
                                   dropout_rate=0.3,
                                   include_preprocessing=False)

    freeze_ratio = 0.7
    for layer in image_model.layers[:int(len(image_model.layers) * freeze_ratio)]:
        layer.trainable = False

    z = image_model(image_inputs)
    z = Flatten()(z)
    z = Dropout(0.3)(z)
    z = BatchNormalization()(z)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.3)(z)
    z = Dense(128, activation='relu')(z)
    z = Dropout(0.3)(z)
    z = Dense(64, activation='relu')(z)
    z = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=image_inputs, outputs=z)
    model.compile(
        optimizer=Adam(
            CosineDecay(initial_learning_rate=1e-5,
                        decay_steps=steps_per_epoch * _EPOCHS - 5,
                        alpha=0.3,
                        warmup_target=5e-5,
                        warmup_steps=steps_per_epoch * 5,)),
        loss=BinaryCrossentropy(),
        metrics=[AUC()]
    )

    callbacks = [
        # ReduceLROnPlateau(factor=0.5,
        #                   min_lr=0.00001,
        #                   cooldown=5,
        #                   patience=10,
        #                   min_delta=0.0005,
        #                   verbose=1),

        # EarlyStopping('val_auc', mode='max',
        #               min_delta=0.0005,
        #               patience=35,
        #               verbose=1),

        ModelCheckpoint(str(MODEL_PATH), 'val_auc', mode='max',
                        initial_value_threshold=0.95,
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1),

        TensorBoard(str(DATA_DIR / 'tb' / datetime.now().strftime("%Y%m%d-%H%M%S")), histogram_freq=1),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=_BATCH_SIZE),
        epochs=_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights,
    )

    model.load_weights(str(MODEL_PATH))

    threshold = CONFIDENCE
    print(f'Threshold: {threshold}')

    y_pred_proba = model.predict(X_holdout).flatten()
    y_pred = y_pred_proba >= threshold

    auc_score = roc_auc_score(y_holdout, y_pred_proba)
    print(f'ROC AUC score: {auc_score:.3f}')

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
