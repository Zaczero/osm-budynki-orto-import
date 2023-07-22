import json
import random
from datetime import datetime
from itertools import chain
from math import ceil
from statistics import mean
from typing import Sequence

import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV3Large, MobileNetV3Small
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPooling2D, concatenate)
from keras.metrics import Precision
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import class_weight

from config import DATA_DIR, MODEL_DATASET_PATH, MODEL_PARAMS_PATH, SEED
from dataset import DatasetEntry, iter_dataset
from utils import save_image

_BATCH_SIZE = 32


# def create_datagen_flow(datagen: ImageDataGenerator, dataset: Sequence[DatasetEntry], batch_size: int = _BATCH_SIZE):
#     indices = np.arange(len(dataset))

#     while True:
#         indices = np.random.permutation(indices)

#         for start in range(0, len(indices), batch_size):
#             batch_indices = indices[start:start+batch_size]

#             batch_images = []
#             batch_masks = []
#             batch_labels = []

#             for i in batch_indices:
#                 entry = dataset[i]

#                 # apply the same transformation to the image and the mask
#                 seed = random.randint(0, 2**32)
#                 transformed_image = datagen.random_transform(entry.image, seed=seed)
#                 transformed_mask = datagen.random_transform(entry.mask, seed=seed)

#                 batch_images.append(transformed_image)
#                 batch_masks.append(transformed_mask)
#                 batch_labels.append(entry.label)

#             yield [np.stack(batch_images), np.stack(batch_masks)], np.array(batch_labels)


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
    z = Dropout(0.3)(z)
    z = Dense(256, activation='relu')(z)
    z = Dropout(0.3)(z)
    z = Dense(128, activation='relu')(z)
    z = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=image_inputs, outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Precision()])

    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=10,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=True,
    )

    callbacks = [
        EarlyStopping(patience=40, verbose=1),
        ModelCheckpoint(str(DATA_DIR / 'model.h5'), save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.2, patience=15, verbose=1),
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

    model: Model = load_model(str(DATA_DIR / 'model.h5'))

    threshold = 0.999
    print(f'Threshold: {threshold}')

    y_pred_proba = model.predict(X_holdout).flatten()
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


# class Model:
    # def __init__(self):
    #     df = load_dataset()

    #     X = df.drop(columns=['id', 'label'])
    #     y = df['label']

    #     with open(MODEL_PARAMS_PATH) as f:
    #         params = json.load(f)

    #     self.model = LGBMClassifier(**(_default_params() | params), random_state=SEED, verbose=-1)
    #     self.model.fit(X, y)

    # def predict_single(self, X: dict, *, threshold: float = 0.8) -> tuple[bool, float]:
    #     X = pd.DataFrame([X])
    #     y_pred_proba = self.model.predict_proba(X)
    #     y_pred = y_pred_proba[:, 1] >= threshold
    #     return y_pred[0], float(y_pred_proba[0, 1])
