"""
Model MobileNetV2

Lekki model głębokiego uczenia zaprojektowany dla urządzeń mobilnych i brzegowych. Jest to rozszerzenie oryginalnej
architektury MobileNet, mające na celu zapewnienie wydajnej i dokładnej klasyfikacji obrazów przy zmniejszonej
złożoności obliczeniowej. Model jest wstępnie wytrenowany na zbiorze danych ImageNet, co pozwala mu
dobrze generalizować do szerokiego zakresu zadań rozpoznawania wizualnego.

"""

import json
import tensorflow as tf
from mpld3._display import NumpyEncoder  # Import NumpyEncoder do obsługi serializacji obiektów NumPy w JSON
from evaluation.metrics import MyF1Score
from tensorflow.keras.metrics import Precision, Recall

from models.callback.test_callback import TestCallback


def train_model_mobilenet(train, validation, test):
    """
    Szkolenie modelu MobileNetV2 do klasyfikacji obrazów.

    Parametry:
    - train: Zbiór danych treningowych.
    - validation: Zbiór danych do walidacji.
    - test: Zbiór danych testowych.

    Zwraca:
    - Wytrenowany model na podanych danych.
    """

    # Wczytanie wstępnie wytrenowanego modelu MobileNetV2
    mobilenet_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Zamrożenie wszystkich warstw w wstępnie wytrenowanym modelu
    for layer in mobilenet_model.layers:
        layer.trainable = False

    # Dodanie warstw do transfer learning
    x = mobilenet_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Stworzenie ostatecznego modelu do treningu
    model = tf.keras.Model(inputs=mobilenet_model.input, outputs=predictions)

    # Konfiguracja funkcji zwrotnej dla wczesnego zatrzymywania i redukcji learning rate
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8)

    # Kompilacja modelu
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[Precision(name='precision'), Recall(name='recall'), MyF1Score()])

    # Szkolenie modelu
    history = model.fit(train, epochs=30,
                        validation_data=validation,
                        steps_per_epoch=100,
                        callbacks=[early_stopping, lr, TestCallback(test)],
                        batch_size=32)

    # Zapisanie wytrenowanego modelu i historii treningu
    model.save('save_models/model_mobilenet.h5')

    with open('save_models/model_mobilenet.json', 'w') as f:
        json.dump(history.history, f, cls=NumpyEncoder)
