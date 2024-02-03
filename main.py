import os
import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from matplotlib import pyplot as plt
from flask import Flask, request, render_template
from tensorflow.keras.metrics import Precision, Recall
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from evaluation.metrics import plot_metrics, MyF1Score

from models.model_vgg19 import train_model_vgg19
from models.model_resnet50 import train_model_resnet50
from models.model_mobilenet import train_model_mobilenet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

PATH_TRAIN = Path('data/chest_xray/train')
PATH_VAL = Path('data/chest_xray/val')
PATH_TEST = Path('data/chest_xray/test')

app = Flask(__name__)

model = tf.keras.models.load_model('save_models/model_resnet50.h5',
                                   custom_objects={'MyF1Score': MyF1Score,
                                                   'Precision': Precision,
                                                   'Recall': Recall})

# Obsługa przesyłania obrazów przez UI przy użyciu Flask
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = image_file.filename
            image_file.save(image_location)
            img = load_img(image_location, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            labels = ['normal', 'pneumonia']
            prediction_label = labels[int(prediction > 0.5)]
            return render_template('index.html', prediction=prediction_label)
    return render_template('index.html', prediction=None)


def create_diagram(path, dataset):
    """
        Tworzy diagram przedstawiający liczbę przypadków dla każdej klasy w zbiorze danych.

        Parametry:
        - path: Ścieżka do katalogu zbioru danych.
        - dataset: Nazwa zbioru danych ('train', 'validation' lub 'test').

        Zwraca:
        - None (Zapisuje i wyświetla diagram).
    """

    class_counts = {subdir: len(os.listdir(os.path.join(path, subdir))) for subdir in ['NORMAL', 'PNEUMONIA']}

    plt.bar(class_counts.keys(), class_counts.values(), color=['blue', 'red'])
    plt.xlabel('Case type')
    plt.ylabel('Count')
    plt.title('Number of cases for ' + dataset + ' data')

    for i, value in enumerate(class_counts.values()):
        plt.text(i, value, str(value), ha='center', va='bottom')

    plt.savefig('results/count/' + dataset + '_count.png')
    plt.show()


def create_generator(directory):
    """
        Tworzy i konfiguruje generator danych.

        Parametry:
        - directory: Ścieżka do katalogu zbioru danych.

        Zwraca:
        - Skonfigurowany generator danych.
    """

    # Tworzenie obiektu ImageDataGenerator w celu skalowania wartości pikseli obrazu w zakresie [0, 1]
    data_generator = ImageDataGenerator(rescale=1. / 255)

    generator = data_generator.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=2,
        class_mode='binary')
    return generator


def main():
    # Tworzenie diagramu dla każdego zestawu danych (train, validation, test)
    # create_diagram(PATH_TRAIN, 'train')
    # create_diagram(PATH_VAL, 'validation')
    # create_diagram(PATH_TEST, 'test')

    # Tworzenie i konfiguracja generatorów danych dla każdego zestawu danych (train, validation, test)
    train_generator = create_generator(PATH_TRAIN)
    val_generator = create_generator(PATH_VAL)
    test_generator = create_generator(PATH_TEST)

    # Szkolenie modeli na danych treningowych z użyciem danych walidacyjnych
    # train_model_vgg19(train_generator, val_generator, test_generator)
    # train_model_resnet50(train_generator, val_generator, test_generator)
    # train_model_mobilenet(train_generator, val_generator, test_generator)

    # Ładowanie wytrenowanych modeli
    model_vgg19 = tf.keras.models.load_model('save_models/model_vgg19.h5',
                                             custom_objects={'MyF1Score': MyF1Score,
                                                             'Precision': Precision,
                                                             'Recall': Recall})
    model_resnet50 = tf.keras.models.load_model('save_models/model_resnet50.h5',
                                                custom_objects={'MyF1Score': MyF1Score,
                                                                'Precision': Precision,
                                                                'Recall': Recall})
    model_mobilenet = tf.keras.models.load_model('save_models/model_mobilenet.h5',
                                                 custom_objects={'MyF1Score': MyF1Score,
                                                                 'Precision': Precision,
                                                                 'Recall': Recall})

    with open('save_models/model_vgg19.json', 'r') as f:
        loaded_history_train_model_vgg19 = json.load(f)

    with open('save_models/model_resnet50.json', 'r') as f:
        loaded_history_train_model_resnet50 = json.load(f)

    with open('save_models/model_mobilenet.json', 'r') as f:
        loaded_history_train_model_mobilenet = json.load(f)

    # Wizualizacja metryk na zbiorze testowym dla każdego modelu
    plot_metrics(test_generator, model_vgg19, loaded_history_train_model_vgg19, 'vgg19')
    plot_metrics(test_generator, model_resnet50, loaded_history_train_model_resnet50, 'resnet50')
    plot_metrics(test_generator, model_mobilenet, loaded_history_train_model_mobilenet, 'mobilenet')


if __name__ == '__main__':
    main()
    app.run(port=5000, debug=True)
