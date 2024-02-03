import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall


class MyF1Score(tf.keras.metrics.Metric):
    """
        Niestandardowa metryka F1, rozszerzająca klasę Metric z TensorFlow.
    """

    def __init__(self, name='f1_score', **kwargs):
        super(MyF1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


def plot_metrics(test, model, loaded_history, model_name):
    """
    Ocenia dany i rysuje metryki precyzji, czułości i F1 modeli na zbiorze testowym.

    Parametry:
    - test: Zbiór danych testowych.
    - model: Wytrenowany model do oceny.
    - loaded_history: Historia treningu modelu wczytana z pliku.
    - model_name: Nazwa modelu do zapisu wykresu.

    Zwraca:
    - None (Wykres metryk jest zapisywany i wyświetlany).
    """

    # Ocena modelu na zbiorze testowym
    score = model.evaluate(test)

    print("Test Loss: ", score[0])
    print("Test Accuracy: ", score[1])

    # Tworzenie wykresów metryk
    fig, ax = plt.subplots(1, 3, figsize=(20, 3))
    ax = ax.ravel()

    metrics = ['test_precision', 'test_recall', 'test_f1_score']
    for i, met in enumerate(metrics):
        if met in loaded_history:
            ax[i].plot(loaded_history[met], label='Test')
            ax[i].set_title(met.capitalize())
            ax[i].set_xlabel('Epochs')
            ax[i].set_ylabel(met)
            ax[i].legend()

    plt.savefig('results/metrics/' + model_name + '_metrics.png')
    plt.show()
