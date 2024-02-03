import tensorflow as tf


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs=None):
        """
        Wywoływane na końcu każdej epoki trenowania.

        Parametry:
        - epoch: Numer bieżącej epoki.
        - logs: Słownik metryk trenowania dla bieżącej epoki.
        """

        loss, precision, recall, f1_score = self.model.evaluate(self.test_data, verbose=0)
        logs["test_precision"] = precision
        logs["test_recall"] = recall
        logs["test_f1_score"] = f1_score
