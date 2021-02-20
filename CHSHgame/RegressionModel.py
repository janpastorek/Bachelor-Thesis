from abc import ABC, abstractmethod

class RegressionModel(ABC):
    """ A linear regression models """

    def predict(self, X):
        """ Predicts output for input """
        pass

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        """ Makes One step of SGD """
        pass

    def load_weights(self, filepath):
        """ Loads weights """
        pass

    def save_weights(self, filepath):
        """ Saves weights """
        pass

    def get_losses(self):
        """ Returns learning loss """
        pass