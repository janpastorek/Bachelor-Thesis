from abc import ABC, abstractmethod

class RegressionModel(ABC):
    """ a linear regression models """

    def predict(self, x):
        """ predicts output for input """
        pass

    def sgd(self, x, y, learning_rate=0.01, momentum=0.9):
        """ makes one step of sgd """
        pass

    def load_weights(self, filepath):
        """ loads weights """
        pass

    def save_weights(self, filepath):
        """ saves weights """
        pass

    def get_losses(self):
        """ returns learning loss """
        pass