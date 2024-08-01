"""#Model definition"""
from tensorflow.keras.layers import Dense, LeakyReLU, Activation
from tensorflow.keras.optimizers import Adam, SGD, Nadam, RMSprop
import tensorflow as tf
import keras_tuner as kt


def set_activation(activation_name):
    """
  set the activation function for the model
  """
    if activation_name == 'leaky_relu':
        activation = LeakyReLU()
    else:
        activation = Activation(activation_name)
    return activation


class DeepClassifierModel:
    hp = kt.HyperParameters()

    def __init__(self, input_dim):
        """
      initialize the hyperparameters for hyperparameter tuning
      """
        self.input_dim = input_dim
        self.neurons_max = 512
        self.neurons_min = 32
        self.neurons_step = 16
        self.learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
        self.activation_name = ['relu', 'tanh', 'selu', 'leaky_relu']
        self.optimizer_name = ['adam', 'nadam', 'rmsprop', 'sgd']

    def build_model(self, hp):
        """
      create a NN architecture for the classifiction task
      hp: hyperparameter for hyperparameter tuning
      """
        model = tf.keras.models.Sequential()

        model.add(Dense(
            units=hp.Int('hidden_1', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_1', values=self.activation_name)),
            input_dim=self.input_dim))

        model.add(Dense(
            units=hp.Int('hidden_2', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_2', values=self.activation_name))))

        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer=self.set_optimizer(hp.Choice('optimizer', values=self.optimizer_name)),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def set_optimizer(self, optimizer_name, hp=hp):
        """
      set the optimizer for the model
      """
        if optimizer_name == 'adam':
            return Adam(hp.Choice('learning_rate', values=self.learning_rates))
        elif optimizer_name == 'sgd':
            return SGD(hp.Choice('learning_rate', values=self.learning_rates))
        elif optimizer_name == 'nadam':
            return Nadam(hp.Choice('learning_rate', values=self.learning_rates))
        elif optimizer_name == 'rmsprop':
            return RMSprop(hp.Choice('learning_rate', values=self.learning_rates))
