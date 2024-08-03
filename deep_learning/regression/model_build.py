"""#Model definition"""
import tensorflow as tf
import keras_tuner as kt


def set_activation(activation_name):
    """
  set the activation function for the model
  """
    if activation_name == 'leaky_relu':
        activation = tf.keras.layers.LeakyReLU()
    else:
        activation = tf.keras.layers.Activation(activation_name)
    return activation


class DeepClassifierModel:
    hp = kt.HyperParameters()

    def __init__(self, input_dim):
        """
      initialize the hyperparameters for hyperparameter tuning
      """
        self.input_dim = input_dim
        self.neurons_max = 1024
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

        model.add(tf.keras.layers.Dense(
            units=hp.Int('hidden_1', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_1', values=self.activation_name)),
            input_dim=self.input_dim))

        model.add(tf.keras.layers.Dense(
            units=hp.Int('hidden_2', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_2', values=self.activation_name))))

        model.add(tf.keras.layers.Dense(
            units=hp.Int('hidden_3', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_3', values=self.activation_name))))
        
        model.add(tf.keras.layers.Dense(
            units=hp.Int('hidden_4', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_4', values=self.activation_name))))
        
        model.add(tf.keras.layers.Dense(
            units=hp.Int('hidden_5', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_5', values=self.activation_name))))
        
        model.add(tf.keras.layers.Dense(
            units=hp.Int('hidden_6', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_6', values=self.activation_name))))

        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        model.compile(optimizer=self.set_optimizer(hp.Choice('optimizer', values=self.optimizer_name),hp.Choice('learning_rate', values=self.learning_rates)),
                      loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def set_optimizer(self, optimizer_name,learn_rate):
        """
      set the optimizer for the model
      """
        if optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer_name == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=learn_rate)
        elif optimizer_name == 'nadam':
            return tf.keras.optimizers.Nadam(learning_rate=learn_rate)
        elif optimizer_name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=learn_rate)
