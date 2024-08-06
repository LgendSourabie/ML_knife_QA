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
        self.l2_min = 0.0
        self.l2_max = 0.1
        self.l2_step = 0.05
        self.layers = [2,3,4,5,6,7,8,9,10]
        self.learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
        self.activation_name = ['relu', 'tanh', 'selu', 'leaky_relu']
        self.optimizer_name = ['adam', 'nadam', 'rmsprop']

    def build_model(self, hp):
        """
      create a NN architecture for the classifiction task
      hp: hyperparameter for hyperparameter tuning
      """
        model = tf.keras.models.Sequential()

        # Input layer
        model.add(tf.keras.layers.Dense(
            units=hp.Int('input_neurons', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_1', values=self.activation_name)),
            kernel_regularizer=tf.keras.regularizers.L2(hp.Float('l2_1', min_value=self.l2_min, max_value=self.l2_max, step=self.l2_step)),
            input_dim=self.input_dim))

        # Number of hidden layers
        num_layers = hp.Choice('num_layers', values=self.layers)
        
        for i in range(2, num_layers + 1):
            model.add(tf.keras.layers.Dense(
                units=hp.Int(f'hidden_neuron_{i}', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
                activation=set_activation(hp.Choice(f'activation_{i}', values=self.activation_name)),
                kernel_regularizer=tf.keras.regularizers.L2(hp.Float(f'l2_{i}', min_value=self.l2_min, max_value=self.l2_max, step=self.l2_step))
            ))

        # Output layer
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        model.compile(optimizer=self.set_optimizer(hp.Choice('optimizer', values=self.optimizer_name),hp.Choice('learning_rate', values=self.learning_rates)),
                      loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
        return model

    def set_optimizer(self, optimizer_name,learn_rate):
        """
      set the optimizer for the model
      """
        if optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer_name == 'nadam':
            return tf.keras.optimizers.Nadam(learning_rate=learn_rate)
        elif optimizer_name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=learn_rate)
