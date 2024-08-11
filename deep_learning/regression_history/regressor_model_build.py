"""#Model definition"""
import tensorflow as tf
import keras_tuner as kt


def set_activation(activation_name):
    """
    set the activation function for the model
    """
    if activation_name == 'leaky_relu':
        activation = tf.keras.layers.LeakyReLU()
    elif activation_name == 'elu':
        activation = tf.keras.layers.ELU()
    elif activation_name == 'selu':
        activation = tf.keras.layers.Activation('selu')
    else:
        activation = tf.keras.layers.Activation(activation_name)
    return activation


class DeepRegressionModel:
    hp = kt.HyperParameters()

    def __init__(self, input_dim):
        """
      initialize the hyperparameters for hyperparameter tuning
      """
        self.input_dim = input_dim
        self.neurons_max = 256
        self.neurons_min = 32
        self.neurons_step = 32
        self.drop_min = 0.0
        self.drop_max = 0.2
        self.drop_step = 0.1
        self.l2_min = 0.0
        self.l2_max = 0.3
        self.l2_step = 0.01
        self.layers = [2, 3, 4, 5]
        self.learning_rates = [1e-3, 1e-4, 1e-5]
        self.activation_name = ['relu', 'leaky_relu', 'tanh', 'selu']
        self.optimizer_name = ['adam', 'nadam', 'rmsprop']

    def build_model(self, hp):
        """
      create a NN architecture for the classifiction task
      hp: hyperparameter for hyperparameter tuning
      """
        model = tf.keras.models.Sequential()

        # Input layer
        model.add(tf.keras.layers.Dense(
            units=hp.Int('hidden_1', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_1', values=self.activation_name)),
            input_dim=self.input_dim))
        model.add(tf.keras.layers.Dropout(
            hp.Float('drop_1', min_value=self.drop_min, max_value=self.drop_max,
                     step=self.drop_step)))  # Dropout layer
        model.add(tf.keras.layers.BatchNormalization())  # batch normalization layer

        # Number of hidden layers
        num_layers = hp.Choice('num_layers', values=self.layers)

        for i in range(2, num_layers + 1):
            model.add(tf.keras.layers.Dense(
                units=hp.Int(f'hidden_{i}', min_value=self.neurons_min, max_value=self.neurons_max,
                             step=self.neurons_step),
                activation=set_activation(hp.Choice(f'activation_{i}', values=self.activation_name))))
            model.add(tf.keras.layers.Dropout(
                hp.Float(f'drop_{i}', min_value=self.drop_min, max_value=self.drop_max,
                         step=self.drop_step)))  # Dropout layer
            model.add(tf.keras.layers.BatchNormalization())  # batch normalization layer

        model.add(tf.keras.layers.Dense(units=1, activation=set_activation(
            hp.Choice('activation_out', values=self.activation_name))))

        model.compile(optimizer=self.set_optimizer(hp.Choice('optimizer', values=self.optimizer_name),
                                                   hp.Choice('learning_rate', values=self.learning_rates)),
                      loss='mean_squared_error', metrics=[tf.keras.metrics.R2Score(name='r_squared'),
                                                          tf.keras.metrics.MeanSquaredError(name='mean_squared_error')])
        return model

    def set_optimizer(self, optimizer_name, learn_rate):
        """
      set the optimizer for the model
      """
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer_name == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=learn_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learn_rate)
        return optimizer
