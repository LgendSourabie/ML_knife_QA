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
      self.neurons_max = 1024
      self.neurons_min = 8
      self.neurons_step = 16
      self.dropout_min = 0.0
      self.dropout_max = 0.5
      self.dropout_step = 0.1
      self.l2_min = 0.0
      self.l2_max = 0.1
      self.l2_step = 0.01
      self.learning_rates = [1e-2, 1e-3, 1e-4,1e-5]
      self.activation_name = ['relu', 'leaky_relu', 'elu', 'selu']
      self.optimizer_name = ['adam', 'nadam','rmsprop']

  def build_model(self,hp):
      """
      create a NN architecture for the classifiction task
      hp: hyperparameter for hyperparameter tuning
      """
      model = tf.keras.models.Sequential()

      model.add(tf.keras.layers.Dense(units = hp.Int('hidden_1', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
                      activation =set_activation(hp.Choice('activation_1', values=self.activation_name)),input_dim=self.input_dim))

      tf.keras.regularizers.L2(hp.Float('l2_1', min_value=self.l2_min, max_value=self.l2_max, step=self.l2_step))

      model.add(tf.keras.layers.Dense(units = hp.Int('hidden_2', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
                      activation = set_activation(hp.Choice('activation_2', values=self.activation_name))))

      tf.keras.regularizers.L2(hp.Float('l2_2', min_value=self.l2_min, max_value=self.l2_max, step=self.l2_step))

      model.add(tf.keras.layers.Dense(units = hp.Int('hidden_3', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
                      activation = set_activation(hp.Choice('activation_3', values=self.activation_name))))
      
      tf.keras.regularizers.L2(hp.Float('l2_3', min_value=self.l2_min, max_value=self.l2_max, step=self.l2_step))

      model.add(tf.keras.layers.Dense(units = hp.Int('hidden_4', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
                      activation = set_activation(hp.Choice('activation_4', values=self.activation_name))))
      
      tf.keras.regularizers.L2(hp.Float('l2_4', min_value=self.l2_min, max_value=self.l2_max, step=self.l2_step))

      model.add(tf.keras.layers.Dense(units = hp.Int('hidden_5', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
                      activation = set_activation(hp.Choice('activation_5', values=self.activation_name))))
      
      tf.keras.regularizers.L2(hp.Float('l2_5', min_value=self.l2_min, max_value=self.l2_max, step=self.l2_step))
    

      model.add(tf.keras.layers.Dense(units = hp.Int('hidden_6', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
                      activation = set_activation(hp.Choice('activation_6', values=self.activation_name))))
      
      tf.keras.regularizers.L2(hp.Float('l2_6', min_value=self.l2_min, max_value=self.l2_max, step=self.l2_step))

      model.add(tf.keras.layers.Dense(units= 1, activation =set_activation(hp.Choice('activation_out', values=self.activation_name))))

      model.compile(optimizer =self.set_optimizer(hp.Choice('optimizer', values=self.optimizer_name), hp.Choice('learning_rate', values=self.learning_rates)) ,
                    loss='mean_squared_error',metrics=[tf.keras.metrics.R2Score(name='r_squared'), tf.keras.metrics.MeanSquaredError(name='mean_squared_error')])
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
