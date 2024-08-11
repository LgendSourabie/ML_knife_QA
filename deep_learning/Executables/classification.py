from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import pandas as pd
import seaborn as sns

"""#Importing data"""

dataset = pd.read_excel('../../data/chiefs_knife_dataset.xlsx')
index_Ra = dataset.columns.get_loc('Ra')  # index of the surface roughness column for inserting the class. label

LOWER_SPECIFICATION_LIMIT = 0.125  # lower bound of good quality product region
UPPER_SPECIFICATION_LIMIT = 0.215  # upper bound of good quality product region

is_between_specification_bounds = (dataset['Ra'] >= LOWER_SPECIFICATION_LIMIT) & (
        dataset['Ra'] < UPPER_SPECIFICATION_LIMIT)
good_product_range = np.where(is_between_specification_bounds, "good", "bad")
dataset.insert(index_Ra + 1, 'Quality', good_product_range)

"""# constructing Label"""

X = dataset.loc[:, 'Original_Linienanzahl':'DFT_Median_sobel_Bereich'].values
y_classifier = dataset['Quality'].values
y_regressor = dataset['Ra'].values

"""#Encoding categorical data"""

y_classifier = np.where(y_classifier == 'good', 0, 1)

"""#Splitting dataset into training and test set"""


def get_split_dataset(regressor=True, min_max_scaler=True, X=X, y_regressor=y_regressor, y_classifier=y_classifier,
                      test_size=0.2, rnd_state=1):
    mm_sc = MinMaxScaler()
    sc = StandardScaler()
    if regressor:
        X_train, X_test, y_train, y_test = train_test_split(X, y_regressor, test_size=test_size, shuffle=True,
                                                            random_state=rnd_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_classifier, test_size=test_size, shuffle=True,
                                                            random_state=rnd_state)

    if min_max_scaler:
        X_train = mm_sc.fit_transform(X_train)
        X_test = mm_sc.transform(X_test)
    else:
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    # we Resample the dataset to balance the classes
    return X_train, X_test, y_train, y_test


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
        self.neurons_max = 256
        self.neurons_min = 32
        self.neurons_step = 32
        self.drop_min = 0.0
        self.drop_max = 0.5
        self.drop_step = 0.1
        self.layers = [2, 3, 4, 5]
        self.learning_rates = [1e-3, 1e-4, 1e-5]
        self.activation_name = ['relu', 'tanh', 'selu', 'leaky_relu']
        self.optimizer_name = ['adam', 'nadam', 'rmsprop']

    def build_model(self, hp):
        """
      create a NN architecture for the classifiction task
      hp: hyperparameter for hyperparameter tuning
      """
        model = tf.keras.models.Sequential()

        # Input layer and first hidden layer
        model.add(tf.keras.layers.Dense(
            units=hp.Int('hidden_1', min_value=self.neurons_min, max_value=self.neurons_max, step=self.neurons_step),
            activation=set_activation(hp.Choice('activation_1', values=self.activation_name)),
            input_dim=self.input_dim))
        model.add(tf.keras.layers.Dropout(
            hp.Float('drop_1', min_value=self.drop_min, max_value=self.drop_max, step=self.drop_step)))  # Dropout layer
        model.add(tf.keras.layers.BatchNormalization())  # batch normalization layer

        # Number of hidden layers , use to let the algorithm look for the best # of hidden layer
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

        # Output layer
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        model.compile(optimizer=self.set_optimizer(hp.Choice('optimizer', values=self.optimizer_name),
                                                   hp.Choice('learning_rate', values=self.learning_rates)),
                      loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                                                           tf.keras.metrics.Recall(name='recall')])
        return model

    def set_optimizer(self, optimizer_name, learn_rate):
        """
      set the optimizer for the model
      """
        if optimizer_name == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=learn_rate)
        elif optimizer_name == 'nadam':
            return tf.keras.optimizers.Nadam(learning_rate=learn_rate)
        elif optimizer_name == 'rmsprop':
            return tf.keras.optimizers.RMSprop(learning_rate=learn_rate)


# Hyperparameter tuning
X_train, _, _, _ = get_split_dataset(regressor=False, min_max_scaler=False)

directory = "hyperparameter_tuning"
project_name = "knife_classifier"

classifier = DeepClassifierModel(X_train.shape[1])

tuner = kt.RandomSearch(
    hypermodel=classifier.build_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=1,
    overwrite=False,  # we don't want to overwrite the existing parameter
    directory=directory,
    project_name=project_name,
)


def get_best_models(rank_index=0):
    """
    get the best model and give a detailed summary of the model
    :param rank_index:
    :return: give the architecture of the best model
    """
    models = tuner.get_best_models(num_models=rank_index + 2)
    return models[rank_index].summary()


def get_tuning_summary():
    """
    get the summary of hyperparameter tuning
    :return: the best 10 models with their architecture and hyperparameters
    """
    return tuner.results_summary()


def get_best_parameter():
    """
    get the four best model
    :return: four best models for fittin purpose
    """
    return tuner.get_best_hyperparameters(4)


def get_best_hyperparameter(var=tuner):
    """
    get the best hyperparameter
    :param var: tuner - the hyperparameter tuner
    :return: best model parameters as a dictionary
    """
    best_params = var.get_best_hyperparameters(5)[0].values
    return best_params


# classifier

X_train, X_test, y_train, y_test = get_split_dataset(regressor=False, min_max_scaler=False)

classifier = DeepClassifierModel(input_dim=X_train.shape[1])
best_parameters = get_best_parameter()[1]

######## take secon model

# we reduce the optimizer for better convergence and stability
best_parameters.values['learning_rate'] = 1e-5

model = classifier.build_model(best_parameters)  # we rebuild the model with the chosen hyperparameters

checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint.model.keras",
                                                monitor="val_loss",
                                                mode="min",
                                                save_best_only=True,
                                                verbose=1)

# the history of improvement of the model based on validation loss is stored
callbacks = [checkpoint]

history = model.fit(X_train, y_train, batch_size=64, callbacks=callbacks, epochs=155, validation_split=0.2)

# prediction
y_pred = model.predict(X_test)
y_pred = np.reshape(y_pred, len(y_pred))

# save the metrics
dic_metric = {'Quality observed': y_test, 'Quality predicted': y_pred}
prediction = pd.DataFrame(data=dic_metric).to_csv('classifier_only_test.csv')

# Evaluation of the model
score = model.evaluate(X_test, y_test, verbose=0)

#Metrics of  the model
confusion_matrix = confusion_matrix(y_test, y_pred.round())
accuracy_model = accuracy_score(y_test, y_pred.round())
recall_model = recall_score(y_test, y_pred.round())
precision_model = precision_score(y_test, y_pred.round())
f1_model = f1_score(y_test, y_pred.round())
print('\n')
print(f'Test loss       : {round(score[0], 5)}')
print(f'accuracy_model  : {round(accuracy_model * 100, 2)}')
print(f'recall_model    : {round(recall_model * 100, 2)}')
print(f'precision_model : {round(precision_model * 100, 2)}')
print(f'f1_model        : {round(f1_model * 100, 2)}')
print(f"confusion matrix: \n{confusion_matrix}")

with open('metrics_eval_summary.txt', 'w+') as file:
    file.write("################### RESULT OF PREDICTION ###################\n\n")
    file.write(f" Test loss : {round(score[0], 5)}\n")
    file.write(f" accuracy  : {round(accuracy_model * 100, 2)} %\n")
    file.write(f" Recall    : {round(recall_model * 100, 2)} %\n")
    file.write(f" Precision : {round(precision_model * 100, 2)} %\n")
    file.write(f" F1-Score  : {round(f1_model * 100, 2)} %\n")

plt.figure(figsize=(15, 8))

plt.subplot(121)
plt.plot(history.history['loss'], label='Training loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation loss', linewidth=2)
plt.title('Losses', fontsize=18)
plt.xlabel('Epochs [-]', fontsize=12)
plt.ylabel('Loss [-]', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(122)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Accuracies', fontsize=18)
plt.xlabel('Epochs [-]', fontsize=12)
plt.ylabel('Accuracy [-]', fontsize=12)
plt.legend(fontsize=12)

plt.savefig(f'performance_classifier.png', dpi=300)

plt.figure(figsize=(10, 7))
group_names = ["Bad and predicted as Bad", "Bad but predicted as Good", "Good but predicted as Bad",
               "Good and predicted as Good"]
group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
labels = [f"{v1}\n\n{v2}\n\n{v3}" for v1, v2, v3 in zip(group_counts, group_percentages, group_names)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(confusion_matrix, annot=labels, xticklabels=['Bad products', 'Good Products'],
            yticklabels=['Bad products', 'Good products'], fmt="", cmap='Blues')
plt.xlabel('Predicted values', fontsize=16)
plt.ylabel('Actual values', fontsize=16)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.savefig(f'confusion_matrix.png', dpi=300)
