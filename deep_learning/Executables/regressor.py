from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import keras_tuner as kt
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score


sns.set()

"""#Importing data"""

dataset = pd.read_excel('../../data/chiefs_knife_dataset.xlsx')
index_Ra = dataset.columns.get_loc('Ra')  # index of the surface roughness column for inserting the class. label

LOWER_SPECIFICATION_LIMIT = 0.125  # lower bound of good quality product region
UPPER_SPECIFICATION_LIMIT = 0.215  # upper bound of good quality product region

is_between_specification_bounds = (dataset['Ra'] >= LOWER_SPECIFICATION_LIMIT) & (
        dataset['Ra'] <= UPPER_SPECIFICATION_LIMIT)
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
    smote = SMOTE(random_state=42)
    if regressor:
        X_train, X_test, y_train, y_test = train_test_split(X, y_regressor, test_size=test_size, shuffle=True,
                                                            random_state=rnd_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_classifier, test_size=test_size, shuffle=True,
                                                            random_state=rnd_state)
        # we Resample the dataset to balance the classes
        X_train, y_train = smote.fit_resample(X_train, y_train)
    if min_max_scaler:
        X_train = mm_sc.fit_transform(X_train)
        X_test = mm_sc.transform(X_test)
    else:
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


# Model Builder

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


# Hyperparameter tuning
X_train, _, _, _ = get_split_dataset(regressor=False, min_max_scaler=False)

directory = "hyperparameter_tuning"
project_name = "knife_regressor"

regressor = DeepRegressionModel(X_train.shape[1])

tuner = kt.RandomSearch(
    hypermodel=regressor.build_model,
    objective=kt.Objective("val_r_squared", direction="max"),
    max_trials=1,
    executions_per_trial=1,
    overwrite=False,
    directory=directory,
    project_name=project_name,
)


def get_best_models(rank_index=0):
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
    best_params = var.get_best_hyperparameters(5)[1].values
    return best_params


# regressor

X_train, X_test, y_train, y_test = get_split_dataset(regressor=True, min_max_scaler=False)

regressor = DeepRegressionModel(input_dim=X_train.shape[1])
best_parameters = get_best_parameter()[0]

######## take secon model
model = regressor.build_model(best_parameters)

checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint.model.keras",
                                                monitor="val_loss",
                                                mode="min",
                                                save_best_only=True,
                                                verbose=1)

callbacks = [checkpoint]

history = model.fit(X_train, y_train, batch_size=64, callbacks=callbacks, verbose=1, epochs=150, validation_split=0.2)

# prediction

y_pred = model.predict(X_test)
y_pred = np.reshape(y_pred, len(y_pred))

#evaluation of model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
r2_score = r2_score(y_true=y_test, y_pred=y_pred)

#Error of prediction
error_pred = y_test - y_pred

# save the metrics
dic_metric = {'Ra observed': y_test, 'Ra predicted': y_pred, 'Error': error_pred}
prediction = pd.DataFrame(data=dic_metric).to_csv('only_regressor_test.csv')

with open('metrics_eval_summary.txt', 'w+') as file:
    file.write("################### RESULT OF PREDICTION ###################\n\n")
    file.write(f" Test loss: {round(score[0], 5)}\n")
    file.write(f" Mean Square Error: {round(mse, 5)}\n")
    file.write(f" R2Score: {round(r2_score * 100, 2)} %\n")

plt.figure(figsize=(15, 8))

plt.subplot(121)
plt.plot(history.history['loss'], label='Training loss', linewidth=4)
plt.plot(history.history['val_loss'], label='Validation loss', linewidth=4)
plt.title('Losses', fontsize=20)
plt.ylabel('Loss [-]', fontsize=14)
plt.xlabel('Epochs [-]', fontsize=14)
plt.legend(fontsize=12)

plt.subplot(122)
plt.plot(history.history['mean_squared_error'], label='Training mse', linewidth=4)
plt.plot(history.history['val_mean_squared_error'], label='Validation mse', linewidth=4)
plt.title('Mean Squared Errors', fontsize=20)
plt.ylabel('MSE [-]', fontsize=16)
plt.xlabel('Epochs [-]', fontsize=16)
plt.legend(fontsize=12)
plt.savefig('performance_regressor.png', dpi=600)

plt.figure(figsize=(15, 8))
plt.plot(range(0, len(y_test)), y_test, '-r', label="Observed Ra")
plt.plot(range(0, len(y_pred)), y_pred, '-g', label="Predicted Ra")
plt.plot(range(0, len(y_pred)), error_pred, '-k', label="Error")
plt.title('Ra Prediction Error', fontsize=20)
plt.xlabel('# of Instances [-]', fontsize=16)
plt.ylabel('Ra values [-]', fontsize=16)
plt.legend(fontsize=12)
plt.savefig('regressor_error.png', dpi=600)

plt.figure(figsize=(15, 8))

plt.axvline(LOWER_SPECIFICATION_LIMIT, color='black', linewidth=4)
plt.axvline(UPPER_SPECIFICATION_LIMIT, color='black', linewidth=4)
plt.fill_betweenx([0, 1], 0, 0.5, color='green', alpha=0.4)
plt.fill_betweenx([0, 1], LOWER_SPECIFICATION_LIMIT, UPPER_SPECIFICATION_LIMIT, color='blue', alpha=0.4)
plt.scatter(y_test, y_test + np.random.uniform(0.10, 0.6, size=y_test.shape), color='red', linewidth=4,
            label='Observed Ra')
plt.scatter(y_pred, y_pred + np.random.uniform(0.10, 0.6, size=y_pred.shape), color='blue', linewidth=4,
            label='Predicted Ra')
plt.legend()
plt.xlim([0, 0.5])
plt.ylim([0, 1])
plt.title("Comparison of observed and predicted Ra", fontsize=20)
plt.xlabel('Observed Ra [-], Predicted Ra [-]', fontsize=16)
plt.ylabel('random uniform distribution [-]', fontsize=16)
plt.savefig(f'regressor_region_insights.png', dpi=600)

plt.figure(figsize=(15, 8))
plt.scatter(y_pred, y_test, color='blue', label='prediction')
plt.plot(np.linspace(0, 0.5, 3), np.linspace(0, 0.5, 3), color='red', linewidth=10, label='error free prediction')
plt.legend()
plt.xlim([0, 0.5])
plt.ylim([0, 0.5])
plt.xlabel('Observed quality [-]', fontsize=16)
plt.ylabel('Predicted quality [-]', fontsize=16)
plt.savefig(f'regressor_prediction_insights.png', dpi=300)

# plot of confusion matrix regressor

df = pd.read_csv('only_regressor_test.csv')

is_observation_in_limits = (df['Ra observed'] > LOWER_SPECIFICATION_LIMIT) & (
        df['Ra observed'] < UPPER_SPECIFICATION_LIMIT)
is_prediction_in_limits = (df['Ra predicted'] > LOWER_SPECIFICATION_LIMIT) & (
        df['Ra predicted'] < UPPER_SPECIFICATION_LIMIT)

df['actual Ra'] = np.where(is_observation_in_limits, 0, 1)
df['predicted Ra'] = np.where(is_prediction_in_limits, 0, 1)

y_true = df['actual Ra'].values
y_pred = df['predicted Ra'].values

#Metrics of  the model
confusion_matrix = confusion_matrix(y_true, y_pred)
accuracy_model = accuracy_score(y_true, y_pred)
recall_model = recall_score(y_true, y_pred.round())
precision_model = precision_score(y_true, y_pred.round())
report = classification_report(y_true, y_pred)

print('\n')
print(f'accuracy_model  : {round(accuracy_model * 100, 2)}')
print(f'recall_model    : {round(recall_model * 100, 2)}')
print(f'precision_model : {round(precision_model * 100, 2)}')
print(f"confusion matrix: \n{confusion_matrix}")

with open('from_reg_metrics_summary.txt', 'w+') as file:
    file.write("################### RESULT OF PREDICTION ###################\n\n")
    file.write(f" accuracy  : {round(accuracy_model * 100, 2)} %\n")
    file.write(f" Recall    : {round(recall_model * 100, 2)} %\n")
    file.write(f" Precision : {round(precision_model * 100, 2)} %\n")
    file.write(f" F1-Score  : {report} %\n")

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
plt.savefig(f'from_reg_conf_matrix.png', dpi=300)
