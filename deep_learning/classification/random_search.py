import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from model_build import DeepClassifierModel

import matplotlib.pyplot as plt

"""#Importing data"""

dataset = pd.read_excel('../../data/chiefs_knife_dataset.xlsx')
index_Ra = dataset.columns.get_loc('Ra')  # index of the surface roughness column for inserting the class. label

LOWER_SPECIFICATION_LIMIT = 0.125  # lower bound of good quality product region
UPPER_SPECIFICATION_LIMIT = 0.215  # upper bound of good quality product region

is_between_specification_bounds = (dataset['Ra'] >= LOWER_SPECIFICATION_LIMIT) & (dataset['Ra'] < UPPER_SPECIFICATION_LIMIT)
good_product_range = np.where(is_between_specification_bounds, "good", "bad")
dataset.insert(index_Ra + 1, 'Quality', good_product_range) 


"""# constructing Label"""

X = dataset.loc[:,'Original_Linienanzahl':'DFT_Median_sobel_Bereich'].values
y = dataset['Quality'].values

"""#Encoding categorical data"""

le = LabelEncoder()
y = le.fit_transform(y)


"""#Splitting dataset into training and test set"""

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=1)

"""#Scaling the features"""

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""#Hyperparameter definition"""

import keras_tuner as kt
# from keras_tuner import HyperParameters

# neurons_max = 1024
# neurons_min = 32
# neurons_step = 16
# dropout_min = 0.0
# dropout_max = 0.5
# dropout_step = 0.1
# l2_min = 0.0
# l2_max = 0.1
# l2_step = 0.01
# learning_rates = [1e-2, 1e-3, 1e-4,1e-5]
# activations = ['sigmoid', 'tanh']

"""#random search for hyperparameter tuning"""

classifier = DeepClassifierModel(X_train.shape[1])

tuner = kt.RandomSearch(
hypermodel=classifier.build_model,
objective='val_accuracy',
max_trials=25,
executions_per_trial=1,
overwrite=False, # I set it to false to avoid any override of the existing tuning results
directory="hyperparameter_tuning_classifier",
project_name="knife_classifier",
)

# callback = EarlyStopping(monitor='val_loss', patience=5)
# tuner.search(X_train, y_train, batch_size=16, epochs=15, validation_split=0.1, callbacks=[callback])
tuner.search(X_train, y_train, batch_size=8, epochs=15, validation_split=0.1)


models = tuner.get_best_models(num_models=2)
models[0].summary()

tuner.results_summary()

best_params = tuner.get_best_hyperparameters()[0].values

print(best_params)


#######################################################################

# best_model = tf.keras.models.Sequential()

# best_model.add(tf.keras.layers.Dense(units=288, activation='relu', input_dim=X_train.shape[1]))

# best_model.add(tf.keras.layers.Dense(units=464, activation='leaky_relu'))

# best_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# history = best_model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.1)

# # confusion matrix - Model evaluation

# matrix_conf = confusion_matrix(y_test, best_model.predict(X_test).round())
# accuracy_model = accuracy_score(y_test, best_model.predict(X_test).round())
# print(f"{matrix_conf=}")
# print(f'{accuracy_model=}')


# def show_loss_acc(h):
#     plt.figure(figsize=(12, 8))
#     plt.subplot(1, 2, 1)
#     plt.plot(h['loss'], label='train loss')
#     plt.plot(h['val_loss'], label='val loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(h['accuracy'], label='train accuracy')
#     plt.plot(h['val_accuracy'], label='val accuracy')
#     plt.legend()
#     plt.savefig('loss.png', dpi=300)


# show_loss_acc(history.history)
