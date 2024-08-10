from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from classification_model_build import DeepClassifierModel
from feature_label import get_split_dataset
from scipy.stats import uniform, randint

X_train, X_test, y_train, y_test = get_split_dataset(regressor=False, min_max_scaler=False)
"""#random search for hyperparameter tuning"""


#########################################################
def create_model(activate, hidden_1, hidden_2, hidden_3, drop, lr):
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(hidden_1, input_dim=X_train.shape[1], activation=activate))
    model.add(tf.keras.layers.Dropout(rate=drop))

    model.add(tf.keras.layers.Dense(hidden_2, activation=activate))
    model.add(tf.keras.layers.Dropout(rate=drop))

    model.add(tf.keras.layers.Dense(hidden_3, activation=activate))
    model.add(tf.keras.layers.Dropout(rate=drop))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])
    # history = model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.2, epochs=epochs, verbose=1)
    return model


#########################################################

# classifier = DeepClassifierModel(X_train.shape[1])

# model = KerasClassifier(model=create_model, epochs=50, batch_size=16, verbose=1)

# # Define hyperparameters grid
# param_grid = {
#     'model__hidden_1': randint(32, 256),
#     'model__hidden_2': randint(32, 256),
#     'model__hidden_3': randint(32, 256),
#     'model__activate': ['relu', 'tanh', 'leaky_relu', 'selu'],
#     'model__drop': uniform(0.2, 0.3),
#     'model__lr': [1e-3, 1e-4, 1e-5],
# }

# Set up GridSearchCV with KFold cross-validation
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=5, scoring='accuracy',
#                                    cv=kfold,
#                                    n_jobs=-1)
#
# # Execute the grid search
# random_result = random_search.fit(X_train, y_train)
#
# # Summarize the results
# print(f"Best: {random_result.best_score_} using {random_result.best_params_}")
# means = random_result.cv_results_['mean_test_score']
# stds = random_result.cv_results_['std_test_score']
# params = random_result.cv_results_['params']
# for mean, std, param in zip(means, stds, params):
#     print(f"{mean:.4f} (+/-{std:.4f}) with: {param}")
#
# # Evaluate the best model on the test set
# best_model = random_result.best_estimator_
# y_pred = (best_model.predict(X_test) > 0.5).astype(int)
#
# print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(f"Test Precision: {precision_score(y_test, y_pred):.4f}")
# print(f"Test Recall: {recall_score(y_test, y_pred):.4f}")
#
# with open('random_search_result.txt', 'w+') as file:
#     file.write("################### RESULT OF PREDICTION ###################\n\n")
#     file.write(f" Test loss : {random_result.best_params_}\n")
#     file.write(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
#     file.write(f"Test Precision: {precision_score(y_test, y_pred):.4f}\n")
#     file.write(f"Test Recall: {recall_score(y_test, y_pred):.4f}\n")
#     for i, (mean, std, param) in enumerate(zip(means, stds, params)):
#         file.write(f"Mean{i}:{mean * 100:.2f} % Std{i}:(+/-{std * 100:.4f} %) with: {param}\n")
