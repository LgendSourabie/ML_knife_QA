
from hyperparameter_tuning_summary import get_best_hyperparameter
from classification_model_build import DeepClassifierModel
from feature_label import get_split_dataset
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_validate


X_train, X_test, y_train, y_test = get_split_dataset(regressor=False,min_max_scaler=True)


deep_classifier_model = DeepClassifierModel(X_train.shape[1])

best_hyperparameter = get_best_hyperparameter()  # get the best hyperparams from the tuning process done in hyperparameter_tuning_summary


def create_model(best_hyperparameter=best_hyperparameter):
      """
      create a NN architecture for the classifiction task
      hp: hyperparameter for hyperparameter tuning
      """
      model = tf.keras.models.Sequential()

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_1'],activation=tf.keras.layers.ELU(),input_dim=X_train.shape[1]))
      tf.keras.regularizers.L2(best_hyperparameter['l2_1'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_2'],activation=best_hyperparameter['activation_2']))
      tf.keras.regularizers.L2(best_hyperparameter['l2_2'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_3'],activation=best_hyperparameter['activation_3']))
      tf.keras.regularizers.L2(best_hyperparameter['l2_3'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_4'],activation=best_hyperparameter['activation_4']))
      tf.keras.regularizers.L2(best_hyperparameter['l2_4'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_5'],activation=tf.keras.layers.ELU()))
      tf.keras.regularizers.L2(best_hyperparameter['l2_5'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_6'],activation=best_hyperparameter['activation_6']))
      tf.keras.regularizers.L2(best_hyperparameter['l2_6'])

      model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

      model.compile(optimizer =tf.keras.optimizers.Adam(learning_rate=best_hyperparameter['learning_rate']) ,
                    loss='binary_crossentropy',metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
      return model


# # Create the KerasRegressor
model = KerasRegressor(build_fn=create_model, verbose=1)


# we take 10 Folds for the project
cv_folds = 10
scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
results = cross_validate(estimator=model, X=X_train, y=y_train, cv=cv_folds, scoring=scoring, n_jobs=-1, return_train_score=True)

r2_scores = results['test_r2']
mse_scores = results['test_mse']  

mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

mean_mse = -np.mean(mse_scores)  # we Convert negative MSE to positive, since use neg_mean_square_error
std_mse = np.std(mse_scores)


print("R² scores for each fold:", r2_scores)
print("MSE scores for each fold:", -mse_scores)


for index, (score_val,mse_val) in enumerate(zip(r2_scores,-mse_scores)):
    print(f'Fold {index+1}: R²= {round(score_val,2)} MSE= {round(mse_val,6)}')


print('\n\n')
print(f'Mean R²: {round(mean_r2,4)}')
print(f'Standard Deviation of R²: {round(std_r2,5)}')
print(f'Mean MSE: {round(mean_mse,6)}')
print(f'Standard Deviation of MSE: {round(std_mse,6)}')

with open('cross_validation_summary.txt', 'w+') as file:

    file.write(f"################### RESULT OF {cv_folds}-Folds CROSS VALIDATION ###################\n\n")
    for index, (score_val,mse_val) in enumerate(zip(r2_scores,-mse_scores)):
         file.write(f'Fold {index+1}: R²= {round(score_val,2)} MSE= {round(mse_val,6)}')

    file.write(f'\n\nMean R²: {round(mean_r2,4)}')
    file.write(f'Standard Deviation of R²: {round(std_r2,5)}')
    file.write(f'Mean MSE: {round(mean_mse,6)}')
    file.write(f'Standard Deviation of MSE: {round(std_mse,6)}')
    