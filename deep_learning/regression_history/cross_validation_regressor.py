
from hyperparameter_tuning_summary import get_best_hyperparameter
from regressor_model_build import DeepRegressionModel, set_activation
from feature_label import get_split_dataset
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_validate
from prettytable import PrettyTable


X_train, X_test, y_train, y_test = get_split_dataset(regressor=True,min_max_scaler=False)


deep_regressor_model = DeepRegressionModel(X_train.shape[1])

best_hyperparameter = get_best_hyperparameter()  # get the best hyperparams from the tuning process done in hyperparameter_tuning_summary


def create_model(best_hyperparameter=best_hyperparameter):
      """
      create a NN architecture for the classifiction task
      hp: hyperparameter for hyperparameter tuning
      """
      model = tf.keras.models.Sequential()

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_1'],activation=set_activation(best_hyperparameter['activation_1']),input_dim=X_train.shape[1]))
      tf.keras.regularizers.L2(best_hyperparameter['l2_1'])

    # Number of hidden layers
      num_layers = best_hyperparameter['num_layers'] 
        
      for i in range(2, num_layers + 1):

          model.add(tf.keras.layers.Dense(units=best_hyperparameter[f'hidden_{i}'],activation=set_activation(best_hyperparameter[f'activation_{i}'])))
          tf.keras.regularizers.L2(best_hyperparameter[f'l2_{i}'])

      model.add(tf.keras.layers.Dense(units= 1, activation =set_activation(best_hyperparameter['activation_out'])))

      model.compile(optimizer = deep_regressor_model.set_optimizer(best_hyperparameter['optimizer'],best_hyperparameter['learning_rate']) ,
                    loss='mean_squared_error',metrics=[tf.keras.metrics.R2Score(name='r_squared'), tf.keras.metrics.MeanSquaredError(name='mean_squared_error')])
      return model


# # Create the KerasRegressor
model = KerasRegressor(build_fn=create_model, verbose=1)


# we take 10 Folds for the project
cv_folds = 10
scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
results = cross_validate(estimator=model, X=X_train, y=y_train, cv=cv_folds, scoring=scoring, n_jobs=-1, return_train_score=True)

r2_scores = results['test_r2']*100
mse_scores = results['test_mse']*100  

mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
mean_mse = -np.mean(mse_scores)  # we Convert negative MSE to positive, since use neg_mean_square_error
std_mse = np.std(mse_scores)

table = PrettyTable()
table.add_column('Folds',list(np.arange(1,cv_folds+1)))
table.add_column('R² Score [%]',list(np.round(r2_scores,2)))
table.add_column('MSE [%]',list(np.round(mse_scores,2)))


print(f"R² scores for each fold: {np.round(r2_scores,2)}")
print(f"MSE scores for each fold: {np.round(-mse_scores,2)}")


for index, (score_val,mse_val) in enumerate(zip(r2_scores,-mse_scores)):
    print(f'Fold {index+1}: R²= {score_val:.2f} MSE= {mse_val:.2f}')


print('\n\n')
print(f'Mean R²: {mean_r2:.2f} %')
print(f'Standard Deviation of R²: {std_r2:.2f} %')
print(f'Mean MSE: {mean_mse:.2f} %')
print(f'Standard Deviation of MSE: {std_mse:.2f} %')

with open('cross_validation_summary.txt', 'w+') as file:

    file.write(f"################### RESULT OF {cv_folds}-Folds CROSS VALIDATION ###################\n\n")

    file.write(f"{table}")
    file.write(f'\n\nMean R²: {mean_r2:.2f} %\n')
    file.write(f'Standard Deviation of R²: {std_r2:.2f} %\n')
    file.write(f'Mean MSE: {mean_mse:.2f} %\n')
    file.write(f'Standard Deviation of MSE: {std_mse:.2f} %\n')
        