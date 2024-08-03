import pandas as pd
import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from model_build import DeepClassifierModel
from hyperparameter_tuning_summary import get_best_hyperparameter


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

deep_classifier_model = DeepClassifierModel(X_train.shape[1])

best_hyperparameter = get_best_hyperparameter()  # get the best hyperparams from the tuning process done in hyperparameter_tuning_summary


def create_model(best_hyperparameter=best_hyperparameter):
    """
    create a NN architecture for the classifiction task
    hp: hyperparameter for hyperparameter tuning
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_1'],activation=best_hyperparameter['activation_1'],input_dim=X_train.shape[1]))

    model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_2'], activation=best_hyperparameter['activation_2']))
    model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_3'], activation=best_hyperparameter['activation_3']))
    model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_4'], activation=best_hyperparameter['activation_4']))
    model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_5'], activation=best_hyperparameter['activation_5']))
    model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_6'], activation=best_hyperparameter['activation_6']))

    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=deep_classifier_model.set_optimizer(best_hyperparameter['optimizer'], best_hyperparameter['learning_rate']),loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model



# Create the KerasRegressor
model = KerasRegressor(build_fn=create_model, verbose=1)

# Define the grid search parameters
search_parameter = {
    'batch_size': [8],
    'epochs': [50,100],
}


# we perform here a grid search for 5-folds
grid = GridSearchCV(estimator=model, param_grid=search_parameter, n_jobs=-1, cv=3)


grid_result = grid.fit(X_train, y_train)



best_accuracy = grid_result.best_score_
best_parameters = grid_result.best_params_
best_index = grid_result.best_index_
best_cv_result = grid_result.cv_results_
number_split = grid_result.n_splits_

print(f"##################### Result of {number_split}-Fold Cross validation #####################\n\n")
print("BEST ACCURACY: {:.2f} %".format(best_accuracy*100))
print("BEST PARAMETER:", best_parameters)
print("Best Accuracy: %f with the following Parameters %s" % (best_accuracy, best_parameters))

with open('best_hyperparameters.txt', 'w+') as file:
    file.write(f"##################### Result of {number_split}-Fold Cross validation #####################\n\n")
    file.write("BEST ACCURACY: {:.2f} %".format(best_accuracy*100))
    file.write(f"BEST PARAMETERS: {best_parameters}")
    file.write(f"BEST INDEX: {best_index}\n")
    file.write(f"BEST CV RESULT: \n {best_cv_result}\n")
    file.write("Best Accuracy: %f with the following Parameters %s" % (best_accuracy, best_parameters))
    file.close()
