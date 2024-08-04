import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder
from regressor_model_build import DeepRegressionModel
import keras_tuner as kt
import matplotlib.pyplot as plt
from prettytable import PrettyTable

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
y = dataset['Ra'].values

"""#Encoding categorical data"""

le = LabelEncoder()
y = le.fit_transform(y)


"""#Splitting dataset into training and test set"""

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=1)

"""#Scaling the features"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""#random search for hyperparameter tuning"""

directory = "hyperparameter_tuning_regressor"
project_name = "knife_regressor"


regressor = DeepRegressionModel(30)

tuner = kt.RandomSearch(
    hypermodel=regressor.build_model,
    objective=kt.Objective("val_r_squared", direction="max"),
    max_trials=25,
    executions_per_trial=1,
    overwrite=False,  # I put False not to override the previous data
    directory=directory,
    project_name=project_name,
)


# callback = EarlyStopping(monitor='val_loss', patience=5)
# tuner.search(X_train, y_train, batch_size=16, epochs=15, validation_split=0.1, callbacks=[callback])
tuner.search(X_train, y_train, batch_size=8, epochs=15, validation_split=0.15)


models = tuner.get_best_models(num_models=2)
models[0].summary()

tuner.results_summary()

best_params = tuner.get_best_hyperparameters(4)

print(best_params[0].values)

def save_in_table(best_parameters):

    table = PrettyTable()
    table.add_column('Parameter names',list(best_parameters[0].values.keys()))
    table.add_column('1st best hyperparameters',list(best_parameters[0].values.values()))
    table.add_column('2nd best hyperparameters',list(best_parameters[1].values.values()))
    table.add_column('3rd best hyperparameters',list(best_parameters[2].values.values()))
    table.add_column('4th best hyperparameters',list(best_parameters[3].values.values()))
    table.add_row(['END','END','END','END','END'],divider=True)
    table.add_row(['BEST SCORES',1,5,7,8])

    # write report file for checking purpose, data also available
    with open('best_hyperparameters.txt', 'w+') as file:
        file.write(f"##################### REPORT OF HYPERPARAMETER TUNING: RANDOM SEARCH #####################\n\n")
        file.write(f"BEST PARAMETERS:\n {table}\n\n")
    return 

save_in_table(best_parameters=best_params)

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
