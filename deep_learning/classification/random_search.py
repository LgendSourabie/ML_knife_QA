from classification_model_build import DeepClassifierModel
import keras_tuner as kt
from feature_label import get_split_dataset


X_train, X_test, y_train, y_test = get_split_dataset(regressor=False,min_max_scaler=True)
"""#random search for hyperparameter tuning"""

classifier = DeepClassifierModel(X_train.shape[1])

tuner = kt.RandomSearch(
hypermodel=classifier.build_model,
objective='val_accuracy',
max_trials=25,
executions_per_trial=1,
overwrite=False, # I set it to false to avoid any override of the existing tuning results
directory="hyperparameter_tuning_classifier",
project_name="knife_classifier_2",
)


tuner.search(X_train, y_train, epochs=100, validation_split=0.2)


models = tuner.get_best_models(num_models=2)
models[0].summary()

tuner.results_summary()

best_params = tuner.get_best_hyperparameters()[0].values

print(best_params)
