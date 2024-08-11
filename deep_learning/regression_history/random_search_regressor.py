from regressor_model_build import DeepRegressionModel
import keras_tuner as kt
from feature_label import get_split_dataset

X_train, X_test, y_train, y_test = get_split_dataset(regressor=True,min_max_scaler=False)


directory = "hyperparameter_tuning_regressor"
project_name = "knife_regressor"


regressor = DeepRegressionModel(X_train.shape[1])

tuner = kt.RandomSearch(
    hypermodel=regressor.build_model,
    objective=kt.Objective("val_r_squared", direction="max"),
    max_trials=10,
    executions_per_trial=1,
    overwrite=False,  # I put False not to override the previous data
    directory=directory,
    project_name=project_name,
)


tuner.search(X_train, y_train, batch_size=64, epochs=150, validation_split=0.2)


models = tuner.get_best_models(num_models=2)
models[0].summary()

tuner.results_summary()

best_params = tuner.get_best_hyperparameters(4)

print(best_params[0].values)
