import keras_tuner as kt
from model_build import DeepClassifierModel


directory = "hyperparameter_tuning_classifier"
project_name = "knife_classifier"


classifier = DeepClassifierModel(30)

tuner = kt.RandomSearch(
    hypermodel=classifier.build_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=1,
    overwrite=False,  
    directory=directory,
    project_name=project_name,
)


models = tuner.get_best_models(num_models=2)
models[0].summary()

tuner.results_summary()

best_params = tuner.get_best_hyperparameters()[0].values

print(best_params)
