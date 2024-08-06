import keras_tuner as kt
from classification_model_build import DeepClassifierModel


directory = "hyperparameter_tuning_classifier"
project_name = "knife_classifier_2"


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



def get_best_models(rank_index=0):
    models = tuner.get_best_models(num_models=rank_index + 2)
    return models[rank_index].summary()

def get_tuning_summary():
    return tuner.results_summary()

def get_best_parameter():
    return tuner.get_best_hyperparameters(4)

def get_best_hyperparameter(var = tuner):
    best_params = var.get_best_hyperparameters(5)[0].values
    return best_params

get_best_models()
get_tuning_summary()
get_best_hyperparameter()

