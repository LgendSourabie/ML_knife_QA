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



def get_best_models(rank_index=0):
    models = tuner.get_best_models(num_models=rank_index + 2)
    return models[rank_index].summary()

def get_tuning_summary():
    return tuner.results_summary()
    

def get_best_hyperparameter(var = tuner):
    best_params = var.get_best_hyperparameters()[0].values
    return best_params

print(get_best_models())
print(get_tuning_summary())
print(get_best_hyperparameter())