import keras_tuner as kt
from regressor_model_build import DeepRegressionModel


directory = "hyperparameter_tuning_regressor"
project_name = "knife_regressor_10"


regressor = DeepRegressionModel(30)

tuner = kt.RandomSearch(
    hypermodel=regressor.build_model,
    objective=kt.Objective("val_r_squared", direction="max"),
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
    best_params = var.get_best_hyperparameters(5)[1].values
    return best_params

get_best_models()
get_tuning_summary()
get_best_hyperparameter()
