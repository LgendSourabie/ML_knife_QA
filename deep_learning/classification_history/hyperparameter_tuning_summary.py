import keras_tuner as kt
from classification_model_build import DeepClassifierModel
from feature_label import  get_split_dataset

X_train, _, _, _ = get_split_dataset(regressor=False, min_max_scaler=False)

directory = "hyperparameter_tuning_classifier"
project_name = "knife_classifier"

classifier = DeepClassifierModel(X_train.shape[1])

tuner = kt.RandomSearch(
    hypermodel=classifier.build_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=1,
    overwrite=False, # we don't want to overwrite the existing parameter
    directory=directory,
    project_name=project_name,
)


def get_best_models(rank_index=0):
    """
    get the best model and give a detailed summary of the model
    :param rank_index:
    :return: give the architecture of the best model
    """
    models = tuner.get_best_models(num_models=rank_index + 2)
    return models[rank_index].summary()


def get_tuning_summary():
    """
    get the summary of hyperparameter tuning
    :return: the best 10 models with their architecture and hyperparameters
    """
    return tuner.results_summary()


def get_best_parameter():
    """
    get the four best model
    :return: four best models for fittin purpose
    """
    return tuner.get_best_hyperparameters(4)


def get_best_hyperparameter(var=tuner):
    """
    get the best hyperparameter
    :param var: tuner - the hyperparameter tuner
    :return: best model parameters as a dictionary
    """
    best_params = var.get_best_hyperparameters(5)[0].values
    return best_params


get_best_models()
get_tuning_summary()
get_best_hyperparameter()
