
from hyperparameter_tuning_summary import get_best_hyperparameter
from classification_model_build import DeepClassifierModel, set_activation
from feature_label import get_split_dataset
from prettytable import PrettyTable
import tensorflow as tf
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_validate


X_train, X_test, y_train, y_test = get_split_dataset(regressor=False,min_max_scaler=True)


deep_classifier_model = DeepClassifierModel(X_train.shape[1])

best_hyperparameter = get_best_hyperparameter()  # get the best hyperparams from the tuning process done in hyperparameter_tuning_summary


def create_model(best_hyperparameter=best_hyperparameter):
      """
      create a NN architecture for the classifiction task
      hp: hyperparameter for hyperparameter tuning
      """
      model = tf.keras.models.Sequential()

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_1'],activation=set_activation(best_hyperparameter['activation_1']),input_dim=X_train.shape[1]))
      tf.keras.regularizers.L2(best_hyperparameter['l2_1'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_2'],activation=set_activation(best_hyperparameter['activation_2'])))
      tf.keras.regularizers.L2(best_hyperparameter['l2_2'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_3'],activation=set_activation(best_hyperparameter['activation_3'])))
      tf.keras.regularizers.L2(best_hyperparameter['l2_3'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_4'],activation=set_activation(best_hyperparameter['activation_4'])))
      tf.keras.regularizers.L2(best_hyperparameter['l2_4'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_5'],activation=set_activation(best_hyperparameter['activation_5'])))
      tf.keras.regularizers.L2(best_hyperparameter['l2_5'])

      model.add(tf.keras.layers.Dense(units=best_hyperparameter['hidden_6'],activation=set_activation(best_hyperparameter['activation_6'])))
      tf.keras.regularizers.L2(best_hyperparameter['l2_6'])

      model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

      model.compile(optimizer = deep_classifier_model.set_optimizer(best_hyperparameter['optimizer'],best_hyperparameter['learning_rate']),
                    loss='binary_crossentropy',metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
      return model



# # Create the KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=1)


# we take 10 Folds for the project
cv_folds = 10

scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}
results = cross_validate(estimator=model, X=X_train, y=y_train, cv=cv_folds, scoring=scoring, n_jobs=-1, return_train_score=True)


accuracy_scores = results['test_accuracy']
precision_scores = results['test_precision']
recall_scores = results['test_recall']
f1_scores = results['test_f1']

# mean and standard deviation for each metric
mean_accuracy = np.mean(accuracy_scores)*100
std_accuracy = np.std(accuracy_scores)*100
mean_precision = np.mean(precision_scores)*100
std_precision = np.std(precision_scores)*100
mean_recall = np.mean(recall_scores)*100
std_recall = np.std(recall_scores)*100
mean_f1 = np.mean(f1_scores)*100
std_f1 = np.std(f1_scores)*100


table = PrettyTable()
table.add_column('Folds',list(np.arange(1,cv_folds+1)))
table.add_column('Accuracy [%]',list(np.round(accuracy_scores*100,2)))
table.add_column('Precision [%]',list(np.round(precision_scores*100,2)))
table.add_column('Recall [%]',list(np.round(recall_scores*100,2)))
table.add_column('F1Score [%]',list(np.round(f1_scores*100,2)))


for index, (acc,pre,rec,f1) in enumerate(zip(accuracy_scores,precision_scores,recall_scores,f1_scores)):
    print(f'Fold {index+1}: Accuracy= {round(acc,2)} Precision= {round(pre,2)} Recall= {round(rec,2)} F1Score= {round(f1,2)}')



print('\n\n')
print(f'Mean Accuracy: {mean_accuracy:.2f} %')
print(f'Standard Deviation of Accuracy: {std_accuracy:.2f} %')
print(f'Mean Precision: {mean_precision:.2f} %')
print(f'Standard Deviation of Precision: {std_precision:.2f} %')
print(f'Mean Recall: {mean_recall:.2f} %')
print(f'Standard Deviation of Recall: {std_recall:.2f} %')
print(f'Mean F1: {mean_f1:.2f} %')
print(f'Standard Deviation of F1: {std_f1:.2f} %')


with open('cross_validation_summary.txt', 'w+') as file:

    file.write(f"################### RESULT OF {cv_folds}-Folds CROSS VALIDATION ###################\n\n")
    file.write(f"{table}")
    file.write(f'\n\nMean Accuracy: {mean_accuracy:.2f} %\n')
    file.write(f'Standard Deviation of Accuracy: {std_accuracy:.2f} %\n')
    file.write(f'Mean Precision: {mean_precision:.2f} %\n')
    file.write(f'Standard Deviation of Precision: {std_precision:.2f} %\n')
    file.write(f'Mean Recall: {mean_recall:.2f} %\n')
    file.write(f'Standard Deviation of Recall: {std_recall:.2f} %\n')
    file.write(f'Mean F1: {mean_f1:.2f} %\n')
    file.write(f'Standard Deviation of F1: {std_f1:.2f} %\n')
    