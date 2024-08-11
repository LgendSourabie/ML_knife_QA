from hyperparameter_tuning_summary import get_best_parameter
from classification_model_build import DeepClassifierModel
from feature_label import get_split_dataset, LOWER_SPECIFICATION_LIMIT, UPPER_SPECIFICATION_LIMIT
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

X_train, X_test, y_train, y_test = get_split_dataset(regressor=False, min_max_scaler=False)

classifier = DeepClassifierModel(input_dim=X_train.shape[1])
best_parameters = get_best_parameter()[1]

######## take secon model

# we reduce the optimizer for better convergence and stability
best_parameters.values['learning_rate'] = 1e-5

model = classifier.build_model(best_parameters)  # we rebuild the model with the chosen hyperparameters

checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint.model.keras",
                                                monitor="val_loss",
                                                mode="min",
                                                save_best_only=True,
                                                verbose=1)

# the history of improvement of the model based on validation loss is stored
callbacks = [checkpoint]

history = model.fit(X_train, y_train, batch_size=64, callbacks=callbacks, epochs=155, validation_split=0.2)

# prediction
y_pred = model.predict(X_test)
y_pred = np.reshape(y_pred, len(y_pred))

# save the metrics
dic_metric = {'Quality observed': y_test, 'Quality predicted': y_pred}
prediction = pd.DataFrame(data=dic_metric).to_csv('output_only_test_set.csv')

# Evaluation of the model
score = model.evaluate(X_test, y_test, verbose=0)

#Metrics of  the model
confusion_matrix = confusion_matrix(y_test, y_pred.round())
accuracy_model = accuracy_score(y_test, y_pred.round())
recall_model = recall_score(y_test, y_pred.round())
precision_model = precision_score(y_test, y_pred.round())
f1_model = f1_score(y_test, y_pred.round())
print('\n')
print(f'Test loss       : {round(score[0], 5)}')
print(f'accuracy_model  : {round(accuracy_model * 100, 2)}')
print(f'recall_model    : {round(recall_model * 100, 2)}')
print(f'precision_model : {round(precision_model * 100, 2)}')
print(f'f1_model        : {round(f1_model * 100, 2)}')
print(f"confusion matrix: \n{confusion_matrix}")

with open('metrics_eval_summary.txt', 'w+') as file:
    file.write("################### RESULT OF PREDICTION ###################\n\n")
    file.write(f" Test loss : {round(score[0], 5)}\n")
    file.write(f" accuracy  : {round(accuracy_model * 100, 2)} %\n")
    file.write(f" Recall    : {round(recall_model * 100, 2)} %\n")
    file.write(f" Precision : {round(precision_model * 100, 2)} %\n")
    file.write(f" F1-Score  : {round(f1_model * 100, 2)} %\n")

plt.figure(figsize=(15, 8))

plt.subplot(121)
plt.plot(history.history['loss'], label='Training loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation loss', linewidth=2)
plt.title('Losses', fontsize=18)
plt.xlabel('Epochs [-]', fontsize=12)
plt.ylabel('Loss [-]', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(122)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Accuracies', fontsize=18)
plt.xlabel('Epochs [-]', fontsize=12)
plt.ylabel('Accuracy [-]', fontsize=12)
plt.legend(fontsize=12)

plt.savefig(f'performance_classifier.png', dpi=300)

plt.figure(figsize=(10, 7))
group_names = ["Bad and predicted as Bad", "Bad but predicted as Good", "Good but predicted as Bad",
               "Good and predicted as Good"]
group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
labels = [f"{v1}\n\n{v2}\n\n{v3}" for v1, v2, v3 in zip(group_counts, group_percentages, group_names)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(confusion_matrix, annot=labels, xticklabels=['Bad products', 'Good Products'],
            yticklabels=['Bad products', 'Good products'], fmt="", cmap='Blues')
plt.xlabel('Predicted values', fontsize=16)
plt.ylabel('Actual values', fontsize=16)
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
plt.savefig(f'confusion_matrix.png', dpi=300)
