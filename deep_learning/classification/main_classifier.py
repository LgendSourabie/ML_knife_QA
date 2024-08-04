from hyperparameter_tuning_summary import get_best_parameter
from classification_model_build import DeepClassifierModel
from feature_label import get_split_dataset, LOWER_SPECIFICATION_LIMIT, UPPER_SPECIFICATION_LIMIT
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score,f1_score
import tensorflow as tf
import numpy as np
import pandas as pd



X_train, X_test, y_train, y_test = get_split_dataset(regressor=False,min_max_scaler=True)


classifier = DeepClassifierModel(input_dim=X_train.shape[1])
best_parameters = get_best_parameter()

######## take secon model

model = classifier.build_model(best_parameters[0])

earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,  
                          verbose = 1,
                          restore_best_weights = True) 

checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint.model.keras",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

callbacks = [earlystop, checkpoint]

history = model.fit(X_train, y_train, batch_size=8,callbacks=callbacks, epochs=15, validation_split=0.15)

# prediction

y_pred = model.predict(X_test)
y_pred = np.reshape(y_pred, len(y_pred))

# save the metrics
dic_metric = {'Quality observed':y_test,'Quality predicted':y_pred} 
prediction = pd.DataFrame(data=dic_metric).to_csv('output_only_test_set.csv')

# Evaluation of the model

score = model.evaluate(X_test, y_test, verbose=0)

#Metrics of  the model

matrix_conf=confusion_matrix(y_test, y_pred.round())
accuracy_model=accuracy_score(y_test, y_pred.round())
recall_model=recall_score(y_test, y_pred.round())
precision_model=precision_score(y_test, y_pred.round())
f1_model=f1_score(y_test, y_pred.round())
print('\n')
print(f'Test loss       : {round(score[0],5)}')
print(f'accuracy_model  : {round(accuracy_model*100,2)}')
print(f'recall_model    : {round(recall_model*100,2)}')
print(f'precision_model : {round(precision_model*100,2)}')
print(f'f1_model        : {round(f1_model*100,2)}')
print(f"confusion matrix: \n{matrix_conf}")


with open('metrics_eval_summary.txt', 'w+') as file:
    file.write("################### RESULT OF PREDICTION ###################\n\n")
    file.write(f" Test loss : {round(score[0],5)}\n")
    file.write(f" accuracy  : {round(accuracy_model*100,2)} %\n")
    file.write(f" Recall    : {round(recall_model*100,2)} %\n")
    file.write(f" Precision : {round(precision_model*100,2)} %\n")
    file.write(f" F1-Score  : {round(f1_model*100,2)} %\n")
 

plt.figure(figsize=(15, 8))

plt.subplot(221)
plt.plot(history.history['loss'], label='Training loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation loss', linewidth=2)
plt.title('Losses', fontsize=18)
plt.ylabel('Loss [-]', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(222)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Accuracies', fontsize=18)
plt.ylabel('Accuracy [-]', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(223)
plt.plot(history.history['precision'], label='Training precision', linewidth=2)
plt.plot(history.history['val_precision'], label='Validation precision', linewidth=2)
plt.plot(history.history['recall'], label='Training recall', linewidth=2)
plt.plot(history.history['val_recall'], label='Validation recall', linewidth=2)
plt.title('Precisions vs. Recalls', fontsize=18)
plt.ylabel('precision & recall [-]', fontsize=12)
plt.xlabel('Epoch [-]', fontsize=12)
plt.legend(fontsize=12)

precision = np.array(history.history['precision'])
recall = np.array(history.history['recall'])
val_precision = np.array(history.history['recall'])
val_recall = np.array(history.history['recall'])

f1_score = 2 * (precision*recall)/(precision + recall)

val_f1_score = 2 * (val_precision * val_recall)/(val_precision + val_recall)

plt.subplot(224)
plt.plot(f1_score, label='Training F1_Score', linewidth=2)
plt.plot(val_f1_score, label='Validation F1_Score', linewidth=2)
plt.title('F1_Score', fontsize=18)
plt.ylabel('f1_score [-]', fontsize=12)
plt.xlabel('Epoch [-]', fontsize=12)
plt.legend(fontsize=12)
plt.savefig(f'performance_classifier.png', dpi=300)

