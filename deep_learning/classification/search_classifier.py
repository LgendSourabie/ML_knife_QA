from class_gridsearchCV import create_model
from feature_label import get_split_dataset
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score

X_train, X_test, y_train, y_test = get_split_dataset(regressor=False, min_max_scaler=False)

best_params = {'activate': 'relu', 'drop': 0.2, 'hidden_1': 183, 'hidden_2': 160, 'hidden_3': 200,
               'lr': 0.00001}

model = create_model(**best_params)

checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint.model.keras",
                                                monitor="val_loss",
                                                mode="min",
                                                save_best_only=True,
                                                verbose=1)

callbacks = [checkpoint]

history = model.fit(X_train, y_train, batch_size=16, callbacks=callbacks, epochs=200, validation_split=0.2, verbose=1)

y_pred = (model.predict(X_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy * 100:.2f}")
print(f"Test Precision: {precision * 100:.2f}")
print(f"Test Recall: {recall * 100:.2f}")

with open('class_perfo_search.txt', 'w+') as file:
    file.write("################### RESULT OF PREDICTION ###################\n\n")
    file.write(f"Test Accuracy: {accuracy * 100:.2f}\n")
    file.write(f"Test Precision: {precision * 100:.2f}\n")
    file.write(f"Test Recall: {recall * 100:.2f}\n")

plt.figure(figsize=(15, 8))

plt.subplot(121)
plt.plot(history.history['loss'], label='Training loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation loss', linewidth=2)
plt.title('Losses', fontsize=18)
plt.ylabel('Loss [-]', fontsize=12)
plt.legend(fontsize=12)

plt.subplot(122)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Accuracies', fontsize=18)
plt.ylabel('Accuracy [-]', fontsize=12)
plt.legend(fontsize=12)

plt.savefig(f'perf_class.png', dpi=300)
