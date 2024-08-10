from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from feature_label import LOWER_SPECIFICATION_LIMIT, UPPER_SPECIFICATION_LIMIT
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('output_only_test_set.csv')

is_observation_in_limits = (df['Ra observed'] > LOWER_SPECIFICATION_LIMIT) & (
        df['Ra observed'] < UPPER_SPECIFICATION_LIMIT)
is_prediction_in_limits = (df['Ra predicted'] > LOWER_SPECIFICATION_LIMIT) & (
        df['Ra predicted'] < UPPER_SPECIFICATION_LIMIT)

df['actual Ra'] = np.where(is_observation_in_limits, 0, 1)
df['predicted Ra'] = np.where(is_prediction_in_limits, 0, 1)

y_true = df['actual Ra'].values
y_pred = df['predicted Ra'].values

#Metrics of  the model
confusion_matrix = confusion_matrix(y_true, y_pred)
accuracy_model = accuracy_score(y_true, y_pred)
recall_model = recall_score(y_true, y_pred.round())
precision_model = precision_score(y_true, y_pred.round())
report = classification_report(y_true, y_pred)

print('\n')
print(f'accuracy_model  : {round(accuracy_model * 100, 2)}')
print(f'recall_model    : {round(recall_model * 100, 2)}')
print(f'precision_model : {round(precision_model * 100, 2)}')
print(f"confusion matrix: \n{confusion_matrix}")

with open('from_reg_metrics_summary.txt', 'w+') as file:
    file.write("################### RESULT OF PREDICTION ###################\n\n")
    file.write(f" accuracy  : {round(accuracy_model * 100, 2)} %\n")
    file.write(f" Recall    : {round(recall_model * 100, 2)} %\n")
    file.write(f" Precision : {round(precision_model * 100, 2)} %\n")
    file.write(f" F1-Score  : {report} %\n")

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
plt.savefig(f'from_reg_conf_matrix.png', dpi=300)
