from hyperparameter_tuning_summary import get_best_parameter
from regressor_model_build import DeepRegressionModel
from feature_label import get_split_dataset, UPPER_SPECIFICATION_LIMIT,LOWER_SPECIFICATION_LIMIT
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score



X_train, X_test, y_train, y_test = get_split_dataset(regressor=True,min_max_scaler=False)


regressor = DeepRegressionModel(input_dim=30)
best_parameters = get_best_parameter()

######## take secon model

model = regressor.build_model(best_parameters[0])

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

callbacks = [earlystop,checkpoint]

history = model.fit(X_train, y_train, batch_size=8,callbacks = callbacks,verbose=2, epochs=150, validation_split=0.15)

# prediction

y_pred = model.predict(X_test)
y_pred = np.reshape(y_pred, len(y_pred))



#evaluation of model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
r2_score = r2_score(y_true=y_test,y_pred=y_pred)

#Error of prediction
error_pred = y_test-y_pred

# save the metrics
dic_metric = {'Ra observed':y_test,'Ra predicted':y_pred,'Error':error_pred} 
prediction = pd.DataFrame(data=dic_metric).to_csv('output_only_test_set.csv')

with open('metrics_eval_summary.txt', 'w+') as file:
    file.write("################### RESULT OF PREDICTION ###################\n\n")
    file.write(f" Test loss: {round(score[0],5)}\n")
    file.write(f" Mean Square Error: {round(mse,5)}\n")
    file.write(f" R2Score: {round(r2_score*100,2)} %\n")



plt.figure(figsize=(15, 8))

plt.subplot(131)
plt.plot(history.history['loss'], label='Training loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation loss', linewidth=2)
plt.title('Losses', fontsize=20)
plt.ylabel('Loss [-]', fontsize=12)
plt.xlabel('Epoch [-]', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(132)
plt.plot(history.history['r_squared'], label='Training r_squared', linewidth=2)
plt.plot(history.history['val_r_squared'], label='Validation r_squared', linewidth=2)
plt.title('R2Sores', fontsize=20)
plt.ylabel('R Squared [-]', fontsize=12)
plt.xlabel('Epoch [-]', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(133)
plt.plot(history.history['mean_squared_error'], label='Training mse', linewidth=2)
plt.plot(history.history['val_mean_squared_error'], label='Validation mse', linewidth=2)
plt.title('Mean Squared Error', fontsize=20)
plt.ylabel('MSE [-]', fontsize=12)
plt.xlabel('Epoch [-]', fontsize=12)
plt.legend(fontsize=12)
plt.savefig('performance_regressor.png', dpi=300)


plt.figure(figsize=(15, 8))
plt.plot(range(0,len(y_test)), y_test, '-b', label="Observed Ra")
plt.plot(range(0,len(y_pred)), y_pred, '-g', label="Predicted Ra")
plt.plot(range(0,len(y_pred)), error_pred, '-k', label="Error")
plt.title('Ra Prediction Error', fontsize=20)
plt.xlabel('# of Instances [-]', fontsize=12)
plt.ylabel('Ra values [-]', fontsize=12)
plt.legend(fontsize=12)
plt.savefig('regressor_error.png', dpi=300)

plt.figure(figsize=(10, 6))

plt.axvline(LOWER_SPECIFICATION_LIMIT,color='black', linewidth=4)
plt.axvline(UPPER_SPECIFICATION_LIMIT,color='black',linewidth=4)
plt.fill_betweenx([0, 1], 0, 0.5, color='green', alpha=0.4) 
plt.fill_betweenx([0, 1], LOWER_SPECIFICATION_LIMIT, UPPER_SPECIFICATION_LIMIT, color='blue', alpha=0.4)
plt.scatter(y_test,y_test + np.random.uniform(0.10,0.6, size=y_test.shape),color='red',linewidth=4, label='Quality observed')
plt.scatter(y_pred,y_pred + np.random.uniform(0.10,0.6, size=y_pred.shape), color='blue',linewidth=4,label='Quality predicted')
plt.legend()
plt.xlim([0,0.5])
plt.ylim([0,1])
plt.savefig(f'regressor_region_insights.png', dpi=300)