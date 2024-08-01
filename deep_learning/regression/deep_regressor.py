import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

dataframe = pd.read_excel('../../data/chiefs_knife_dataset.xlsx')

index_Ra = dataframe.columns.get_loc('Ra')
lower_specification_limit = 0.125 # lower bound of good quality product region
upper_specification_limit = 0.215  # upper bound of good quality product region

good_product_range = (dataframe['Ra']>=lower_specification_limit) & (dataframe['Ra'] < upper_specification_limit)
dataframe.insert(index_Ra +1, 'Good_Quality',good_product_range.astype(int))

X = dataframe.loc[:,'Original_Linienanzahl':'DFT_Median_sobel_Bereich'].values
target_regression = dataframe['Ra'].values
y_class = dataframe['Good_Quality'].values


# X = dataset.iloc[:,:-1].values
# y_class = dataset.iloc[:,-1].values


# def optimizer_choice(learning_rate = 1e-3, choice = 'SGD'):
#     if choice.lower() == 'sgd':
#         tf.keras.optimizers.SGD(learning_rate=learning_rate)
#     elif choice.lower() == 'adam': 
#          tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     else:
#         raise Exception("Sorry, choose SGD or Adam") 


# initial hyperparameters
optimizer = 'SGD'
activation = 'relu'
hidden_layers = [6, 16]  # (hidden layers)
learning_rate = [1e-2, 1e-3, 1e-4,1e-5]
batch_size = 32
epochs = 50

# hyperparameters
hyperparameters = {
'optimizer':{
    'sgd_0':[tf.keras.optimizers.SGD(learning_rate=learning_rate[0]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[1]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[2]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[3])],
    'sgd_1':[tf.keras.optimizers.SGD(learning_rate=learning_rate[0]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[1]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[2]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[3])],
    'adam_0':[tf.keras.optimizers.Adam(learning_rate=learning_rate[0]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[1]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[2]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[3])],
    'adam_1':[tf.keras.optimizers.Adam(learning_rate=learning_rate[0]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[1]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[2]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[3])],
    'sgd_10':[tf.keras.optimizers.SGD(learning_rate=learning_rate[0]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[1]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[2]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[3])],
    'sgd_11':[tf.keras.optimizers.SGD(learning_rate=learning_rate[0]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[1]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[2]),
           tf.keras.optimizers.SGD(learning_rate=learning_rate[3])],
    'adam_10':[tf.keras.optimizers.Adam(learning_rate=learning_rate[0]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[1]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[2]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[3])],
    'adam_11':[tf.keras.optimizers.Adam(learning_rate=learning_rate[0]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[1]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[2]),
           tf.keras.optimizers.Adam(learning_rate=learning_rate[3])],
},
'activation': ['relu','tanh'],
'hidden_layers' : [128,64,32],  # (hidden layers)
'batch_size' : [32, 16],
'epochs' : [100, 200],
}




X_train, X_test, y_train, y_test = train_test_split(X,target_regression, test_size=0.2,shuffle=True,random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def create_model(optimizer=optimizer, activation=activation, neurons=hidden_layers,batch_size = batch_size,epochs = epochs):
    # create model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons[0], input_dim=X_train.shape[1], activation=activation)) # EingangsLayer verbunden zu erstem versteckem Layer
    for hidden_layer in neurons[1:]:       # restliche versteckte Layers
        model.add(tf.keras.layers.Dense(hidden_layer, activation=activation)) 
    model.add(tf.keras.layers.Dense(1))   # Ausgangslayer
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    history = model.fit(X_train, y_train,batch_size = batch_size, validation_split=0.1, epochs=epochs, verbose = 1)
    return history,model


def predictor(model, test_feature=X_test):
    return model.predict(test_feature)

history, nn = create_model()

y_pred = predictor(model=nn, test_feature= X_test)


#Modellfehler
mse = nn.evaluate(X_test,y_test, batch_size=batch_size)
print(f"Root Mean Square {mse = }")

   

fig, axes = plt.subplots(1,2,figsize=(25, 15))

axes[0].plot(history.history['loss'], label='Training loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation loss', linewidth=2)
axes[0].set_title('Training and Validation Loss',fontsize=20)
axes[0].set_ylabel('Loss',fontsize=12)
axes[0].set_xlabel('Epoch',fontsize=12)
axes[0].legend(fontsize=12)
# axes[0].text(0.95, 0.5, f'Root Mean Square={mse:.3f}', ha='right', va='top', transform=axes[0].transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
stats = (
        f'activation = {activation}\n'
        f'Root mse= {mse:.3f}\n'
        f'optimizer = {optimizer}\n'
        f'epochs = {epochs}\n'
        f'batch_size = {batch_size}\n'
        f'hidden_layers = {hidden_layers}')
bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
axes[0].text(0.95, 0.5, stats, fontsize=20, bbox=bbox,
            transform=axes[0].transAxes, horizontalalignment='right')

axes[1].plot(y_test, color='red', label='Ground true value')
axes[1].plot(y_pred, color='blue', label='Predicted value')
axes[1].set_title('Predicted vs. True Target Values', fontsize=20)
axes[1].set_xlabel('True Target Values',fontsize=12)
axes[1].set_ylabel('Predicted Target Values',fontsize=12)
axes[1].legend(fontsize=12)
plt.savefig(f'initial_model.png', dpi=300)


def parameter_tuning(optimizer_name,activation_index = 0, batch_index = 0, epoch_index = 0):
    training_history_list = []

    for value in hyperparameters['optimizer'][optimizer_name]:
        history, nn = create_model(optimizer=value,
                                    activation=hyperparameters['activation'][activation_index],
                                    neurons=hyperparameters['hidden_layers'],
                                    batch_size=hyperparameters['batch_size'][batch_index],
                                    epochs=hyperparameters['epochs'][epoch_index],
                                    )
        training_history_list.append(history)
    return training_history_list


def row_plot(ax,row,training_history,activation, optimizer_name,batch_size):
    for column in range(len(learning_rate)):
         ax[row,column].plot(training_history[column].history['loss'], label='Training loss')
         ax[row,column].plot(training_history[column].history['val_loss'], label='Validation loss')
         ax[row,column].set_title(f'Loss for activ.={activation}, opt.={optimizer_name},lr={learning_rate[column]}, batch={batch_size}')
         ax[row,column].set_ylabel('Loss')
         ax[row,column].set_xlabel('Epoch')
         ax[row,column].legend()


'''

'''

training_history_0= parameter_tuning(optimizer_name='sgd_0',activation_index=0,batch_index=0,epoch_index=0)
training_history_1= parameter_tuning(optimizer_name='adam_0',activation_index=0,batch_index=0,epoch_index=0)
training_history_2= parameter_tuning(optimizer_name='sgd_1',activation_index=0,batch_index=1,epoch_index=1)
training_history_3= parameter_tuning(optimizer_name='adam_1',activation_index=0,batch_index=1,epoch_index=1)

fig, ax = plt.subplots(4,len(learning_rate), figsize=(20, 15))


def save_gragh(fig_axes,training_history0,training_history1,training_history2,training_history3,activation):
    row_plot(ax=fig_axes,row=0,training_history=training_history0,activation=activation,optimizer_name='sgd',batch_size=16)
    row_plot(ax=fig_axes,row=1,training_history=training_history1,activation=activation,optimizer_name='adam',batch_size=16)
    row_plot(ax=fig_axes,row=2,training_history=training_history2,activation=activation,optimizer_name='sgd',batch_size=8)
    row_plot(ax=fig_axes,row=3,training_history=training_history3,activation=activation,optimizer_name='adam',batch_size=8)              
    plt.tight_layout()
    plt.savefig(f'loss_{activation}.png', dpi=300)

save_gragh(fig_axes= ax,training_history0 = training_history_0,training_history1=training_history_1,training_history2=training_history_2,training_history3=training_history_3,activation='relu')


training_history_10= parameter_tuning(optimizer_name='sgd_10',activation_index=1,batch_index=0,epoch_index=0)
training_history_11= parameter_tuning(optimizer_name='adam_10',activation_index=1,batch_index=0,epoch_index=0)
training_history_12= parameter_tuning(optimizer_name='sgd_11',activation_index=1,batch_index=1,epoch_index=1)
training_history_13= parameter_tuning(optimizer_name='adam_11',activation_index=1,batch_index=1,epoch_index=1)


fig, bx = plt.subplots(4,len(learning_rate), figsize=(20, 15))

save_gragh(fig_axes= bx,training_history0 = training_history_10,training_history1=training_history_11,training_history2=training_history_12,training_history3=training_history_13,activation='tanh')



end_training_history, end_model = create_model(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4),
                                               activation='tanh',
                                               neurons=[128,64,32],
                                               batch_size=16,
                                               epochs=100)
y_pred_end = predictor(model=end_model, test_feature= X_test)
mse_end = end_model.evaluate(X_test,y_test, batch_size=batch_size)
print(f"Root Mean Square {mse_end = }")

fig, axes = plt.subplots(1,2,figsize=(25, 15))

axes[0].plot(end_training_history.history['loss'], label='Training loss', linewidth=2)
axes[0].plot(end_training_history.history['val_loss'], label='Validation loss', linewidth=2)
axes[0].set_title('Training and Validation Loss',fontsize=20)
axes[0].set_ylabel('Loss',fontsize=12)
axes[0].set_xlabel('Epoch',fontsize=12)
axes[0].legend(fontsize=12)
stats = (
        f'activation = tanh\n'
        f'Root mse= {mse_end:.3f}\n'
        f'optimizer = SGD\n'
        f'epochs = 100\n'
        f'batch_size = 16\n'
        f'hidden_layers = [128,64,32]')
bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
axes[0].text(0.95, 0.5, stats, fontsize=20, bbox=bbox,
            transform=axes[0].transAxes, horizontalalignment='right')

axes[1].plot(y_test, color='red', label='Ground true value')
axes[1].plot(y_pred_end, color='blue', label='Predicted value')
axes[1].set_title('Predicted vs. True Target Values', fontsize=20)
axes[1].set_xlabel('True Target Values',fontsize=12)
axes[1].set_ylabel('Predicted Target Values',fontsize=12)
axes[1].legend(fontsize=12)
plt.savefig(f'initial_model.png', dpi=300)
