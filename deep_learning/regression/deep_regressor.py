import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

#1. Entscheiden Sie sich für ein Framework zum Training eines Neuronalen Netzes (PyTorch, TensorFlow,
# etc.), dass Sie verwenden werden und lesen Sie die Daten der extruder.csv ein.


dataframe = pd.read_csv('chiefs_knife_dataset.csv',delimiter=';')

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


# 3. Definieren Sie als nächstes die Hyperparamter des Neuronalen Netzes. Dies ist erforderlich, da die
# Struktur des Neuronalen Netzes unklar ist.

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
hidden_layers = [6, 16]  # verteckte Ebenen (hidden layers)
learning_rate = [1e-2, 1e-3, 1e-4,1e-5]
batch_size = 32
epochs = 50

# hyperparameter, um Modell zu optimieren (wird später weiterverwendet)
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
'hidden_layers' : [128,64,32],  # verteckte Ebenen (hidden layers)
'batch_size' : [32, 16],
'epochs' : [100, 200],
}

# 4. Unterteilen Sie die Daten in Trainings- und Testdaten.


X_train, X_test, y_train, y_test = train_test_split(X,target_regression, test_size=0.2,shuffle=True,random_state=0)


# 5. Untersuchen Sie die Trainingsergebnisse Ihres Neuronalen Netzes und diskutieren deren
# Qualität/Modellfehler.


''' ich habe eine Skalierung vorgenommen, da die Werte von unterschiedliche Einheiten und auch unterschiedlich 
    groß sind. Ohne Skalierung würde die großere Werte den Lernalgorithmus beeinflussen oder steuern.'''

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# eine Funktion zur vereinfachung der Wahl von Parameter beim Trainieren

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

#Vorhersage von Testdaten
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


# Diskussion:

'''
Das erste Modell wurde mit stochastic gradient descent trainiert (mit standard learning rate= 0.01) mit ein
batch_size = 32, ein epoch = 50 und mit ReLu als Aktivierungsfunction. Allerdings overfit das Modell, weil
wenn man sich die Kurven von Loss-epoch für die Trainings- und validationsdaten anschaut 
sieht man, dass beide Kurven nicht zusammen konvergieren. Das Modell verhält sich besser 
bei den Trainingsdaten. Bei den validationsdaten scatter die Kurve was ein overfitting darstellt.
Der Vergleich von vorhesage mit den Testdaten-ausgang y_test ist schwierig zu bewerten, da 
manche Werte werden akzeptabel abgebildet manche jedoch nicht. 

Um das Modell zu verbessern, müssen die Hyperparameter angepasst werden.
'''

# 6. Erproben Sie weitere 2 Netzkonfigurationen (Hyperparameter) und diskutieren Sie alle Ergebnisse
# anhand der folgenden Fragestellung:


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
hier werden nur 32 Modelle (2 waren gefragt) aus der unterschiedlichen Möglichkeiten (siehe Hyperparametzer) angeschaut
'''

training_history_0= parameter_tuning(optimizer_name='sgd_0',activation_index=0,batch_index=0,epoch_index=0)
training_history_1= parameter_tuning(optimizer_name='adam_0',activation_index=0,batch_index=0,epoch_index=0)
training_history_2= parameter_tuning(optimizer_name='sgd_1',activation_index=0,batch_index=1,epoch_index=1)
training_history_3= parameter_tuning(optimizer_name='adam_1',activation_index=0,batch_index=1,epoch_index=1)

fig, ax = plt.subplots(4,len(learning_rate), figsize=(20, 15))

# neu Optimizer muss definiert werden, da sie nur einmal verwendet werden dürfen 


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




# • Kann ein Overfitting durch das Modell ausgeschlossen werden und wie begründen Sie dies?

'''
Ausgehend von den Darstellung der Loss-epoch für beide Trainings- und Validierungsdaten kann man sagen, dass 
in dieser Studie ein Overfitting ausgeschlossen werden kann. Da das Overfitting wurde nur für einige learning_rate beobachtet 
und diese wurde nicht weiter genutzt. Nur die beste Modell wurde genutzt, die den kleinste Fehler in Validierungsdaten aufweist.

Für Relu: Das Modell overfit für ein learning_rate = 0.01 unabhängig von anderen Parametern. Es overfit auch für ein learning_rate= 0.001
für alle außer der Optimierung=SGD, batch_size=16

Für tanh: Das Modell overfit für ein learning_rate = 0.01 unabhängig von anderen Parametern. Es overfit auch für ein learning_rate= 0.001
für alle außer der Optimierung=SGD, batch_size=16

'''


# • Ist der Modellfehler ausreichend klein, um das Modell auch sinnvoll anzuwenden?

'''
In dem Fall habe ich eine Regressionsanalyse vorgenommen. Daher ist es sehr schwierig zu sagen ob die Werte ausreichend klein sind
da keine Maßtab dafür gibt. Hätte eine Klassifizierung gewesen, dann könnte man miHilfe der Konfusionsmatrix schnell bewerten.
Hier kann man auch den R^2 anszeigen lassen aber da die Berechnung viel Zeit nimmt und schon gemacht wurde habe ich nicht mitgenommen.
'''


# • Würden Sie das Modell einsetzen, wenn Sie mit Ihrer Firma für mehr als 5% fehlerhafter
# Qualitätsaussagen aufkommen müssten (produzierter Ausschuss trotz prognostizierter hoher
# Qualität)?

# Ausgehend von den Abbildungen wähle ich folgende Parameter:
'''
activation: tanh
optimizer: SGD
learning_rate: 1e-4
neurons:[128,64,32]
batch_size:16
'''
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


#ja das Modell würde ich anwenden wegen sehr kleine fehler



# • Welche Maßnahmen sind aus Ihrer Sicht erforderlich, um das Modell weiter zu verbessern?

'''
 --> mehr Daten für das trainieren kann das Modell verbessern
 --> alle sinnvolle Kombinierung von Hyperparametern anschauen (hyperparameter tunning)
 --> das Modell Architektur ändern

'''