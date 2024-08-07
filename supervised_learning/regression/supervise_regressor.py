import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns

# Importing the data
dataset = pd.read_excel('/Users/mustafa/Documents/GitHub/ML_knife_QA/data/chiefs_knife_dataset.xlsx')

# Features und Zielvariable
X = dataset.loc[:, 'Original_Linienanzahl':'DFT_Median_sobel_Bereich'].values
y = dataset['Ra'].values

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data Augmentation durch Rauschzugabe
noise = np.random.normal(0, 0.01, X_train.shape)  # Rauschen mit Mittelwert 0 und Standardabweichung 0.01
X_train_augmented = X_train + noise
y_train_augmented = np.copy(y_train)  # Labels bleiben gleich

# Verknüpfung von Original- und Augmented Features
X_train = np.vstack((X_train, X_train_augmented))
y_train = np.concatenate((y_train, y_train_augmented))

# Feature-Skalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Modelltraining mit RandomForestRegressor (Vor Randomized Search)
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_scaled, y_train)

# Vorhersagen
y_pred = regressor.predict(X_test_scaled)
y_val_pred = regressor.predict(X_val_scaled)

# Validation - Evaluierung
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
print(f'Mean Squared Error (Validation): {mse_val}')
print(f'R^2 Score (Validation): {r2_val}')

# Evaluierung
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error (Test): {mse}')
print(f'R^2 Score (Test): {r2}')

# Hyperparameter-Raster für Randomized Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Randomized Search für Random Forest Regressor
random_search_regressor = RandomizedSearchCV(estimator=regressor,
                                             param_distributions=param_grid, 
                                             n_iter=100, # Anzahl der zufälligen Kombinationen
                                             cv=5,       # Cross-Validation-Folds
                                             # scoring='neg_mean_squared_error', # Metrik für Regression
                                             scoring='r2',
                                             n_jobs=-1,  # Alle verfügbaren Kerne verwenden
                                             verbose=0,  # Ausführlichkeit
                                             random_state=42)  # Für Reproduzierbarkeit
random_search_regressor.fit(X_train_scaled, y_train)

# Ausgabe der besten Hyperparameter
print(f'Beste Hyperparameter für Regressor: {random_search_regressor.best_params_}')

# Bestes Modell basierend auf der Suche
best_model = random_search_regressor.best_estimator_

# Vorhersagen mit dem besten Modell
y_pred_best = best_model.predict(X_test_scaled)

# Evaluierung des besten Modells
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
print(f'Best Model Mean Squared Error: {mse_best}')
print(f'Best Model R^2 Score: {r2_best}')

# Feature Importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Namen der Features
feature1 = dataset.columns.get_loc('Original_Linienanzahl')
feature2 = dataset.columns.get_loc('DFT_Median_sobel_Bereich')
feature_names = dataset.columns[feature1:feature2+1]

# Visualisierung der Feature Importances
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Ausgabe der Feature Importances
feature_importances = pd.DataFrame({'Feature': feature_names[indices], 'Importance': importances[indices]})
print(feature_importances)
