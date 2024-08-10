import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder

"""#Importing data"""

dataset = pd.read_excel('../../data/chiefs_knife_dataset.xlsx')
index_Ra = dataset.columns.get_loc('Ra')  # index of the surface roughness column for inserting the class. label

LOWER_SPECIFICATION_LIMIT = 0.125  # lower bound of good quality product region
UPPER_SPECIFICATION_LIMIT = 0.215  # upper bound of good quality product region

is_between_specification_bounds = (dataset['Ra'] >= LOWER_SPECIFICATION_LIMIT) & (dataset['Ra'] < UPPER_SPECIFICATION_LIMIT)
good_product_range = np.where(is_between_specification_bounds, "good", "bad")
dataset.insert(index_Ra + 1, 'Quality', good_product_range) 


"""# constructing Label"""

X = dataset.loc[:,'Original_Linienanzahl':'DFT_Median_sobel_Bereich'].values
y_classifier = dataset['Quality'].values
y_regressor = dataset['Ra'].values

"""#Encoding categorical data"""

y_classifier = np.where(y_classifier == 'good', 0, 1)

"""#Splitting dataset into training and test set"""

def get_split_dataset(regressor=True,min_max_scaler=True, X=X, y_regressor=y_regressor, y_classifier=y_classifier,test_size=0.2, rnd_state=1):
    mm_sc = MinMaxScaler()
    sc = StandardScaler()
    if regressor:
        X_train, X_test, y_train, y_test = train_test_split(X, y_regressor, test_size=test_size, shuffle=True, random_state=rnd_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y_classifier, test_size=test_size, shuffle=True, random_state=rnd_state)
    
    if min_max_scaler:
        X_train = mm_sc.fit_transform(X_train)
        X_test = mm_sc.transform(X_test)
    else:
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

    # we Resample the dataset to balance the classes
    return X_train, X_test, y_train, y_test
