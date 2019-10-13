import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

from collections import Counter
import warnings

###################################### HYPERAS IMPORTS  #################################
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import (choice, uniform)

######################################  KERAS AND TENSORFLOW IMPORTS ####################
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam, Nadam, Adagrad, Adamax, Adadelta
from hyperas import optim

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


def data():
    def anomaly_hunter(df):
        # Turkey method to hunt ouliers
        outliers_indices = []
        for col in df.columns.tolist():
            #1st quartile (%25)
            Q1 = np.percentile(df[col],25)

            # 3rd quantile(%75)
            Q3 = np.percentile(df[col], 75)

            IQR = Q3 - Q1
            # Outlier step
            outlier_step = 1.5* IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

            #append the found outlier indices for col to the list of outlier indices
            outliers_indices.extend(outlier_list_col)

        #Select the outliers with more than 2 occurances

        outliers_indices = Counter(outliers_indices)
        multiple_outliers = list(k for k,v in outliers_indices.items() if v > 2)

        return multiple_outliers


    def oneHot(col, labels):
        lst = []
        for item in col:
            temp = [0 for i in range(len(labels))]
            temp[item-1] = 1
            lst.append(temp)
        df = pd.DataFrame(lst, columns=labels)
        return df

    df = pd.read_csv('glass.csv')
    features = df.columns[:-1].tolist()
    outliers_indices = anomaly_hunter(df[features])
    df = df.drop(outliers_indices).reset_index(drop=True)

    df.Type = df.Type.apply(lambda x:4 if x==7 else x)
    X = df[features]
    y = df['Type']
    seed = 7
    test_size = 0.3
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=seed)

    # Using oversampling technique

    # Converting to onehot encoded
    labels = df.Type.unique().tolist()
    y_train = oneHot(y_train, labels)
    y_test = oneHot(y_test, labels)
    

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return X_train, y_train, X_test, y_test


def create_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    
    # First layer
    model.add(Dense({{choice([9, 18, 36])}}, input_dim=9))
    model.add(Activation({{choice(['relu', 'softmax', 'elu', 'selu', 'softplus'])}}))
    model.add(Dropout({{uniform(0, 0.2)}}))
    
    # Second layer
    model.add(Dense({{choice([9, 18, 36])}}))
    model.add(Activation({{choice(['relu', 'softmax', 'elu', 'selu', 'softplus'])}}))

    # Output layer
    model.add(Dense(6))
    model.add(Activation('softmax'))

    # Optimizers
    
    adam = Adam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    sgd = SGD(lr={{choice([10**-3, 10**-2, 10**-1])}})
    nadam = Nadam(lr={{choice([10**-3, 10**-2, 10**-1])}})
    adagrad = Adagrad(lr={{choice([10**-3, 10**-2, 10**-1])}})
    # adamax = Adamax(lr={{choice([10**-3, 10**-2, 10**-1])}})
    # adadelta = Adadelta(lr={{choice([10**-3, 10**-2, 10**-1])}})
    # rmsprop = RMSprop(lr={{choice([10**-3, 10**-2, 10**-1])}})
    
    optimizers = dict(adam=adam, sgd=sgd, nadam=nadam, adagrad=adagrad)
    choiceval = {{choice(['adam', 'sgd', 'nadam', 'adagrad'])}}
    optimizer = optimizers[choiceval]
    
    model.compile(loss={{choice(['kullback_leibler_divergence', 'poisson',
                                 'cosine_proximity', 'huber_loss',
                                 'categorical_hinge', 'hinge',
                                 'mean_absolute_percentage_error'])}},
                  metrics=['accuracy'], optimizer=optimizer)
    model.fit(
              X_train, y_train, batch_size={{choice([5, 8, 15])}}, epochs={{choice([200, 300])}},
              verbose=2, validation_data=(X_test, y_test))

    score, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: ', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials())
    
    X_train, y_train, X_test, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

y_pred = best_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test = np.argmax(y_test.values, axis=-1)
print('Accuracy score: ', accuracy_score(y_test, y_pred))
