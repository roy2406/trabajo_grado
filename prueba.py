#!/usr/bin/python
from connect import *
from aic_bic import aic, bic

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    # Lista de las clases de los algoritmos de regresión a utilizar

    normalizadores = [MinMaxScaler()]

    # Lista de los nombres de los productos a analizar
    IDs = [38, 2, 497, 16, 23]

    variablesUtilizadas = []
    for (normalizador) in (normalizadores):
        for id in IDs:
            #Extracción de datos
            df = getTableVidrieriaDiario(id)
            columns_all = [x for x in list(df.columns.values) if x!='cantidad']
            CV = np.ceil(df.shape[0] * 0.75).astype(int)
            y = pd.DataFrame(df, columns=['cantidad'])
            y_train = y[:CV]
            y_test = y[CV:]
            x_normalizado = pd.DataFrame(normalizador.fit_transform(pd.DataFrame(df, columns=columns_all)), columns=columns_all)
            x_all = x_normalizado[:CV]

            feature_selector = RFECV(estimator=SVR(kernel="linear"), cv=LeaveOneOut(), scoring='r2').fit(x_all, y_train).support_
            columns_selected = [columns_all[idx] for idx, val in enumerate(feature_selector) if val]
            #print(columns_all)
            #print(columns_selected)
            variablesUtilizadas.append(columns_selected)
            x = pd.DataFrame(x_normalizado, columns=columns_selected)
            x_train = x[:CV]
            x_test = x[CV:]
            #mean_absolute_error
            #mean_squared_error
            print(np.sqrt(mean_squared_error(y_test, SVR(kernel="linear").fit(x_train, y_train).predict(x_test))))
            #print(mean_absolute_error(y_test, SVR(kernel="linear").fit(x_train, y_train).predict(x_test)))
            #spamwriter.writerow(mean_absolute_error(y_test, SVR(kernel="linear", C=1.0, epsilon=0.2).fit(x_train, y_train).predict(x_test)))
            #resultList.append(mean_absolute_percentage_error(y_test, estimator.fit(x_train, y_train).predict(x_test)))


            #feature_selector = RFECV(estimator=MLPRegressor(), cv=LeaveOneOut(), scoring='r2').fit(x_all, y_train).support_
            #columns_selected = [columns_all[idx] for idx, val in enumerate(feature_selector) if val]
            #print(columns_all)
            #print(columns_selected)
            variablesUtilizadas.append(columns_selected)
            x = pd.DataFrame(x_normalizado, columns=columns_selected)
            x_train = x[:CV]
            x_test = x[CV:]
            #mean_absolute_error
            #mean_squared_error
            print(np.sqrt(mean_squared_error(y_test, MLPRegressor().fit(x_train, y_train).predict(x_test))))
            #print(mean_absolute_error(y_test, MLPRegressor().fit(x_train, y_train).predict(x_test)))
            #spamwriter.writerow(mean_absolute_error(y_test, SVR(kernel="linear", C=1.0, epsilon=0.2).fit(x_train, y_train).predict(x_test)))
            #resultList.append(mean_absolute_percentage_error(y_test, estimator.fit(x_train, y_train).predict(x_test)))
if __name__ == "__main__":
    main()
