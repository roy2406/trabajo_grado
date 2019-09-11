#!/usr/bin/python
from connect import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():

    # Lista de las clases de los algoritmos de regresión a utilizar
    estimators = [Ridge(),
                  Lasso(),
                  LogisticRegression(),
                  LinearRegression()]
    #estimators = [DecisionTreeRegressor(),
    #              GradientBoostingRegressor(),
    #              RandomForestRegressor(),
    #              AdaBoostRegressor()]

    estimators_name = []
    for (estimator) in (estimators):
        estimators_name.append(type(estimator).__name__)
    estimators_name.append('TriggerModel')

    normalizadores = [StandardScaler(),
                    MinMaxScaler(),
                    MaxAbsScaler(),
                    RobustScaler()]

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

            # gca stands for 'get current axis'
            ax = plt.gca()
            x_normalizado.plot(kind='line', y='precioproducto_min',color='red', ax=ax)
            x_normalizado.plot(kind='line', y='precioproducto_max',color='blue', ax=ax)
            x_normalizado.plot(kind='line', y='precioproducto_mean',color='green', ax=ax)
            plt.show()

            x_all = x_normalizado[:CV]

            list_result_cv_error = []
            resultList = []
            for (estimator) in (estimators):
                feature_selector = RFECV(estimator=estimator, cv=LeaveOneOut(), scoring='r2').fit(x_all, y_train).support_
                columns_selected = [columns_all[idx] for idx, val in enumerate(feature_selector) if val]
                print(columns_selected)
                variablesUtilizadas.append(columns_selected)
                x = pd.DataFrame(x_normalizado, columns=columns_selected)
                x_train = x[:CV]
                x_test = x[CV:]
                #neg_mean_absolute_error
                #neg_mean_squared_error
                #r2
                cv_result = cross_val_score(estimator, x_train, y_train.values.ravel(), cv=LeaveOneOut(),
                                                    scoring='r2').mean()
                list_result_cv_error.append(cv_result)
                # mean_absolute_error
                # mean_squared_error
                # resultList.append(np.sqrt(mean_squared_error(y_test, estimator.fit(x_train, y_train).predict(x_test))))
                resultList.append(mean_absolute_error(y_test, estimator.fit(x_train, y_train).predict(x_test)))
                # resultList.append(mean_absolute_percentage_error(y_test, estimator.fit(x_train, y_train).predict(x_test)))

            i_best = list_result_cv_error.index(max(list_result_cv_error))
            print(resultList)
            print(resultList[i_best])
            resultList =  resultList + [resultList[i_best]]

if __name__ == "__main__":
    main()
