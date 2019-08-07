#!/usr/bin/python
from connect import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

# from sklearn.ensemble import BaggingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import BayesianRidge
# from sklearn.svm import SVR
# from sklearn.svm import LinearSVR
# from sklearn.linear_model import LogisticRegression

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


def main():
    fecha = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    with open('/home/rodrigo/Escritorio/Tesis/pruebas/resultados_diarios '+fecha+'.csv', 'w+') as csvfile:
        spamwriter = csv.writer(csvfile, lineterminator='\n')
        # Lista de los nombres de los algoritmos de regresión a utilizar

        # Lista de las clases de los algoritmos de regresión a utilizar
        estimators = [AdaBoostRegressor(),
                      DecisionTreeRegressor(),
                      RandomForestRegressor(),
                      GradientBoostingRegressor()]

        normalizadores = [StandardScaler(),
                        MinMaxScaler(),
                        MaxAbsScaler(),
                        RobustScaler()]

        # Lista de los nombres de los productos a analizar
        IDs = [38, 2, 497, 16, 23]

        variablesUtilizadas = []
        for (normalizador) in (normalizadores):
            spamwriter.writerow(['AdaBoostRegressor',
                                 'DecisionTreeRegressor',
                                 'RandomForestRegressor',
                                 'GradientBoostingRegressor',
                                 'TriggerModel'])
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

                list_result_cv_error = []
                resultList = []
                for (estimator) in (estimators):
                    feature_selector = RFECV(estimator=estimator, cv=LeaveOneOut(), scoring='r2').fit(x_all, y_train).support_
                    columns_selected = [columns_all[idx] for idx, val in enumerate(feature_selector) if val]
                    #print(columns_all)
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

                    #mean_absolute_error
                    #mean_squared_error
                    #r2_score
                    resultList.append(mean_squared_error(y_test, estimator.fit(x_train, y_train).predict(x_test)))

                i_best = list_result_cv_error.index(max(list_result_cv_error))
                print(resultList)
                print(resultList[i_best])
                resultList =  resultList + [resultList[i_best]]
                spamwriter.writerow(resultList)

    with open('/home/rodrigo/Escritorio/Tesis/pruebas/resultados_diarios_variables '+fecha+'.csv', 'w+') as csvfile:
        spamwriter = csv.writer(csvfile, lineterminator='\n')
        spamwriter.writerows(variablesUtilizadas)

if __name__ == "__main__":
    main()
