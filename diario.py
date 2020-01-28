#!/usr/bin/python
from connect import *
from aic_bic import aic, bic, check_aic_bic

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectFromModel

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
    fecha = datetime.today().strftime('%Y-%m-%d %H%M%S')
    with open('/home/rodrigo/Escritorio/Tesis/pruebas/resultados_diarios '+fecha+'.csv', 'w+') as csvfile:
        spamwriter = csv.writer(csvfile, lineterminator='\n')

        # Lista de las clases de los algoritmos de regresión a utilizar
        estimators = [Ridge(),
                      Lasso(),
                      LogisticRegression(),
                      LinearRegression()]

        estimators_name = []
        for (estimator) in (estimators):
            estimators_name.append(type(estimator).__name__)
        estimators_name.append('TriggerModel')

        normalizadores = [MinMaxScaler()]

        # Lista de los nombres de los productos a analizar
        IDs = [('ESPEJO_INCOLORO_Optimirror', 38),
               ('VIDRIO_INCOLORO_(BAJA)', 2),
               ('LAMINADO_INCOLORO_(Baja)', 497),
               ('VIDRIO_INCOLORO_(ALTA)', 16),
               ('VIDRIO_GRIS_(BAJA)', 23)]

        variablesUtilizadas = []
        for (normalizador) in (normalizadores):
            spamwriter.writerow(estimators_name)
            for nombre, id in IDs:
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
                    #feature_selector = SelectFromModel(estimator).fit(x_all, y_train).get_support()
                    #no_hay = True
                    #for hay in feature_selector:
                    #    if hay:
                    #        columns_selected = [columns_all[idx] for idx, val in enumerate(feature_selector) if val]
                    #        no_hay = False
                    #        break
                    #if no_hay:
                    #    columns_selected = ['precioproducto_max']
                    #print(columns_all)
                    print(columns_selected)
                    variablesUtilizadas.append(columns_selected)
                    x = pd.DataFrame(x_normalizado, columns=columns_selected)
                    x_train = x[:CV]
                    x_test = x[CV:]
                    #neg_mean_absolute_error
                    #neg_mean_squared_error
                    #r2
                    cv_result = cross_val_score(estimator, x_train, y_train, cv=LeaveOneOut(),
                                                        scoring='r2').mean()
                    list_result_cv_error.append(cv_result)

                    check_aic_bic(x_train, y_train, estimator)
                    #mean_absolute_error
                    #mean_squared_error
                    resultList.append(np.sqrt(mean_squared_error(y_test, estimator.fit(x_train, y_train).predict(x_test))))
                    #resultList.append(mean_absolute_error(y_test, estimator.fit(x_train, y_train).predict(x_test)))
                    #resultList.append(mean_absolute_percentage_error(y_test, estimator.fit(x_train, y_train).predict(x_test)))

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
