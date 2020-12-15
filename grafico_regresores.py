#!/usr/bin/python
from setuptools.msvc import winreg

from connect import *
from aic_bic import aic, bic, check_aic_bic

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

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

        estimators_name = []
        for (estimator) in (estimators):
            estimators_name.append(type(estimator).__name__)
        estimators_name.append('TriggerModel')

        normalizadores = [MinMaxScaler()]

        # Lista de los nombres de los productos a analizar
        IDs = [38, 2, 497, 16, 23]

        listaResultList = []
        listaStdList = []
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

                list_result_cv_error = []
                resultList = []
                stdList = []
                for (estimator) in (estimators):
                    feature_selector = RFECV(estimator=estimator, cv=10, scoring='r2').fit(x_all, y_train).support_
                    columns_selected = [columns_all[idx] for idx, val in enumerate(feature_selector) if val]
                    #print(columns_all)
                    x = pd.DataFrame(x_normalizado, columns=columns_selected)
                    x_train = x[:CV]
                    x_test = x[CV:]
                    #neg_mean_absolute_error
                    #neg_mean_squared_error
                    #r2
                    cv_result = cross_val_score(estimator, x_train, y_train, cv=10,
                                                        scoring='r2').mean()
                    list_result_cv_error.append(cv_result)

                    #mean_absolute_error
                    #mean_squared_error
                    error = np.sqrt(mean_squared_error(y_test, estimator.fit(x_train, y_train).predict(x_test)))
                    #error = mean_absolute_error(y_test, estimator.fit(x_train, y_train).predict(x_test))
                    #mean_absolute_percentage_error(y_test, estimator.fit(x_train, y_train).predict(x_test))
                    #print(np.squeeze(np.asarray(y_test.values)))

                    var_y = np.squeeze(np.asarray(y_test.values))
                    var_y_pred = np.squeeze(np.asarray(estimator.fit(x_train, y_train).predict(x_test)))

                    standard_deviation = np.std(var_y - var_y_pred)
                    resultList.append(error)
                    stdList.append(standard_deviation)

                i_best = list_result_cv_error.index(max(list_result_cv_error))
                print(resultList)
                print(resultList[i_best])
                print(stdList)
                resultList =  resultList + [resultList[i_best]]
                stdList =  stdList + [stdList[i_best]]
                listaResultList.append(resultList)
                listaStdList.append(stdList)

        #plt.clf()
        #labels = ['C1', 'C2', 'C3', 'C4', 'C5']
        #plt.plot(labels,[row[0] for row in listaResultList], label='Ridge')
        #plt.plot(labels,[row[1] for row in listaResultList], label='Lasso')
        #plt.plot(labels,[row[2] for row in listaResultList], label='Logistic')
        #plt.plot(labels,[row[3] for row in listaResultList], label='Linear')
        #plt.plot(labels,[row[4] for row in listaResultList], label='TriggerModel')
        ##plt.grid(b=True, which='major', color='#666666', linestyle='-')
        #plt.xlabel('Caso')
        #plt.ylabel('Error')
        #plt.legend()
        #plt.savefig('/home/rodrigo/Escritorio/comparacion.eps', format='eps')

        print(listaResultList)
        print(listaStdList)

        labels = ['C1', 'C2', 'C3', 'C4', 'C5']
        x = np.arange(len(labels))  # the label locations
        width = 0.19  # the width of the bars
        fig, ax = plt.subplots()
        ax.bar(x - width*2,
                [row[0] for row in listaResultList],
                align='center',
                alpha=0.5,
               width=width,
               label='Ridge')
        ax.bar(x - width,
               [row[1] for row in listaResultList],
               align='center',
               alpha=0.5,
               width=width,
               label='Lasso')
        ax.bar(x,
               [row[2] for row in listaResultList],
               align='center',
               alpha=0.5,
               width=width,
               label='LogisticRegression')
        ax.bar(x + width,
               [row[3] for row in listaResultList],
               align='center',
               alpha=0.5,
               width=width,
               label='LinearRegression')
        ax.bar(x + width*2,
               [row[4] for row in listaResultList],
               align='center',
               alpha=0.5,
               width=width,
               label='TriggerModel')


        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Error')
        ax.set_title('Caso')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True)
        ax.legend()

        fig.tight_layout()

        plt.savefig('/home/rodrigo/Desktop/Tesis/pruebas/graficos/COMPARACION'+'_'+datetime.today().strftime('%Y-%m-%d %H%M%S')+'.eps', format='eps')

if __name__ == "__main__":
    main()
