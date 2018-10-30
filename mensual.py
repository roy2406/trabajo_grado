#!/usr/bin/python
from connect import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

#from sklearn.ensemble import BaggingRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.neural_network import MLPRegressor
#from sklearn.linear_model import BayesianRidge
#from sklearn.svm import SVR
#from sklearn.svm import LinearSVR
#from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

N_ESTIMATORS = 100

def correlacion(df):
    cols = list(df.columns.values)
    cm = np.corrcoef(df.values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols,
                     xticklabels=cols)
    plt.show()


def main():

    #df = getTableVidrieria()
    #IDs = [38,2,497,16,23]
    df = getTableImportadora()
    IDs = [60069, 89917, 14130, 539, 27062]

    with open('/home/rodrigo/Escritorio/resultados_mensual.csv', 'w+') as csvfile:
        spamwriter = csv.writer(csvfile, lineterminator='\n')

        spamwriter.writerow(['TriggerModel',
                             'AdaBoostRegressor',
                             'DecisionTreeRegressor',
                             'RandomForestRegressor',
                             'GradientBoostingRegressor'])

        estimators = [AdaBoostRegressor(),
                      DecisionTreeRegressor(),
                      RandomForestRegressor(),
                      GradientBoostingRegressor()]
        columns_all = np.array(
            ['mes', 'cuatrimestre', 'anho', 'precioproducto_mean', 'precioproducto_min', 'precioproducto_max'])
        for id in IDs:
            #filtrado = df[(df['idproducto'] == id) & (pd.to_datetime(df['fecha'], format='%Y-%m-%d') < '2018-03-01')]
            filtrado = df[(df['idproducto'] == id) & (pd.to_datetime(df['fecha'], format='%Y-%m-%d') < '2017-10-01')]
            agrupado = filtrado.groupby(['mes', 'cuatrimestre', 'anho']).aggregate(
                {'precioproducto': {'precioproducto_mean': np.mean, 'precioproducto_max': np.max, 'precioproducto_min': np.min},
                 'cantidad': {'cantidad_sum': np.sum}})

            agrupado = agrupado.reset_index(col_level=1)

            agrupado.columns = agrupado.columns.get_level_values(1)

            agrupado = agrupado.sort_values(by=['anho', 'mes'])

            CV = np.ceil(agrupado.shape[0] * 0.75).astype(int)

            x_all = pd.DataFrame(agrupado, columns=columns_all)[:CV]
            y = pd.DataFrame(agrupado, columns=['cantidad_sum'])
            y_train = y[:CV]
            y_test = y[CV:]


            list_result_cv_error = []
            resultList = []
            for (estimator) in (estimators):
                feature_selector = RFECV(estimator=estimator,cv=LeaveOneOut())
                x = pd.DataFrame(agrupado, columns=columns_all[feature_selector.fit(x_all, y_train).support_])
                x_train = x[:CV]
                x_test = x[CV:]
                cv_result = np.mean(cross_val_score(estimator, x_train, y_train.values.ravel(),cv=CV, scoring='neg_mean_squared_error'))
                list_result_cv_error.append(cv_result)
                resultList.append(mean_squared_error(y_test, estimator.fit(x_train,y_train).predict(x_test)))

            i_best = list_result_cv_error.index(max(list_result_cv_error))
            print(resultList)
            print(resultList[i_best])
            resultList = [resultList[i_best]] + resultList
            spamwriter.writerow(resultList)

if __name__ == "__main__":
    main()
