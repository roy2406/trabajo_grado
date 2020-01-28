#!/usr/bin/python
from connect import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools as it

N_ESTIMATORS = 100
CV=4

def main():
    df = getTableVidrieria()
    filtrado = df.loc[df['idproducto'] == 38]

    agrupado = filtrado.groupby(['cuatrimestre','anho'] ).aggregate(
        {'precioproducto': {'precioproducto_mean':np.mean, 'precioproducto_max':np.max, 'precioproducto_min':np.min},
         'cantidad': {'cantidad_sum':np.sum}})

    agrupado = agrupado.reset_index(col_level=1)

    agrupado.columns = agrupado.columns.get_level_values(1)

    agrupado = agrupado.sort_values(by=['anho', 'cuatrimestre'])

    x = pd.DataFrame(agrupado, columns= ['cuatrimestre','precioproducto_min'])
    y = pd.DataFrame(agrupado, columns= ['cantidad_sum'])
    aList = []
    nameList = []
    for i in range(0, 4):
        x_new = x[i:(i + 5)]
        y_new = y[i:(i + 5)]

        x_train = x_new[:4]
        x_test = x_new[4:]

        y_train = y_new[:4]
        y_test = y_new[4:]

        cv_lr = np.mean(cross_val_score(LinearRegression(), x_train, y_train.values.ravel(), cv=CV,
                                        scoring='neg_mean_absolute_error'))
        cv_tsr = np.mean(cross_val_score(TheilSenRegressor(), x_train, y_train.values.ravel(), cv=CV,
                                         scoring='neg_mean_absolute_error'))
        cv_gbr = np.mean(
            cross_val_score(GradientBoostingRegressor(n_estimators=N_ESTIMATORS), x_train, y_train.values.ravel(),
                            cv=CV, scoring='neg_mean_absolute_error'))
        cv_ext = np.mean(
            cross_val_score(ExtraTreesRegressor(n_estimators=N_ESTIMATORS), x_train, y_train.values.ravel(), cv=CV,
                            scoring='neg_mean_absolute_error'))
        cv_ab = np.mean(
            cross_val_score(AdaBoostRegressor(n_estimators=N_ESTIMATORS), x_train, y_train.values.ravel(), cv=CV,
                            scoring='neg_mean_absolute_error'))
        cv_bag = np.mean(
            cross_val_score(BaggingRegressor(n_estimators=N_ESTIMATORS), x_train, y_train.values.ravel(), cv=CV,
                            scoring='neg_mean_absolute_error'))
        # cv_mlp = np.mean(cross_val_score(MLPRegressor(),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))

        myList = (cv_lr, cv_tsr, cv_gbr, cv_ext, cv_ab, cv_bag)
        xi = myList.index(max(myList))

        if (xi == 0):
            regr = LinearRegression().fit(x_train, y_train)
            nameList.append('Linear Regression')
            aList.append(mean_absolute_error(y_test, regr.predict(x_test)))

        elif (xi == 1):
            tsr = TheilSenRegressor().fit(x_train, y_train)
            nameList.append('Theil-Sen Regression')
            aList.append(mean_absolute_error(y_test, tsr.predict(x_test)))

        elif (xi == 2):
            gbr = GradientBoostingRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
            nameList.append('Gradient Boosting Regression')
            aList.append(mean_absolute_error(y_test, gbr.predict(x_test)))

        elif (xi == 3):
            ext = ExtraTreesRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
            nameList.append('Extra Trees Regression')
            aList.append(mean_absolute_error(y_test, ext.predict(x_test)))

        elif (xi == 4):
            ab = AdaBoostRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
            nameList.append('Ada Boost Regression')
            aList.append(mean_absolute_error(y_test, ab.predict(x_test)))

        elif (xi == 5):
            bag = BaggingRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
            nameList.append('Bagging Regression')
            aList.append(mean_absolute_error(y_test, bag.predict(x_test)))

    print(aList)
    print(nameList)
    print(np.var(aList))

    fig, ax = plt.subplots()
    data_line = ax.plot(aList, label='Data', marker='o')
    mean_line = ax.plot([np.mean(aList)] * len(aList), label='Mean', linestyle='--')
    legend = ax.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
