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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools as it

N_ESTIMATORS = 100

def main():
    df = getTableVidrieria()
    filtrado = df[(df['idproducto'] == 38) & (pd.to_datetime(df['fecha'],format='%Y-%m-%d') < '2018-03-01')]

    agrupado = filtrado.groupby(['mes','cuatrimestre','anho'] ).aggregate(
        {'precioproducto': {'precioproducto_mean':np.mean, 'precioproducto_max':np.max, 'precioproducto_min':np.min},
         'cantidad': {'cantidad_sum':np.sum}})

    agrupado = agrupado.reset_index(col_level=1)

    agrupado.columns = agrupado.columns.get_level_values(1)

    agrupado = agrupado.sort_values(by=['anho', 'mes'])

    #all_columns = ['mes','cuatrimestre','anho', 'precioproducto_mean', 'precioproducto_min', 'precioproducto_max']
    all_columns = ['anho', 'precioproducto_mean', 'precioproducto_min', 'precioproducto_max']
    all_the_features = []
    for r in range(2, len(all_columns) + 1):
        all_the_features = all_the_features + [x for x in it.combinations(all_columns, r)]

    var_column =[[0,[],2147483647],[0,[],2147483647],[0,[],2147483647],[0,[],2147483647],[0,[],2147483647]]
    for columns in all_the_features:

        x = pd.DataFrame(agrupado, columns=columns)
        y = pd.DataFrame(agrupado, columns=['cantidad_sum'])
        aList = []
        for tam_ventana in range(10, 20):
            hasta = 23 - tam_ventana + 1
            for i in range(0, hasta):
                x_new = x[i:(i + tam_ventana)]
                y_new = y[i:(i + tam_ventana)]

                x_train = x_new[:tam_ventana-1]
                x_test = x_new[tam_ventana-1:]

                y_train = y_new[:tam_ventana-1]
                y_test = y_new[tam_ventana-1:]
                CV = tam_ventana -1
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

                myList = (cv_lr, cv_tsr, cv_gbr, cv_ext, cv_ab, cv_bag)
                xi = myList.index(max(myList))

                if (xi == 0):
                    regr = LinearRegression().fit(x_train, y_train)
                    aList.append(np.absolute(y_test.iloc[0]['cantidad_sum'] - np.array(regr.predict(x_test)).item()) /
                                 y_test.iloc[0]['cantidad_sum'])

                elif (xi == 1):
                    tsr = TheilSenRegressor().fit(x_train, y_train)
                    aList.append(np.absolute(y_test.iloc[0]['cantidad_sum'] - np.array(tsr.predict(x_test)[0]).item()) /
                                 y_test.iloc[0]['cantidad_sum'])

                elif (xi == 2):
                    gbr = GradientBoostingRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
                    aList.append(np.absolute(y_test.iloc[0]['cantidad_sum'] - np.array(gbr.predict(x_test)[0]).item()) /
                                 y_test.iloc[0]['cantidad_sum'])

                elif (xi == 3):
                    ext = ExtraTreesRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
                    aList.append(np.absolute(y_test.iloc[0]['cantidad_sum'] - np.array(ext.predict(x_test)[0]).item()) /
                                 y_test.iloc[0]['cantidad_sum'])

                elif (xi == 4):
                    ab = AdaBoostRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
                    aList.append(np.absolute(y_test.iloc[0]['cantidad_sum'] - np.array(ab.predict(x_test)[0]).item()) /
                                 y_test.iloc[0]['cantidad_sum'])

                elif (xi == 5):
                    bag = BaggingRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
                    aList.append(np.absolute(y_test.iloc[0]['cantidad_sum'] - np.array(bag.predict(x_test)[0]).item()) /
                                 y_test.iloc[0]['cantidad_sum'])
            myVar = np.var(aList)
            for iv in range(len(var_column)):
                var_iv = var_column[iv]
                if(var_iv[2] > myVar):
                    var_column[iv] = [tam_ventana,columns,myVar]
                    print(var_column[iv])
                    break
    print("-----------------------------------------------------------------")
    print(var_column)

if __name__ == "__main__":
    main()
