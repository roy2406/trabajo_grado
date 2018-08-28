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

N_ESTIMATORS = 100
CV = 22

def correlacion(df):
    cols = list(df.columns.values)
    cm = np.corrcoef(df.values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    plt.show()

def main():
    df = getTableVidrieria()
    filtrado = df[(df['idproducto'] == 38) & (pd.to_datetime(df['fecha'],format='%Y-%m-%d') < '2018-03-01')]

    agrupado = filtrado.groupby(['mes','cuatrimestre','anho'] ).aggregate(
        {'precioproducto': {'precioproducto_mean':np.mean, 'precioproducto_max':np.max, 'precioproducto_min':np.min},
         'cantidad': {'cantidad_sum':np.sum}})

    agrupado = agrupado.reset_index(col_level=1)

    agrupado.columns = agrupado.columns.get_level_values(1)

    agrupado = agrupado.sort_values(by=['anho', 'mes'])

    correlacion(agrupado)

    x = pd.DataFrame(agrupado,columns=['anho','precioproducto_mean','precioproducto_min'])
    y = pd.DataFrame(agrupado,columns=['cantidad_sum'])

    x_train = x[:22]
    x_test = x[22:]

    y_train = y[:22]
    y_test = y[22:]

    cv_lr = np.mean(cross_val_score(LinearRegression(),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    cv_tsr = np.mean(cross_val_score(TheilSenRegressor(),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    cv_gbr = np.mean(cross_val_score(GradientBoostingRegressor(n_estimators=N_ESTIMATORS),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    cv_ext = np.mean(cross_val_score(ExtraTreesRegressor(n_estimators=N_ESTIMATORS), x_train, y_train.values.ravel(), cv=CV, scoring='neg_mean_absolute_error'))
    cv_ab = np.mean(cross_val_score(AdaBoostRegressor(n_estimators=N_ESTIMATORS),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    cv_bag = np.mean(cross_val_score(BaggingRegressor(n_estimators=N_ESTIMATORS),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    #cv_mlp = np.mean(cross_val_score(MLPRegressor(),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))

    myList = (cv_lr,cv_tsr,cv_gbr,cv_ext,cv_ab,cv_bag)
    print(myList)
    x = myList.index(max(myList))

    if(x == 0):
        regr = LinearRegression().fit(x_train, y_train)
        print("Linear Regression Mean absolute error: %.2f"
            % mean_absolute_error(y_test, regr.predict(x_test)))

    elif(x == 1):
        tsr = TheilSenRegressor().fit(x_train, y_train)
        print("Theil Sen Regressor Mean absolute error: %.2f"
            % mean_absolute_error(y_test, tsr.predict(x_test)))

    elif(x == 2):
        gbr = GradientBoostingRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
        print("Gradient Boosting Regressor Mean absolute error: %.2f"
            % mean_absolute_error(y_test, gbr.predict(x_test)))

    elif(x == 3):
        ext = ExtraTreesRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
        print("Extra Trees Regressor Mean absolute error: %.2f"
              % mean_absolute_error(y_test, ext.predict(x_test)))

    elif(x == 4):
        ab = AdaBoostRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
        print("Ada Boost Regressor Mean absolute error: %.2f"
            % mean_absolute_error(y_test, ab.predict(x_test)))

    elif(x == 5):
        bag = BaggingRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
        print("Bagging Regressor Mean absolute error: %.2f"
            % mean_absolute_error(y_test, bag.predict(x_test)))

    #elif(x == 6):
    #    rf = RandomForestRegressor(n_estimators=1000).fit(x_train, y_train)
    #    print("Random Forest Regressor Mean absolute error: %.2f"
    #        % mean_absolute_error(y_test, rf.predict(x_test)))


if __name__ == "__main__":
    main()
