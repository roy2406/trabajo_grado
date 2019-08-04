#!/usr/bin/python
from connect import *
#from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


N_ESTIMATORS = 100
CV = 7

def correlacion(df):
    cols = list(df.columns.values)
    cm = np.corrcoef(df.values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    plt.show()

def main():
    df = getTableVidrieria()
    filtrado = df[(df['idproducto'] == 38)]
    #print(filtrado.tail(10))
    agrupado = filtrado.groupby(['cuatrimestre','anho'] ).aggregate(
        {'precioproducto': {'precioproducto_mean':np.mean, 'precioproducto_max':np.max, 'precioproducto_min':np.min},
         'cantidad': {'cantidad_sum':np.sum}})

    agrupado = agrupado.reset_index(col_level=1)

    agrupado.columns = agrupado.columns.get_level_values(1)

    agrupado = agrupado.sort_values(by=['anho', 'cuatrimestre'])

    x = pd.DataFrame(agrupado,columns=['anho','precioproducto_mean','precioproducto_min'])
    y = pd.DataFrame(agrupado,columns=['cantidad_sum'])

    x_train = x[:CV]
    x_test = x[CV:]

    y_train = y[:CV]
    y_test = y[CV:]

    # correlacion(agrupado[:CV])

    #cv_lr = np.mean(cross_val_score(LinearRegression(),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    cv_dtr = np.mean(cross_val_score(DecisionTreeRegressor(),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    cv_gbr = np.mean(cross_val_score(GradientBoostingRegressor(n_estimators=N_ESTIMATORS),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    cv_rfr = np.mean(cross_val_score(RandomForestRegressor(n_estimators=N_ESTIMATORS), x_train, y_train.values.ravel(), cv=CV, scoring='neg_mean_absolute_error'))
    cv_ab = np.mean(cross_val_score(AdaBoostRegressor(n_estimators=N_ESTIMATORS),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))
    cv_bag = np.mean(cross_val_score(BaggingRegressor(n_estimators=N_ESTIMATORS),x_train,y_train.values.ravel(),cv=CV,scoring='neg_mean_absolute_error'))

    myList = (cv_dtr,cv_gbr,cv_rfr,cv_ab,cv_bag)
    print(myList)
    x = myList.index(max(myList))

    #if(x == 0):
    #    regr = LinearRegression().fit(x_train, y_train)
    #    print("Linear Regression Mean absolute error: %.2f"
    #        % mean_absolute_error(y_test, regr.predict(x_test)))

    if(x == 0):
        tsr = DecisionTreeRegressor().fit(x_train, y_train)
        print("Decision Tree Regressor Mean absolute error: %.2f"
            % mean_absolute_error(y_test, tsr.predict(x_test)))

    elif(x == 1):
        gbr = GradientBoostingRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
        print("Gradient Boosting Regressor Mean absolute error: %.2f"
            % mean_absolute_error(y_test, gbr.predict(x_test)))

    elif(x == 2):
        ext = RandomForestRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
        print("Random Forest Regressor Mean absolute error: %.2f"
              % mean_absolute_error(y_test, ext.predict(x_test)))

    elif(x == 3):
        ab = AdaBoostRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
        print("Ada Boost Regressor Mean absolute error: %.2f"
            % mean_absolute_error(y_test, ab.predict(x_test)))

    elif(x == 4):
        bag = BaggingRegressor(n_estimators=N_ESTIMATORS).fit(x_train, y_train)
        print("Bagging Regressor Mean absolute error: %.2f"
            % mean_absolute_error(y_test, bag.predict(x_test)))

if __name__ == "__main__":
    main()
