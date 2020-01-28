# Package dependencies
import numpy as np
import pandas as pd
import collections

def aic(y, y_pred, p):
    """
    Return an AIC score for a model.
    Input:
    y: array-like of shape = (n_samples) including values of observed y
    y_pred: vector including values of predicted y
    p: int number of predictive variable(s) used in the model
    Output:
    aic_score: int or float AIC score of the model
    Raise TypeError if y or y_pred are not list/tuple/dataframe column/array.
    Raise TypeError if elements in y or y_pred are not integer or float.
    Raise TypeError if p is not int.
    Raise InputError if y or y_pred are not in same length.
    Raise InputError if length(y) <= 1 or length(y_pred) <= 1.
    Raise InputError if p < 0.
    """

    # User-defined exceptions
    class InputError(Exception):
        """
        Raised when there is any error from inputs that no base Python exceptions cover.
        """
        pass

    # Check conditions:
    ## Type condition 1: y and y_pred should be array-like containing numbers
    ### check type of y and y_pred
    if isinstance(y, (np.ndarray, list, tuple, pd.core.series.Series)) == False or isinstance(y_pred, (np.ndarray, list, tuple, pd.core.series.Series)) == False:
        raise TypeError("Expect array-like shape (e.g. array, list, tuple, data column)")
    ### check if elements of y and y_pred are numeric
    else:
        for i in y:
            for j in y_pred:
                if isinstance(i, (int, float)) != True or isinstance(j, (int, float)) != True:
                    raise TypeError("Expect numeric elements in y and y_pred")

    ## Type condition 2: p should be positive integer
    ### check if p is integer
    if isinstance(p, int) != True:
        raise TypeError("Expect positive integer")
    ### check if p is positive
    elif p <= 0:
        raise InputError("Expect positive integer")

    ## Length condition: length of y and y_pred should be equal, and should be more than 1
    ### check if y and y_pred have equal length
    if not len(y) == len(y_pred):
        raise InputError("Expect equal length of y and y_pred")
    ### check if y and y_pred length is larger than 1
    elif len(y) <= 1 or len(y_pred) <= 1:
        raise InputError("Expect length of y and y_pred to be larger than 1")
    else:
        n = len(y)

    # Calculation
    resid = np.subtract(y_pred, y)
    rss = np.sum(np.power(resid, 2))
    aic_score = n*np.log(rss/n) + 2*p

    return aic_score

def bic(y, y_pred, p):
    """
    Returns the BIC score of a model.
    Input:-
    y: the labelled data in shape of an array of size  = number of samples.
        type = vector/array/list
    y_pred: predicted values of y from a regression model in shape of an array
        type = vector/array/list
    p: number of variables used for prediction in the model.
        type = int
    Output:-
    score: It outputs the BIC score
        type = int
    Tests:-
    Raise Error if length(y) <= 1 or length(y_pred) <= 1.
    Raise Error if length(y) != length(y_pred).
    Raise TypeError if y and y_pred are not vector.
    Raise TypeError if elements of y or y_pred are not integers.
    Raise TypeError if p is not an int.
    Raise Error if p < 0.
    """


    # Input type error exceptions
    if not isinstance(y, (collections.Sequence, np.ndarray, pd.core.series.Series)):
        raise TypeError("Argument 1 not like an array.")

    if not isinstance(y_pred, (collections.Sequence, np.ndarray, pd.core.series.Series)):
        raise TypeError("Argument 2 not like an array.")

    for i in y:
        if not isinstance(i, (int, float)):
            raise TypeError("All elements of argument 1 must be int or float.")

    for i in y_pred:
        if not isinstance(i, (int, float)):
            raise TypeError("All elements of argument 2 must be int or float.")

    if not isinstance(p, (int, float)):
        raise TypeError("'Number of variables' must be of type int or float.")

    if p <= 0:
        raise TypeError("'Number of variables' must be positive integer.")

    if isinstance(p, int) != True:
        raise TypeError("Expect positive integer")

    if len(y) <= 1 or len(y_pred) <= 1:
        raise TypeError("observed and predicted values must be greater than 1")

    # Length exception
    if not len(y) == len(y_pred):
        raise TypeError("Equal length of observed and predicted values expected.")
    else:
        n = len(y)

    # Score

    residual = np.subtract(y_pred, y)
    SSE = np.sum(np.power(residual, 2))
    BIC = n*np.log(SSE/n) + p*np.log(n)
    return BIC

def check_aic_bic(x, y, estimator):
    CV = np.ceil(y.shape[0] * 0.75).astype(int)
    y_train = y[:CV]
    y_test = y[CV:]

    CV = np.ceil(x.shape[0] * 0.75).astype(int)
    x_train = x[:CV]
    x_test = x[CV:]
    print(":::::::::::::::::::::::::::::::::")
    print("AIC:")
    var_y = np.squeeze(np.asarray(y_test.values))
    var_y_pred = np.squeeze(np.asarray(estimator.fit(x_train, y_train).predict(x_test)))
    print(aic(var_y, var_y_pred, x_test.shape[1]))
    print("BIC:")
    print(bic(var_y, var_y_pred, x_test.shape[1]))
    print(":::::::::::::::::::::::::::::::::")
