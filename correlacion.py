#!/usr/bin/python
from connect import *
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
    # Lista de los nombres de los productos a analizar
    IDs = [('ESPEJO_INCOLORO_Optimirror', 38),
           ('VIDRIO_INCOLORO_(BAJA)', 2),
           ('LAMINADO_INCOLORO_(Baja)', 497),
           ('VIDRIO_INCOLORO_(ALTA)', 16),
           ('VIDRIO_GRIS_(BAJA)', 23)]

    df_list = []
    for nombre, id in IDs:
        #Extracción de datos
        df = getTableVidrieriaDiario(id)
        df_list.append(df.corr(method ='pearson'))
        #spamwriter.writerow(df.corr())
    pd.concat(df_list).to_csv('/home/rodrigo/Desktop/Tesis/pruebas/correlacion ' + fecha + '.csv')

if __name__ == "__main__":
    main()

