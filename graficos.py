#!/usr/bin/python
from connect import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # VIDRIERIA
    df = getTableVidrieria()
    IDs = [38, 2, 497, 16, 23]

    frames = pd.DataFrame()
    for id in IDs:
        filtrado = df[(df['idproducto'] == id)]
        filtrado['fecha'] = pd.to_datetime(filtrado['fecha'], errors='coerce')
        #filtrado['mes'] = filtrado['fecha'].dt.month
        #filtrado['anho'] = filtrado['fecha'].dt.year
        #filtrado['dia'] = filtrado['fecha'].dt.day

        agrupado = filtrado.groupby(['fecha']).aggregate({'cantidad': {'cantidad_sum': np.sum}})
        agrupado = agrupado.reset_index(col_level=1)
        agrupado.columns = agrupado.columns.get_level_values(1)
        print(agrupado)
        #agrupado = agrupado.sort_values(by=['anho', 'mes', 'dia'])
        agrupado.columns = ['fecha','cantidad_'+str(id)]
        frames = pd.concat([frames, agrupado['cantidad_'+str(id)]], axis=1)
    print(frames)
    #plt.scatter(frames['cantidad_38'], frames.index)
    #plt.show()

    frames.plot()
    plt.show()

if __name__ == "__main__":
    main()
