#!/usr/bin/python
import psycopg2
import pandas as pd

DBNAME_VIDRIERIA = 'vidrieria'
USERNAME = 'postgres'
PASSWORD = 'postgres'
HOST = 'localhost'

def getTableVidrieriaDiario(id):
    # Define our connection string
    conn_string = "host='{}' dbname='{}' user='{}' password='{}'".format(HOST,DBNAME_VIDRIERIA,USERNAME,PASSWORD)

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    df = pd.read_sql_query("select "
                    "extract(day from venta.fechafactura::date) as dia, "
                    "extract(doy from venta.fechafactura::date) as dia_anho, "  
                    "extract(dow from venta.fechafactura::date) as dia_semana, "  
                    "extract(week from venta.fechafactura::date) as semana_anho, "  
                    "extract(month from venta.fechafactura::date) as mes, "
                    "extract(year from venta.fechafactura::date) as anho, "        
                    "extract(quarter from venta.fechafactura::date) as cuatrimestre, "        
                    "sum(ventadetalle.cantidadproducto) as cantidad, "
                    "avg(ventadetalle.precioproducto) as precioproducto_mean, "
                    "max(ventadetalle.precioproducto) as precioproducto_max, " 
                    "min(ventadetalle.precioproducto) as precioproducto_min, " 
                    "util.meteorologico.temperatura_maxima, "                            
                    "util.meteorologico.temperatura_minima, "
                    "util.meteorologico.temperatura_media, "
                    "util.meteorologico.presion_atmosferica_media, "
                    "util.meteorologico.velocidad_viento, "
                    "util.meteorologico.precipitacion "
                    "from ventadetalle inner join venta  "
                    "on ventadetalle.idventa = venta.id and venta.anulado = FALSE "
                    "inner join producto on ventadetalle.idproducto = producto.id "
                    "inner join util.meteorologico on fecha = venta.fechafactura::date "
                    "where  ventadetalle.idproducto = "+str(id)+" "
                    "group by venta.fechafactura::date, mes,anho ,dia, "                           
                    "dia_anho, dia_semana, semana_anho, " 
                    "temperatura_maxima, temperatura_minima, temperatura_media, "   
                    "presion_atmosferica_media, velocidad_viento, precipitacion "   
                    "order by anho asc,cuatrimestre asc, mes asc, dia asc ", con=conn)

    return df;

def getTableVidrieriaMensual(id):
    # Define our connection string
    conn_string = "host='{}' dbname='{}' user='{}' password='{}'".format(HOST,DBNAME_VIDRIERIA,USERNAME,PASSWORD)

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    df = pd.read_sql_query("select "                         
                    "extract(month from venta.fechafactura::date) as mes, "
                    "extract(year from venta.fechafactura::date) as anho, "        
                    "extract(quarter from venta.fechafactura::date) as cuatrimestre, "        
                    "sum(ventadetalle.cantidadproducto) as cantidad, "
                    "avg(ventadetalle.precioproducto) as precioproducto_mean, "
                    "max(ventadetalle.precioproducto) as precioproducto_max, " 
                    "min(ventadetalle.precioproducto) as precioproducto_min " 
                    "from ventadetalle inner join venta  "
                    "on ventadetalle.idventa = venta.id and venta.anulado = FALSE "
                    "inner join producto on ventadetalle.idproducto = producto.id "
                    "where  ventadetalle.idproducto = "+str(id)+" "
                    "group by venta.fechafactura::date, mes,anho "                 
                    "order by anho asc,cuatrimestre asc, mes asc ", con=conn)

    return df;