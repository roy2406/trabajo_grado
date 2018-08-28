#!/usr/bin/python
import psycopg2
import pandas as pd

DBNAME_IMPORTADORA = 'importadora'
DBNAME_VIDRIERIA = 'vidrieria'
USERNAME = 'postgres'
PASSWORD = 'postgres'
HOST = 'localhost'

def getTableImportadoraMensual():
    # Define our connection string
    conn_string = "host='{}' dbname='{}' user='{}' password='{}'".format(HOST,DBNAME_IMPORTADORA,USERNAME,PASSWORD)

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    df = pd.read_sql_query("select "
                   "extract(month from vtw_comprobantes_cabecera.fec_alta) as mes, "
                   "extract(quarter from vtw_comprobantes_cabecera.fec_alta) as cuatrimestre, "                           
                   "extract(year from vtw_comprobantes_cabecera.fec_alta) as anho, "
                   "avg(vtw_comprobantes_detalle.precio_unitario) as precioproducto, "
                   "sum(vtw_comprobantes_detalle.cantidad) as cantidad "                           
                   "from vtw_comprobantes_detalle inner join vtw_comprobantes_cabecera "
                   "on vtw_comprobantes_detalle.id_comprobante_cabecera = vtw_comprobantes_cabecera.id "
                   "and vtw_comprobantes_detalle.id_articulo = 60069 "
                   "group by extract(month from vtw_comprobantes_cabecera.fec_alta), "
                   "extract(year from vtw_comprobantes_cabecera.fec_alta), "
                   "extract(quarter from vtw_comprobantes_cabecera.fec_alta) "                           
                   "order by anho, cuatrimestre, mes ", con=conn)

    return df;

def getTableImportadoraTrimestre():
    # Define our connection string
    conn_string = "host='{}' dbname='{}' user='{}' password='{}'".format(HOST,DBNAME_IMPORTADORA,USERNAME,PASSWORD)

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    df = pd.read_sql_query("select "
                   "extract(quarter from vtw_comprobantes_cabecera.fec_alta) as cuatrimestre, "                           
                   "extract(year from vtw_comprobantes_cabecera.fec_alta) as anho, "
                   "avg(vtw_comprobantes_detalle.precio_unitario) as precioproducto, "    
                   "sum(vtw_comprobantes_detalle.cantidad) as cantidad "    
                   "from vtw_comprobantes_detalle inner join vtw_comprobantes_cabecera "
                   "on vtw_comprobantes_detalle.id_comprobante_cabecera = vtw_comprobantes_cabecera.id "
                   "and vtw_comprobantes_detalle.id_articulo = 60069 "
                   "group by extract(year from vtw_comprobantes_cabecera.fec_alta), "
                   "extract(quarter from vtw_comprobantes_cabecera.fec_alta) "                           
                   "order by anho, cuatrimestre ", con=conn)

    return df;

def getTableVidrieria():
    # Define our connection string
    conn_string = "host='{}' dbname='{}' user='{}' password='{}'".format(HOST,DBNAME_VIDRIERIA,USERNAME,PASSWORD)

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    df = pd.read_sql_query("select "
                   "venta.fechafactura::date as fecha, "                           
                   "extract(month from venta.fechafactura::date) as mes, "
                   "extract(quarter from venta.fechafactura::date) as cuatrimestre, "                           
                   "extract(year from venta.fechafactura::date) as anho, "
                   "sum(ventadetalle.cantidadproducto) as cantidad, "
                   "sum(ventadetalle.montototal) as monto_total, "
                   "ventadetalle.idproducto, "        
                   "ventadetalle.precioproducto as precioproducto "        
                   "from ventadetalle inner join venta "
                   "on ventadetalle.idventa = venta.id "                         
                   "group by venta.fechafactura::date, "
                   "ventadetalle.idproducto, "
                   "ventadetalle.precioproducto "                           
                   "order by venta.fechafactura::date asc ", con=conn)

    return df;