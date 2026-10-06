#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 23:30:05 2026

@author: josejuncol
"""

# -*- coding: utf-8 -*-

"""
EXTRACCION AUTOMATICA DE WRFOUT A EXCEL
=======================================

Estructura esperada:

CARPETA_DEL_SCRIPT/
│
├── extraer_wrf_excel.py
│
├── d01/
│   ├── wrfout_d01_...
│   └── ...
│
├── d02/
│   ├── wrfout_d02_...
│   └── ...
│
└── d03/
    ├── wrfout_d03_...
    └── ...

El programa:

1. Busca automáticamente todos los wrfout de d01, d02 y d03.
2. Los ordena.
3. Localiza la celda WRF más cercana al punto objetivo.
4. Extrae variables superficiales.
5. Extrae variables tridimensionales.
6. Destaggeriza U, V, W, PH y PHB.
7. Convierte UTC -> hora local de Lima (UTC-5).
8. Detecta fechas duplicadas.
9. Detecta saltos temporales.
10. Genera un Excel por dominio.

Autor: José Junco / EPALIFE
"""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from netCDF4 import Dataset


# ============================================================
# CONFIGURACION
# ============================================================

# El script buscará las carpetas d01, d02 y d03
# en el mismo directorio donde se encuentra este archivo .py

BASE_DIR = Path(__file__).resolve().parent

DOMINIOS = ["d01", "d02", "d03"]


# ------------------------------------------------------------
# PUNTO OBJETIVO
# Aeropuerto Internacional Jorge Chávez
# ------------------------------------------------------------

LAT_OBJETIVO = -12.0167
LON_OBJETIVO = -77.1167


# ------------------------------------------------------------
# Zona horaria Lima
#
# WRF normalmente almacena los tiempos en UTC.
# Lima = UTC - 5
# ------------------------------------------------------------

UTC_OFFSET_LIMA = -5


# ============================================================
# VARIABLES
# ============================================================

VARIABLES_SUPERFICIE = [

    "T2",
    "Q2",
    "U10",
    "V10",
    "PSFC",
    "PBLH",
    "HFX",
    "LH",
    "SWDOWN",
    "GLW",
    "RAINNC",
    "RAINC",
    "SST",
    "TSK",
    "LANDMASK",
    "LU_INDEX",
    "HGT",

]


VARIABLES_3D = [

    "P",
    "PB",
    "T",
    "QVAPOR",
    "U",
    "V",
    "W",
    "PH",
    "PHB",

]


DESCRIPCIONES = {

    "T2":       "Temperatura a 2 m",
    "Q2":       "Razón de mezcla a 2 m",
    "U10":      "Viento U a 10 m",
    "V10":      "Viento V a 10 m",
    "PSFC":     "Presión superficial",
    "PBLH":     "Altura de la capa límite",
    "HFX":      "Flujo de calor sensible",
    "LH":       "Flujo de calor latente",
    "SWDOWN":   "Radiación solar descendente",
    "GLW":      "Radiación de onda larga descendente",
    "RAINNC":   "Precipitación no convectiva acumulada",
    "RAINC":    "Precipitación convectiva acumulada",
    "SST":      "Temperatura superficial del mar",
    "TSK":      "Temperatura de la superficie",
    "LANDMASK": "Máscara tierra/mar",
    "LU_INDEX": "Categoría de uso de suelo",
    "HGT":      "Altura del terreno",

    "P":        "Perturbación de presión",
    "PB":       "Presión base",
    "T":        "Temperatura potencial perturbada",
    "QVAPOR":   "Vapor de agua",
    "U":        "Componente U",
    "V":        "Componente V",
    "W":        "Velocidad vertical",
    "PH":       "Geopotencial perturbado",
    "PHB":      "Geopotencial base",

}


# ============================================================
# FUNCIONES
# ============================================================

def leer_times(nc):
    """
    Lee la variable Times de un wrfout.
    """

    if "Times" not in nc.variables:
        return []

    datos = nc.variables["Times"][:]

    fechas = []

    for fila in datos:

        caracteres = []

        for c in fila:

            if isinstance(c, bytes):
                caracteres.append(c.decode("utf-8"))
            else:
                caracteres.append(str(c))

        texto = "".join(caracteres).strip()

        try:

            fecha = datetime.strptime(
                texto,
                "%Y-%m-%d_%H:%M:%S"
            )

            fechas.append(fecha)

        except ValueError:

            print(
                f"ADVERTENCIA: no se pudo interpretar "
                f"la fecha {texto}"
            )

    return fechas


# ------------------------------------------------------------

def distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Distancia aproximada entre dos puntos en kilómetros.
    """

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(np.sqrt(a))

    return R * c


# ------------------------------------------------------------

def buscar_celda_mas_cercana(nc):
    """
    Encuentra j,i de la celda WRF más cercana
    a LAT_OBJETIVO, LON_OBJETIVO.
    """

    lat = np.asarray(
        nc.variables["XLAT"][0, :, :]
    )

    lon = np.asarray(
        nc.variables["XLONG"][0, :, :]
    )

    distancia = distancia_haversine(
        LAT_OBJETIVO,
        LON_OBJETIVO,
        lat,
        lon
    )

    j, i = np.unravel_index(
        np.nanargmin(distancia),
        distancia.shape
    )

    return (
        int(j),
        int(i),
        float(lat[j, i]),
        float(lon[j, i]),
        float(distancia[j, i])
    )


# ------------------------------------------------------------

def destagger(arr, eje):
    """
    Convierte una variable escalonada de WRF
    al centro de las celdas mediante promedio.

    eje:
        1 -> vertical
        2 -> south_north
        3 -> west_east

    Se supone array con forma típica:
        Time, vertical, south_north, west_east
    """

    sl1 = [slice(None)] * arr.ndim
    sl2 = [slice(None)] * arr.ndim

    sl1[eje] = slice(0, -1)
    sl2[eje] = slice(1, None)

    return 0.5 * (
        arr[tuple(sl1)] +
        arr[tuple(sl2)]
    )


# ------------------------------------------------------------

def valor_superficie(var, tiempo, j, i):
    """
    Extrae una variable superficial.
    """

    dims = var.dimensions

    # Casos habituales:
    # Time, south_north, west_east

    if len(dims) == 3:

        return float(
            np.asarray(var[tiempo, j, i])
        )

    # Variable sin Time:
    # south_north, west_east

    elif len(dims) == 2:

        return float(
            np.asarray(var[j, i])
        )

    # Algún caso poco habitual

    else:

        dato = np.asarray(var[tiempo])

        return float(
            np.ravel(dato)[0]
        )


# ------------------------------------------------------------

def extraer_perfil_variable(
        nc,
        nombre,
        tiempo,
        j,
        i
):
    """
    Extrae un perfil vertical en el punto seleccionado.

    Realiza destaggering para:
      U
      V
      W
      PH
      PHB
    """

    var = nc.variables[nombre]

    # --------------------------------------------------------
    # Variables normales:
    # P, PB, T, QVAPOR
    #
    # dimensiones:
    # Time, bottom_top, south_north, west_east
    # --------------------------------------------------------

    if nombre in ["P", "PB", "T", "QVAPOR"]:

        return np.asarray(
            var[tiempo, :, j, i],
            dtype=float
        )


    # --------------------------------------------------------
    # U
    #
    # west_east_stag
    # --------------------------------------------------------

    elif nombre == "U":

        # Promedio entre los dos puntos U
        # que rodean el centro de la celda

        a = np.asarray(
            var[tiempo, :, j, i],
            dtype=float
        )

        b = np.asarray(
            var[tiempo, :, j, i + 1],
            dtype=float
        )

        return 0.5 * (a + b)


    # --------------------------------------------------------
    # V
    #
    # south_north_stag
    # --------------------------------------------------------

    elif nombre == "V":

        a = np.asarray(
            var[tiempo, :, j, i],
            dtype=float
        )

        b = np.asarray(
            var[tiempo, :, j + 1, i],
            dtype=float
        )

        return 0.5 * (a + b)


    # --------------------------------------------------------
    # W
    #
    # bottom_top_stag
    # --------------------------------------------------------

    elif nombre == "W":

        valores = np.asarray(
            var[tiempo, :, j, i],
            dtype=float
        )

        return 0.5 * (
            valores[:-1] +
            valores[1:]
        )


    # --------------------------------------------------------
    # PH y PHB
    #
    # geopotencial en niveles staggered verticalmente
    # --------------------------------------------------------

    elif nombre in ["PH", "PHB"]:

        valores = np.asarray(
            var[tiempo, :, j, i],
            dtype=float
        )

        return 0.5 * (
            valores[:-1] +
            valores[1:]
        )


    else:

        return np.asarray(
            var[tiempo, :, j, i],
            dtype=float
        )


# ============================================================
# PROCESAR DOMINIO
# ============================================================

def procesar_dominio(dominio):

    print("\n")
    print("=" * 75)
    print(f"PROCESANDO DOMINIO {dominio.upper()}")
    print("=" * 75)

    carpeta = BASE_DIR / dominio

    if not carpeta.exists():

        print(
            f"\nNo existe la carpeta:\n{carpeta}"
        )

        return


    # ========================================================
    # BUSCAR ARCHIVOS
    # ========================================================

    patron = f"wrfout_{dominio}_*"

    archivos = sorted(
        p for p in carpeta.glob(patron)
        if p.is_file()
    )


    if len(archivos) == 0:

        print(
            f"\nNo se encontraron archivos {patron}"
        )

        return


    print(
        f"\nArchivos encontrados: {len(archivos)}"
    )

    for n, archivo in enumerate(
            archivos,
            start=1
    ):

        print(
            f"{n:3d}. {archivo.name}"
        )


    # ========================================================
    # PRIMER ARCHIVO
    # ========================================================

    with Dataset(str(archivos[0]), "r") as nc:

        (
            j,
            i,
            lat_real,
            lon_real,
            distancia_km

        ) = buscar_celda_mas_cercana(nc)


        dx = getattr(
            nc,
            "DX",
            np.nan
        )

        dy = getattr(
            nc,
            "DY",
            np.nan
        )


        nx = len(
            nc.dimensions.get(
                "west_east",
                []
            )
        )

        ny = len(
            nc.dimensions.get(
                "south_north",
                []
            )
        )

        nz = len(
            nc.dimensions.get(
                "bottom_top",
                []
            )
        )


        variables_disponibles = set(
            nc.variables.keys()
        )


    print("\nPunto objetivo:")
    print(
        f"  Latitud  = {LAT_OBJETIVO:.6f}"
    )
    print(
        f"  Longitud = {LON_OBJETIVO:.6f}"
    )

    print("\nCelda WRF seleccionada:")
    print(
        f"  j = {j}"
    )
    print(
        f"  i = {i}"
    )
    print(
        f"  Latitud  = {lat_real:.6f}"
    )
    print(
        f"  Longitud = {lon_real:.6f}"
    )
    print(
        f"  Distancia = {distancia_km:.3f} km"
    )

    print("\nResolución:")
    print(
        f"  DX = {dx / 1000:.3f} km"
    )
    print(
        f"  DY = {dy / 1000:.3f} km"
    )


    # ========================================================
    # LISTAS DE RESULTADOS
    # ========================================================

    superficie = []
    perfiles = []

    lista_archivos = []

    variables_faltantes = set()

    contador_tiempos = 0


    # ========================================================
    # LEER TODOS LOS ARCHIVOS
    # ========================================================

    for numero, archivo in enumerate(
        archivos,
        start=1
    ):

        print("\n" + "-" * 75)

        print(
            f"[{numero}/{len(archivos)}] "
            f"Leyendo: {archivo.name}"
        )

        try:

            nc = Dataset(
                str(archivo),
                "r"
            )

        except Exception as e:

            print(
                f"ERROR abriendo archivo:\n{e}"
            )

            continue


        try:

            fechas = leer_times(nc)

            if len(fechas) == 0:

                print(
                    "No se encontraron tiempos."
                )

                nc.close()

                continue


            print(
                f"  Tiempos = {len(fechas)}"
            )

            print(
                f"  Inicio  = {fechas[0]}"
            )

            print(
                f"  Final   = {fechas[-1]}"
            )


            lista_archivos.append({

                "Archivo": archivo.name,
                "N_tiempos": len(fechas),
                "Fecha_inicial_UTC": fechas[0],
                "Fecha_final_UTC": fechas[-1],

            })


            # =================================================
            # RECORRER TIEMPOS
            # =================================================

            for it, fecha_utc in enumerate(fechas):

                contador_tiempos += 1

                fecha_lima = (
                    fecha_utc
                    +
                    timedelta(
                        hours=UTC_OFFSET_LIMA
                    )
                )


                # =============================================
                # SUPERFICIE
                # =============================================

                fila = {

                    "Fecha_UTC": fecha_utc,
                    "Fecha_Lima": fecha_lima,

                }


                for nombre in VARIABLES_SUPERFICIE:

                    if nombre not in nc.variables:

                        fila[nombre] = np.nan

                        variables_faltantes.add(
                            nombre
                        )

                        continue


                    try:

                        fila[nombre] = (
                            valor_superficie(
                                nc.variables[nombre],
                                it,
                                j,
                                i
                            )
                        )

                    except Exception:

                        fila[nombre] = np.nan

                        variables_faltantes.add(
                            nombre
                        )


                superficie.append(fila)


                # =============================================
                # VARIABLES 3D
                # =============================================

                perfiles_actuales = {}


                n_niveles = None


                for nombre in VARIABLES_3D:

                    if nombre not in nc.variables:

                        variables_faltantes.add(
                            nombre
                        )

                        continue


                    try:

                        valores = (
                            extraer_perfil_variable(
                                nc,
                                nombre,
                                it,
                                j,
                                i
                            )
                        )

                        perfiles_actuales[nombre] = (
                            valores
                        )

                        if n_niveles is None:

                            n_niveles = len(
                                valores
                            )

                    except Exception as e:

                        print(
                            f"  Advertencia "
                            f"{nombre}: {e}"
                        )


                if n_niveles is not None:

                    for k in range(n_niveles):

                        fila3d = {

                            "Fecha_UTC":
                                fecha_utc,

                            "Fecha_Lima":
                                fecha_lima,

                            "Nivel_WRF":
                                k + 1,

                        }


                        for nombre in VARIABLES_3D:

                            valores = (
                                perfiles_actuales.get(
                                    nombre
                                )
                            )

                            if (
                                valores is not None
                                and
                                k < len(valores)
                            ):

                                fila3d[nombre] = (
                                    valores[k]
                                )

                            else:

                                fila3d[nombre] = (
                                    np.nan
                                )


                        # -------------------------------------
                        # VARIABLES DERIVADAS UTILES
                        # -------------------------------------

                        # Presión total
                        if (
                            "P" in fila3d
                            and
                            "PB" in fila3d
                        ):

                            fila3d["P_TOTAL"] = (
                                fila3d["P"]
                                +
                                fila3d["PB"]
                            )

                        else:

                            fila3d["P_TOTAL"] = (
                                np.nan
                            )


                        # Temperatura potencial total
                        #
                        # En WRF:
                        # T = theta - 300 K

                        if "T" in fila3d:

                            fila3d["THETA"] = (
                                fila3d["T"]
                                + 300.0
                            )

                        else:

                            fila3d["THETA"] = (
                                np.nan
                            )


                        # Altura geopotencial
                        #
                        # z = (PH + PHB) / g

                        if (
                            "PH" in fila3d
                            and
                            "PHB" in fila3d
                        ):

                            fila3d["ALTURA_GEO_M"] = (

                                (
                                    fila3d["PH"]
                                    +
                                    fila3d["PHB"]
                                )
                                /
                                9.81
                            )

                        else:

                            fila3d[
                                "ALTURA_GEO_M"
                            ] = np.nan


                        perfiles.append(
                            fila3d
                        )


        finally:

            nc.close()


    # ========================================================
    # DATAFRAMES
    # ========================================================

    df_sup = pd.DataFrame(
        superficie
    )

    df_3d = pd.DataFrame(
        perfiles
    )

    df_archivos = pd.DataFrame(
        lista_archivos
    )


    if df_sup.empty:

        print(
            "\nNo se pudieron extraer datos."
        )

        return


    # ========================================================
    # ORDENAR
    # ========================================================

    df_sup = (
        df_sup
        .sort_values(
            "Fecha_UTC"
        )
        .reset_index(
            drop=True
        )
    )


    if not df_3d.empty:

        df_3d = (
            df_3d
            .sort_values(
                [
                    "Fecha_UTC",
                    "Nivel_WRF"
                ]
            )
            .reset_index(
                drop=True
            )
        )


    # ========================================================
    # DUPLICADOS TEMPORALES
    # ========================================================

    duplicados = (
        df_sup[
            df_sup.duplicated(
                subset="Fecha_UTC",
                keep=False
            )
        ]
        .copy()
    )


    n_duplicados = (
        df_sup.duplicated(
            subset="Fecha_UTC"
        ).sum()
    )


    if n_duplicados > 0:

        print(
            f"\nADVERTENCIA: "
            f"{n_duplicados} fechas duplicadas."
        )

        print(
            "Se conservará únicamente "
            "la primera aparición."
        )


        df_sup = (
            df_sup
            .drop_duplicates(
                subset="Fecha_UTC",
                keep="first"
            )
            .reset_index(
                drop=True
            )
        )


        if not df_3d.empty:

            df_3d = (
                df_3d
                .drop_duplicates(
                    subset=[
                        "Fecha_UTC",
                        "Nivel_WRF"
                    ],
                    keep="first"
                )
                .reset_index(
                    drop=True
                )
            )


    # ========================================================
    # ANALISIS TEMPORAL
    # ========================================================

    fechas_unicas = (
        df_sup["Fecha_UTC"]
        .sort_values()
        .reset_index(
            drop=True
        )
    )


    saltos = []


    if len(fechas_unicas) > 1:

        diferencias = (
            fechas_unicas
            .diff()
            .dropna()
        )


        segundos = (
            diferencias
            .dt
            .total_seconds()
        )


        intervalo_seg = (
            segundos
            .median()
        )


        intervalo_horas = (
            intervalo_seg / 3600.0
        )


        print(
            f"\nIntervalo temporal típico: "
            f"{intervalo_horas:.3f} h"
        )


        for idx in range(
            1,
            len(fechas_unicas)
        ):

            dt = (

                fechas_unicas.iloc[idx]
                -
                fechas_unicas.iloc[idx - 1]

            )

            if abs(
                dt.total_seconds()
                -
                intervalo_seg
            ) > 1:

                saltos.append({

                    "Fecha_anterior":
                        fechas_unicas.iloc[
                            idx - 1
                        ],

                    "Fecha_siguiente":
                        fechas_unicas.iloc[
                            idx
                        ],

                    "Diferencia_horas":
                        dt.total_seconds()
                        /
                        3600.0,

                })


    else:

        intervalo_horas = np.nan


    df_saltos = pd.DataFrame(
        saltos
    )


    # ========================================================
    # INFORMACION DEL DOMINIO
    # ========================================================

    info = [

        ["Dominio", dominio],

        [
            "Latitud objetivo",
            LAT_OBJETIVO
        ],

        [
            "Longitud objetivo",
            LON_OBJETIVO
        ],

        [
            "Latitud celda WRF",
            lat_real
        ],

        [
            "Longitud celda WRF",
            lon_real
        ],

        [
            "Distancia punto-celda (km)",
            distancia_km
        ],

        [
            "Índice i",
            i
        ],

        [
            "Índice j",
            j
        ],

        [
            "DX (m)",
            dx
        ],

        [
            "DY (m)",
            dy
        ],

        [
            "NX",
            nx
        ],

        [
            "NY",
            ny
        ],

        [
            "N niveles",
            nz
        ],

        [
            "Número de archivos",
            len(archivos)
        ],

        [
            "Número tiempos finales",
            len(df_sup)
        ],

        [
            "Fecha inicial UTC",
            df_sup[
                "Fecha_UTC"
            ].min()
        ],

        [
            "Fecha final UTC",
            df_sup[
                "Fecha_UTC"
            ].max()
        ],

        [
            "Fecha inicial Lima",
            df_sup[
                "Fecha_Lima"
            ].min()
        ],

        [
            "Fecha final Lima",
            df_sup[
                "Fecha_Lima"
            ].max()
        ],

        [
            "Intervalo temporal (h)",
            intervalo_horas
        ],

        [
            "Fechas duplicadas encontradas",
            int(n_duplicados)
        ],

        [
            "Saltos temporales encontrados",
            len(saltos)
        ],

    ]


    df_info = pd.DataFrame(
        info,
        columns=[
            "Parámetro",
            "Valor"
        ]
    )


    # ========================================================
    # METADATOS DE VARIABLES
    # ========================================================

    metadata = []


    with Dataset(
        str(archivos[0]),
        "r"
    ) as nc:


        for nombre in (
            VARIABLES_SUPERFICIE
            +
            VARIABLES_3D
        ):

            if nombre in nc.variables:

                var = nc.variables[
                    nombre
                ]

                metadata.append({

                    "Variable":
                        nombre,

                    "Descripción":
                        DESCRIPCIONES.get(
                            nombre,
                            ""
                        ),

                    "Unidades":
                        getattr(
                            var,
                            "units",
                            ""
                        ),

                    "Dimensiones":
                        ", ".join(
                            var.dimensions
                        ),

                    "Shape":
                        str(
                            var.shape
                        ),

                })

            else:

                metadata.append({

                    "Variable":
                        nombre,

                    "Descripción":
                        DESCRIPCIONES.get(
                            nombre,
                            ""
                        ),

                    "Unidades":
                        "NO DISPONIBLE",

                    "Dimensiones":
                        "",

                    "Shape":
                        "",

                })


    # Añadir variables derivadas

    metadata.extend([

        {

            "Variable":
                "P_TOTAL",

            "Descripción":
                "Presión total P + PB",

            "Unidades":
                "Pa",

            "Dimensiones":
                "perfil",

            "Shape":
                "",

        },

        {

            "Variable":
                "THETA",

            "Descripción":
                "Temperatura potencial total T + 300",

            "Unidades":
                "K",

            "Dimensiones":
                "perfil",

            "Shape":
                "",

        },

        {

            "Variable":
                "ALTURA_GEO_M",

            "Descripción":
                "Altura geopotencial (PH+PHB)/9.81",

            "Unidades":
                "m",

            "Dimensiones":
                "perfil",

            "Shape":
                "",

        },

    ])


    df_metadata = pd.DataFrame(
        metadata
    )


    # ========================================================
    # CREAR EXCEL
    # ========================================================

    salida = (
        BASE_DIR
        /
        f"WRF_{dominio}_Jorge_Chavez.xlsx"
    )


    print("\nGenerando Excel:")
    print(salida)


    with pd.ExcelWriter(
        salida,
        engine="xlsxwriter",
        datetime_format="dd/mm/yyyy hh:mm"
    ) as writer:


        # ----------------------------------------------------
        # HOJA SUPERFICIE
        # ----------------------------------------------------

        df_sup.to_excel(
            writer,
            sheet_name="Superficie",
            index=False
        )


        # ----------------------------------------------------
        # HOJA PERFIL 3D
        # ----------------------------------------------------

        if not df_3d.empty:

            df_3d.to_excel(
                writer,
                sheet_name="Perfil_3D",
                index=False
            )


        # ----------------------------------------------------
        # INFORMACION
        # ----------------------------------------------------

        df_info.to_excel(
            writer,
            sheet_name="Info_WRF",
            index=False
        )


        # ----------------------------------------------------
        # ARCHIVOS
        # ----------------------------------------------------

        df_archivos.to_excel(
            writer,
            sheet_name="Archivos",
            index=False
        )


        # ----------------------------------------------------
        # VARIABLES
        # ----------------------------------------------------

        df_metadata.to_excel(
            writer,
            sheet_name="Variables",
            index=False
        )


        # ----------------------------------------------------
        # DUPLICADOS
        # ----------------------------------------------------

        if not duplicados.empty:

            duplicados.to_excel(
                writer,
                sheet_name="Duplicados",
                index=False
            )


        # ----------------------------------------------------
        # SALTOS
        # ----------------------------------------------------

        if not df_saltos.empty:

            df_saltos.to_excel(
                writer,
                sheet_name="Saltos_Temporales",
                index=False
            )


        # ====================================================
        # FORMATO EXCEL
        # ====================================================

        workbook = writer.book


        formato_header = (
            workbook.add_format({

                "bold": True,
                "border": 1,
                "align": "center",
                "valign": "vcenter",

            })
        )


        formato_num = (
            workbook.add_format({

                "num_format":
                    "0.000000",

            })
        )


        formato_fecha = (
            workbook.add_format({

                "num_format":
                    "dd/mm/yyyy hh:mm",

            })
        )


        # ----------------------------------------------------
        # FORMATEAR CADA HOJA
        # ----------------------------------------------------

        for nombre_hoja, dataframe in [

            ("Superficie", df_sup),

            ("Perfil_3D", df_3d),

            ("Info_WRF", df_info),

            ("Archivos", df_archivos),

            ("Variables", df_metadata),

        ]:

            if (
                nombre_hoja
                not in writer.sheets
            ):
                continue


            worksheet = writer.sheets[
                nombre_hoja
            ]


            worksheet.freeze_panes(
                1,
                0
            )


            worksheet.autofilter(
                0,
                0,
                len(dataframe),
                max(
                    len(
                        dataframe.columns
                    ) - 1,
                    0
                )
            )


            # Encabezados

            for col_num, value in enumerate(
                dataframe.columns.values
            ):

                worksheet.write(
                    0,
                    col_num,
                    value,
                    formato_header
                )


            # Anchos

            for idx, columna in enumerate(
                dataframe.columns
            ):

                if "Fecha" in str(columna):

                    worksheet.set_column(
                        idx,
                        idx,
                        20,
                        formato_fecha
                    )

                elif columna in [
                    "Archivo"
                ]:

                    worksheet.set_column(
                        idx,
                        idx,
                        42
                    )

                else:

                    worksheet.set_column(
                        idx,
                        idx,
                        17
                    )


    # ========================================================
    # RESUMEN
    # ========================================================

    print("\n" + "=" * 75)

    print(
        f"DOMINIO {dominio.upper()} FINALIZADO"
    )

    print("=" * 75)

    print(
        f"Archivo Excel:\n{salida}"
    )

    print(
        f"\nTiempos finales : {len(df_sup)}"
    )

    print(
        f"Inicio UTC       : "
        f"{df_sup['Fecha_UTC'].min()}"
    )

    print(
        f"Final UTC        : "
        f"{df_sup['Fecha_UTC'].max()}"
    )

    print(
        f"Inicio Lima      : "
        f"{df_sup['Fecha_Lima'].min()}"
    )

    print(
        f"Final Lima       : "
        f"{df_sup['Fecha_Lima'].max()}"
    )


    if variables_faltantes:

        print(
            "\nVariables no encontradas:"
        )

        for v in sorted(
            variables_faltantes
        ):

            print(
                f"  - {v}"
            )

    else:

        print(
            "\nTodas las variables "
            "solicitadas fueron encontradas."
        )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("EXTRACCION WRF -> EXCEL")
    print("=" * 75)

    print(
        f"\nDirectorio base:\n{BASE_DIR}"
    )

    print(
        "\nPunto de extracción:"
    )

    print(
        f"Lat = {LAT_OBJETIVO}"
    )

    print(
        f"Lon = {LON_OBJETIVO}"
    )


    for dominio in DOMINIOS:

        try:

            procesar_dominio(
                dominio
            )

        except Exception as e:

            print("\n")
            print("!" * 75)

            print(
                f"ERROR PROCESANDO "
                f"{dominio.upper()}"
            )

            print(
                str(e)
            )

            print("!" * 75)


    print("\n")
    print("=" * 75)
    print("PROCESO TERMINADO")
    print("=" * 75)


if __name__ == "__main__":

    main()