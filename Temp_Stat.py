# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 23:47:54 2026

@author: JOSE
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estadísticas de temperatura WRF
Comparación Observaciones vs d01, d02 y d03

@author: jose
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style


# ============================================================
# CONFIGURACIÓN GRÁFICA
# ============================================================

mpl.rcParams.update({
    # Tamaños de fuente
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 12,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    # Tipografía
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    # Líneas
    "lines.linewidth": 2.8,
    "lines.markersize": 5.5,

    # Grid
    "axes.grid": True,
    "grid.alpha": 0.9,
})


# Estilo general
style.use("fivethirtyeight")

# Resolución de figuras
plt.rcParams["figure.dpi"] = 300


# ============================================================
# LEER DATOS
# ============================================================

data = pd.read_csv(
    "temp_data.csv",
    sep=";",
    parse_dates=["date"],
    dayfirst=True
)

data = data.set_index("date")


# ============================================================
# MOSTRAR INFORMACIÓN DE LOS DATOS
# ============================================================

print("\nDatos cargados:")
print(data.head())

print("\nColumnas:")
print(data.columns.tolist())

print("\nNúmero de registros:", len(data))


# ============================================================
# SERIE DE TIEMPO
# ============================================================

plt.figure(figsize=(13, 6))

plt.plot(
    data["Observations"],
    label="OBS"
)

plt.plot(
    data["d01"],
    label="d01"
)

plt.plot(
    data["d02"],
    label="d02"
)

plt.plot(
    data["d03"],
    label="d03"
)

plt.xlabel("Date")
plt.ylabel("2m Temperature (ºC)")

plt.title(
    "Temp. Time Series Plot - Observations & WRF Domains"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "temp_time_series_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# MÉTRICAS DE ERROR
# ============================================================

rows = []

models = [
    "d01",
    "d02",
    "d03"
]

for model in models:

    obs = data["Observations"]
    pred = data[model]

    # Eliminar pares con valores faltantes
    mask = obs.notna() & pred.notna()

    o = obs[mask]
    p = pred[mask]

    # --------------------------------------------------------
    # Mean Bias
    # --------------------------------------------------------

    mb = (p - o).mean()

    # --------------------------------------------------------
    # Mean Absolute Error
    # --------------------------------------------------------

    mae = (p - o).abs().mean()

    # --------------------------------------------------------
    # Root Mean Square Error
    # --------------------------------------------------------

    rmse = np.sqrt(
        ((p - o) ** 2).mean()
    )

    # --------------------------------------------------------
    # Correlación
    # --------------------------------------------------------

    if len(o) > 1:

        r = np.corrcoef(
            o,
            p
        )[0, 1]

    else:

        r = np.nan


    # Guardar resultados
    rows.append({

        "Model": model,

        "MB": round(
            float(mb),
            2
        ),

        "MAE": round(
            float(mae),
            2
        ),

        "RMSE": round(
            float(rmse),
            2
        ),

        "R": (
            round(float(r), 2)
            if not np.isnan(r)
            else np.nan
        )

    })


# ============================================================
# TABLA DE RESULTADOS
# ============================================================

error_metrics = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "MB",
        "MAE",
        "RMSE",
        "R"
    ]
)


print("\n")
print("=" * 50)
print("Error Metrics")
print("=" * 50)

print(error_metrics)

print("=" * 50)


# Guardar tabla
error_metrics.to_csv(
    "error_metrics.csv",
    index=False
)


# ============================================================
# GRÁFICO DE BARRAS DE MÉTRICAS
# ============================================================

metrics = [
    "MB",
    "MAE",
    "RMSE",
    "R"
]

em_idx = error_metrics.set_index(
    "Model"
)


# Posiciones de las métricas
x = np.arange(
    len(metrics)
)


# Ancho de cada barra
width = 0.25


# Valores de cada dominio
d01_vals = em_idx.loc[
    "d01",
    metrics
].values.astype(float)

d02_vals = em_idx.loc[
    "d02",
    metrics
].values.astype(float)

d03_vals = em_idx.loc[
    "d03",
    metrics
].values.astype(float)


# ============================================================
# CREAR GRÁFICO
# ============================================================

plt.figure(
    figsize=(10, 6)
)


bars1 = plt.bar(
    x - width,
    d01_vals,
    width,
    label="d01"
)

bars2 = plt.bar(
    x,
    d02_vals,
    width,
    label="d02"
)

bars3 = plt.bar(
    x + width,
    d03_vals,
    width,
    label="d03"
)


# Etiquetas del eje X
plt.xticks(
    x,
    metrics
)


plt.ylabel(
    "Value"
)


plt.title(
    "WRF Domains vs Observations: Error Metrics"
)


# Leyenda
leg = plt.legend()

leg.set_title(
    "WRF Domains",
    prop={
        "size": 14
    }
)


plt.tight_layout()


# Guardar gráfico
plt.savefig(
    "temp_error_metrics_plot.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()