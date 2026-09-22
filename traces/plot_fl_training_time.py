#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fl_training_time.py
========================

Calcula y representa el TIEMPO TOTAL DE ENTRENAMIENTO en Federated Learning
(FL) a partir de las trazas de red del simulador (carpeta ``net/``),
desglosado en tiempo de cómputo ("Computing") y retardos de red
("Communication"), como barra apilada.

Definiciones
------------
Para cada cliente ``i`` y cada round ``r``::

    computing(i, r)     = computation-time
    communication(i, r) = client-queue-delay + propagation-delay + station-queue-delay
    arrival(i, r)       = computing(i, r) + communication(i, r)

``arrival`` es el instante en que el modelo local de ``i`` llega al servidor
central. El servidor sólo cierra el round cuando recibe la ÚLTIMA
actualización, así que la duración del round la fija el cliente más lento::

    round_duration(r) = max_i arrival(i, r)

La duración se desglosa tomando como referencia una red ideal (retardos
cero): con ella el round terminaría cuando acaba de computar el cliente más
lento, y todo lo que excede de ahí es el sobrecoste de la red::

    computing(r)     = max_i computing(i, r)              # round con red ideal
    communication(r) = round_duration(r) - computing(r)   # sobrecoste de red (>= 0)

Así ``computing`` sólo depende de las cargas de trabajo (no de la red) y
``communication`` recoge exactamente el tiempo extra que añade la red
(colas, propagación, retransmisiones).

El tiempo total de entrenamiento y su desglose son la suma sobre rounds::

    total         = sum_r round_duration(r)
    computing     = sum_r computing(r)
    communication = sum_r communication(r)

de modo que ``computing + communication == total``: la barra apilada suma
exactamente el tiempo total.

Cada configuración se repite con varias semillas. Se dibuja la media de cada
componente y, sobre el total, el intervalo de confianza (t de Student).

Uso
---
    python plot_fl_training_time.py

Salidas (en ``figures/fl_training_time/``):
    * training_time_<dataset>.{pdf,png}               -> un panel por experimento
    * training_time_<dataset>_<experimento>.{pdf,png} -> paneles individuales
    * training_time_per_seed.csv                       -> desglose por semilla
    * training_time_summary.csv                        -> medias, IC, std, n

Todo lo que es razonable cambiar está en la sección "CONFIGURACIÓN".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend sin pantalla (servidor / ssh)
import matplotlib.pyplot as plt

try:
    from scipy import stats as _stats

    def _t_critical(confidence: float, dof: int) -> float:
        """Cuantil de la t de Student para un IC bilateral."""
        return float(_stats.t.ppf(0.5 + confidence / 2.0, df=dof))

except ImportError:  # pragma: no cover  (scipy no instalado)
    from statistics import NormalDist

    print("[WARN] scipy no disponible: el IC usa la aproximación normal.")

    def _t_critical(confidence: float, dof: int) -> float:
        return NormalDist().inv_cdf(0.5 + confidence / 2.0)


# =============================================================================
# CONFIGURACIÓN  (rutas, columnas, semillas, nivel de confianza, experimentos)
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Carpeta con las trazas de red (.csv) y carpeta de salida de las figuras.
TRACES_DIR = SCRIPT_DIR / "net"
FIGURES_DIR = SCRIPT_DIR / "figures" / "fl_training_time"

# Columnas de los ficheros de trazas, en orden. Los ficheros pueden venir con
# o sin línea de cabecera: se detecta automáticamente y se asignan estos
# nombres por posición.
COLUMNS = [
    "client-id",
    "round_number",
    "workload",
    "bg-workload",
    "computation-time",
    "client-queue-delay",
    "propagation-delay",
    "station-queue-delay",
]

# Columna de cómputo local y columnas de retardo de red. Su suma es el
# instante en que el modelo local llega al servidor (arrival).
COMPUTING_COLUMN = "computation-time"
NETWORK_COLUMNS = ["client-queue-delay", "propagation-delay", "station-queue-delay"]
ARRIVAL_COLUMNS = [COMPUTING_COLUMN] + NETWORK_COLUMNS

# Componentes que se calculan y guardan en los CSV.
COMPONENTS = ["computing", "communication", "total"]

# Semillas (repeticiones) de cada configuración.
SEEDS = [42, 1337, 2026, 8888, 9999]

# Nivel de confianza del intervalo (0.95 -> IC del 95 %).
CONFIDENCE_LEVEL = 0.95

# Nº máximo de rounds que entran en el tiempo total, por dataset (None = todos).
# Sirve para comparar en igualdad de condiciones configuraciones cuyas trazas
# tienen distinto nº de rounds (p. ej. en Shakespeare las de minibatch llegan
# a 60 rounds y las de FedAvg a 50): se usan sólo los N primeros rounds.
MAX_ROUNDS = {
    "femnist": None,
    "shakespeare": 50,
}

# Extrapolación del nº de rounds. Caso típico: las trazas de FEMNIST tienen
# 277 rounds pero el tiempo total se quiere para 500. Si EXTRAPOLATE_ROUNDS
# está activado y la traza tiene MENOS rounds que "target", se suman sólo los
# "use" primeros rounds y el resultado se multiplica por target/use
# (250 -> 500: x2). Si la traza ya tiene >= target rounds se usan los "target"
# primeros sin multiplicar. Tiene prioridad sobre MAX_ROUNDS en ese dataset.
EXTRAPOLATE_ROUNDS = True
ROUND_EXTRAPOLATION = {
    "femnist": {"use": 250, "target": 500},
    "shakespeare": None,
}

# Datasets a analizar: clave = identificador en el nombre de fichero,
# valor = nombre que aparece en las figuras.
DATASETS = {
    "femnist": "FEMNIST",
    "shakespeare": "Shakespeare",
}

# ---- Parámetros FIJOS de las trazas ----------------------------------------
# Sólo se procesan ficheros con estos valores salvo en el experimento que varía
# precisamente ese parámetro. En particular, las figuras de clientes y de
# minibatch usan SÓLO trazas con probabilidad de transmisión tx = 1.0.
TX_FIXED = "1.0"                              # probabilidad de transmisión
EPOCHS = 1                                    # épocas locales (FedAvg, e_1)
MB_CLIENTS = 20                               # nº de clientes en minibatch
TX_CLIENTS = {"femnist": 20, "shakespeare": 10}  # nº de clientes al variar tx
FP = "500000000"                              # sufijo fp_ de los ficheros


def fedavg_file(clients="{value}", tx=TX_FIXED) -> str:
    """Plantilla de traza FedAvg. Los marcadores {dataset}/{value}/{seed} se
    rellenan al leer; `clients` o `tx` pueden ser "{value}" (el que varía)."""
    return (f"metrics_network_heterogeneous_{{dataset}}_fedavg_c_{clients}_e_{EPOCHS}"
            f"_bg_multi_tx_{tx}_fp_{FP}_seed_{{seed}}.csv")


def minibatch_file(clients=MB_CLIENTS, mb="{value}", tx=TX_FIXED) -> str:
    """Plantilla de traza minibatch (ver fedavg_file)."""
    return (f"metrics_network_heterogeneous_{{dataset}}_minibatch_c_{clients}_mb_{mb}"
            f"_bg_multi_tx_{tx}_fp_{FP}_seed_{{seed}}.csv")


# Probabilidades de transmisión del experimento "tx", en el orden del eje X
# (Frame Loss Rate creciente: 0 %, 5 %, 10 %, 15 %, 20 %).
TX_VALUES = ["1.0", "0.95", "0.9", "0.85", "0.8"]


def frame_loss_pct(tx) -> str:
    """Frame Loss Rate en % a partir de la probabilidad de transmisión:
    100 · (1 - tx), sin decimales ("1.0" -> "0", "0.95" -> "5")."""
    return f"{round((1.0 - float(tx)) * 100):d}"


# Experimentos (cada uno es un panel). Definen:
#   * xlabel   : etiqueta del eje X del panel (cadena común o {dataset: cadena}).
#   * short    : (opcional) nombre corto del parámetro para la tabla y los CSV.
#   * fixed    : parámetros que permanecen fijos (se muestran en la tabla, en
#                los CSV y, si SHOW_FIXED_PARAMS, sobre cada panel). Un valor
#                puede ser un diccionario {dataset: valor} si difiere.
#   * filename : plantilla del nombre de fichero (cadena común o diccionario
#                {dataset: plantilla}).
#   * values   : valores del parámetro que varía, por dataset, en el orden del
#                eje X. Se usan tal cual en el nombre de fichero (por eso "1"
#                y no "1.0" para el minibatch de FEMNIST).
#   * overrides: (opcional) {dataset: {valor: plantilla}} para tomar un valor
#                concreto de OTRO fichero.
#   * ticklabels: (opcional) rótulos del eje X (lista común o {dataset: lista}),
#                en el mismo orden que `values`, cuando lo que se rotula no es
#                el valor del fichero (p. ej. Frame Loss Rate en vez de tx).
EXPERIMENTS = {
    # --- FedAvg (E épocas, tx fijo), variando el nº de clientes -------------
    "clients": {
        "xlabel": "Clients (Mini-batch = 1.0)",
        "short": "Clients",
        "fixed": {"E": EPOCHS, "tx": TX_FIXED},
        "filename": fedavg_file(clients="{value}"),
        "values": {
            "femnist": [3, 5, 10, 20, 30, 50],
            "shakespeare": [2, 3, 4, 5, 8, 10, 20],
        },
    },
    # --- Minibatch (C clientes, tx fijo), variando el tamaño de batch --------
    "batch": {
        # OJO: los ficheros de Shakespeare son minibatch_c_20 (20 clientes);
        # comprueba la columna "clients" de la tabla antes de fijar este texto.
        "xlabel": {
            "femnist": "Mini-batch (Clients = 20)",
            "shakespeare": "Mini-batch (Clients = 10)",
        },
        "short": "Batch",
        "fixed": {"C": MB_CLIENTS, "tx": TX_FIXED},
        "filename": minibatch_file(mb="{value}"),
        "values": {
            "femnist": ["0.2", "0.4", "0.5", "0.6", "0.8", "0.9"],
            "shakespeare": ["0.2", "0.4", "0.5", "0.6", "0.8", "0.9"],
        },
        # Si algún día quieres incluir mb=1 en Shakespeare (no hay traza mb_1),
        # añade "1" a values y descomenta esto para leerlo de FedAvg c=20:
        # "overrides": {"shakespeare": {"1": fedavg_file(clients=MB_CLIENTS)}},
    },
    # --- FedAvg (C clientes, E épocas), variando la probabilidad tx ---------
    # El eje X se rotula como Frame Loss Rate (%) = 100 · (1 - tx), de menor a
    # mayor pérdida. `values` sigue siendo tx (es lo que va en el nombre del
    # fichero, en los CSV y en la tabla); `ticklabels` es lo que se dibuja.
    "tx": {
        "xlabel": "Frame Loss Rate (%)",
        "short": "tx",
        "fixed": {"C": TX_CLIENTS, "E": EPOCHS},
        "filename": {
            ds: fedavg_file(clients=c, tx="{value}") for ds, c in TX_CLIENTS.items()
        },
        "values": {
            "femnist": TX_VALUES,
            "shakespeare": TX_VALUES,
        },
        "ticklabels": [frame_loss_pct(v) for v in TX_VALUES],
    },
}

# Figuras: cada entrada genera una figura POR DATASET con los paneles indicados
# (en ese orden, eje Y compartido). Fichero: training_time_<dataset>_<nombre>.
FIGURES = {
    "clients_batch": {"panels": ["clients", "batch"], "legend_loc": "upper left"},
    "tx":            {"panels": ["tx"],               "legend_loc": "best"},
}

# Comprobaciones de equivalencia: pares de configuraciones (experimento,
# dataset, valor) que DEBERÍAN dar el mismo tiempo total en cada semilla
# (p. ej. minibatch mb=1 con 20 clientes == FedAvg con 20 clientes y 1 época).
# El script imprime ambos totales por semilla y avisa si difieren más de
# EQUIVALENCE_REL_TOL (diferencia relativa).
EQUIVALENCE_CHECKS = [
    (("tx", "shakespeare", "1.0"), ("clients", "shakespeare", "10")),
    (("tx", "femnist", "1.0"), ("clients", "femnist", "20")),
]
EQUIVALENCE_REL_TOL = 1e-6

# ---- Opciones de la figura --------------------------------------------------
CI_ON_TOTAL = True             # barra de error (IC) en el tope de la barra apilada
CI_ON_COMPUTING = False        # barra de error (IC) también en el tramo de cómputo
ANNOTATE_MEANS = False         # escribe el total medio encima de cada barra
MEAN_LABEL_FMT = "{:.0f}"      # formato de esa anotación
SHOW_TITLE = False             # nombre del dataset como título de la figura
SHOW_FIXED_PARAMS = False      # parámetros fijos (p. ej. "C = 20, tx = 1.0") sobre cada panel
Y_HEADROOM = 1.25              # eje Y hasta (máximo + IC) * Y_HEADROOM, deja sitio a la leyenda
SAVE_INDIVIDUAL_PANELS = True  # además de la figura conjunta, una por panel
FIGURE_FORMATS = ["pdf", "png"]  # cada formato en su subcarpeta (figures/.../pdf, .../png)
PANEL_SIZE = (3.5, 3.5)        # tamaño (ancho, alto) de cada panel, en pulgadas
DPI = 200
# Unidad de tiempo del eje Y de las figuras: "s", "min" o "h". Los CSV y la
# tabla por pantalla siguen siempre en segundos.
TIME_UNIT = "min"
TIME_UNITS = {"s": (1.0, "Time (s)"), "min": (60.0, "Time (min)"), "h": (3600.0, "Time (h)")}
YLABEL = TIME_UNITS[TIME_UNIT][1]
BAR_WIDTH = 0.8
COLORS = {"computing": "#FFA500", "communication": "#6EC6E6"}
LABELS = {"computing": "Computing", "communication": "Communication"}
FONT_SIZE = 11


# =============================================================================
# LECTURA DE TRAZAS
# =============================================================================

def _has_header(path: Path) -> bool:
    """Devuelve True si la primera línea del CSV no empieza por un número."""
    with open(path, "r", encoding="utf-8") as fh:
        first_field = fh.readline().split(",")[0].strip()
    try:
        float(first_field)
        return False
    except ValueError:
        return True


def read_trace(path: Path) -> pd.DataFrame:
    """Lee un fichero de trazas y devuelve un DataFrame con las columnas COLUMNS."""
    df = pd.read_csv(path, header=0 if _has_header(path) else None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: se esperaban {len(COLUMNS)} columnas y hay {df.shape[1]}"
        )
    df.columns = COLUMNS                       # nombres por posición
    return df.apply(pd.to_numeric).reset_index(drop=True)


# =============================================================================
# CÁLCULO DEL TIEMPO DE CADA ROUND Y DEL TIEMPO TOTAL
# =============================================================================

def round_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duración de cada round de FL, desglosada en cómputo y comunicación.

    Paso 1 - Para cada fila (cliente, round) se calcula el instante en que su
             modelo local llega al servidor central:
                 arrival = computation-time
                         + client-queue-delay + propagation-delay + station-queue-delay
    Paso 2 - El round termina cuando llega el ÚLTIMO cliente:
                 total = max_i arrival
             Se desglosa respecto a una red ideal (retardos cero), con la que
             el round terminaría al acabar de computar el cliente más lento:
                 computing     = max_i computation-time     (round con red ideal)
                 communication = total - computing          (sobrecoste de red, >= 0)
             Así, computing + communication == duración del round, computing
             no depende de la red y communication es el tiempo extra que
             añade la red (colas, propagación, retransmisiones).

    Devuelve un DataFrame indexado por round_number (ordenado) con las
    columnas computing, communication y total.
    """
    arrival = df[ARRIVAL_COLUMNS].sum(axis=1)                        # Paso 1
    grp = df.assign(arrival=arrival).groupby("round_number")        # Paso 2
    out = pd.DataFrame({
        "total": grp["arrival"].max(),
        "computing": grp[COMPUTING_COLUMN].max(),
    }).sort_index()
    out["communication"] = out["total"] - out["computing"]
    return out[["computing", "communication", "total"]]


def total_training_time(df: pd.DataFrame, max_rounds: int | None = None,
                        extrapolation: dict | None = None) -> dict:
    """
    Paso 3 - Tiempo total de entrenamiento = suma de la duración de los rounds
             (y lo mismo para cada componente).

    Selección de rounds (en orden de round_number):
      * `extrapolation` = {"use": U, "target": T} (si no es None):
          - traza con >= T rounds -> se suman los T primeros, factor 1.
          - traza con  < T rounds -> se suman los U primeros y se multiplica
            por T/U para estimar el tiempo de T rounds (p. ej. 250 -> 500: x2).
            Si hay menos de U rounds se avisa y se usa lo que hay.
      * si no, `max_rounds` (si no es None) -> se suman los max_rounds primeros.
      * si no, todos los rounds.

    Devuelve un diccionario con computing, communication, total (ya escalados),
    n_rounds (rounds sumados), n_rounds_file (rounds en la traza), scale
    (factor aplicado) y n_rounds_equiv (rounds que representa el total).
    """
    rb = round_breakdown(df)
    n_rounds_file = int(rb.shape[0])
    scale = 1.0

    if extrapolation is not None:
        use, target = int(extrapolation["use"]), int(extrapolation["target"])
        if n_rounds_file >= target:
            rb = rb.iloc[:target]
        else:
            if n_rounds_file < use:
                print(f"[WARN] la traza tiene {n_rounds_file} rounds (< use = {use}): "
                      f"se extrapola desde {n_rounds_file} rounds")
            rb = rb.iloc[:min(use, n_rounds_file)]
            scale = target / rb.shape[0]
    elif max_rounds is not None:
        rb = rb.iloc[:max_rounds]

    n_rounds = int(rb.shape[0])
    return {
        "n_clients": int(df["client-id"].nunique()),   # clientes distintos en la traza
        "computing": float(rb["computing"].sum()) * scale,
        "communication": float(rb["communication"].sum()) * scale,
        "total": float(rb["total"].sum()) * scale,
        "n_rounds": n_rounds,
        "n_rounds_file": n_rounds_file,
        "scale": scale,
        "n_rounds_equiv": int(round(n_rounds * scale)),
    }


# =============================================================================
# ESTADÍSTICA: MEDIA E INTERVALO DE CONFIANZA ENTRE SEMILLAS
# =============================================================================

def mean_and_ci(values, confidence: float = CONFIDENCE_LEVEL) -> tuple[float, float, float]:
    """
    Media, semi-anchura del IC y desviación típica muestral de `values`.

    IC = media ± t_{(1+conf)/2, n-1} · s / sqrt(n)   (t de Student).
    Con una sola muestra el IC no está definido (se devuelve NaN).
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    std = float(x.std(ddof=1))
    half_width = _t_critical(confidence, n - 1) * std / np.sqrt(n)
    return mean, float(half_width), std


# =============================================================================
# RECOLECCIÓN DE RESULTADOS
# =============================================================================

def _per_dataset(spec, dataset: str):
    """Permite que un campo de la configuración sea común o por dataset."""
    return spec[dataset] if isinstance(spec, dict) else spec


def fixed_label(exp_name: str, dataset: str) -> str:
    """Texto con los parámetros fijos del experimento, p. ej. "C = 20, tx = 1.0"."""
    fixed = EXPERIMENTS[exp_name].get("fixed", {})
    return ", ".join(f"{k} = {_per_dataset(v, dataset)}" for k, v in fixed.items())


def collect_per_seed(exp_name: str) -> pd.DataFrame:
    """
    Calcula el tiempo total (y su desglose) de cada (dataset, valor, semilla)
    del experimento. Los ficheros que falten se avisan y se omiten.
    """
    exp = EXPERIMENTS[exp_name]
    rows = []
    for dataset in DATASETS:
        default_template = _per_dataset(exp["filename"], dataset)
        overrides = exp.get("overrides", {}).get(dataset, {})
        max_rounds = MAX_ROUNDS.get(dataset)
        extrapolation = ROUND_EXTRAPOLATION.get(dataset) if EXTRAPOLATE_ROUNDS else None
        for value in _per_dataset(exp["values"], dataset):
            # Plantilla específica para este valor (si la hay) o la general
            template = overrides.get(str(value), default_template)
            for seed in SEEDS:
                path = TRACES_DIR / template.format(dataset=dataset, value=value, seed=seed)
                if not path.is_file():
                    print(f"[WARN] no encontrado, se omite: {path.name}")
                    continue
                res = total_training_time(read_trace(path), max_rounds, extrapolation)
                if extrapolation is None and max_rounds is not None \
                        and res["n_rounds_file"] < max_rounds:
                    print(f"[WARN] {path.name}: sólo tiene {res['n_rounds_file']} rounds "
                          f"(< MAX_ROUNDS = {max_rounds})")
                rows.append({
                    "experiment": exp_name,
                    "dataset": dataset,
                    "value": str(value),
                    "fixed": fixed_label(exp_name, dataset),
                    "seed": seed,
                    **res,
                    "file": path.name,
                })
    return pd.DataFrame(rows)


def check_equivalences(per_seed: pd.DataFrame) -> None:
    """
    Verificación del cálculo: para cada par de EQUIVALENCE_CHECKS imprime el
    tiempo total de ambas configuraciones semilla a semilla y la diferencia.
    Si las trazas de entrada son iguales, los totales deben coincidir
    exactamente; si difieren, la discrepancia está en las trazas.
    """
    if not EQUIVALENCE_CHECKS:
        return
    print("\n=== Comprobación de equivalencia (tiempo total por semilla) ===")
    idx = per_seed.set_index(["experiment", "dataset", "value", "seed"])
    for cfg_a, cfg_b in EQUIVALENCE_CHECKS:
        name_a = "/".join(cfg_a)
        name_b = "/".join(cfg_b)
        print(f"  {name_a}  vs  {name_b}")
        all_ok, any_found = True, False
        for seed in SEEDS:
            key_a, key_b = (*cfg_a, seed), (*cfg_b, seed)
            if key_a not in idx.index or key_b not in idx.index:
                missing = name_a if key_a not in idx.index else name_b
                print(f"    seed {seed:>5}: falta {missing}, no se compara")
                continue
            any_found = True
            ta, tb = idx.loc[key_a, "total"], idx.loc[key_b, "total"]
            rel = abs(ta - tb) / max(abs(ta), abs(tb), 1e-12)
            ok = rel <= EQUIVALENCE_REL_TOL
            all_ok &= ok
            print(f"    seed {seed:>5}: {ta:12.4f}  vs {tb:12.4f}"
                  f"   diff = {ta - tb:+.3e}  ({'OK' if ok else 'DIFIEREN'})")
            if not ok:
                print(f"      ficheros: {idx.loc[key_a, 'file']}  |  {idx.loc[key_b, 'file']}")
        if any_found:
            print("    -> " + ("coinciden en todas las semillas."
                              if all_ok else
                              "NO coinciden: revisa que las trazas sean las mismas "
                              "(mismos clientes, rounds y workload)."))


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las semillas de cada (experimento, dataset, valor): para cada
    componente, media, semi-anchura del IC y std; además n_seeds y n_rounds.
    """
    records = []
    for (exp, dataset, value), grp in per_seed.groupby(
        ["experiment", "dataset", "value"], sort=False
    ):
        rec = {
            "experiment": exp,
            "dataset": dataset,
            "value": value,
            "fixed": fixed_label(exp, dataset),
            "n_seeds": len(grp),
            "n_clients": int(grp["n_clients"].iloc[0]),
            "n_rounds": int(grp["n_rounds"].iloc[0]),
            "n_rounds_file": int(grp["n_rounds_file"].iloc[0]),
            "scale": float(grp["scale"].iloc[0]),
            "n_rounds_equiv": int(grp["n_rounds_equiv"].iloc[0]),
        }
        for comp in COMPONENTS:
            mean, half, std = mean_and_ci(grp[comp])
            rec[f"{comp}_mean"] = mean
            rec[f"{comp}_ci"] = half
            rec[f"{comp}_std"] = std
        records.append(rec)
    return pd.DataFrame(records)


# =============================================================================
# FIGURAS
# =============================================================================

def _draw_panel(ax, exp_name: str, dataset: str, summary: pd.DataFrame) -> None:
    """Dibuja en `ax` las barras apiladas (cómputo + comunicación) con el IC."""
    exp = EXPERIMENTS[exp_name]
    values = [str(v) for v in _per_dataset(exp["values"], dataset)]
    sub = summary[(summary.experiment == exp_name) & (summary.dataset == dataset)]
    sub = sub.set_index("value")

    def col(name: str) -> np.ndarray:
        # valores en el orden configurado; NaN si falta la configuración
        return np.array([sub[name].get(v, np.nan) for v in values], dtype=float)

    # Conversión de segundos a la unidad del eje Y (sólo para dibujar)
    unit = TIME_UNITS[TIME_UNIT][0]
    computing = col("computing_mean") / unit
    communication = col("communication_mean") / unit
    total = computing + communication
    total_ci = col("total_ci") / unit
    computing_ci = col("computing_ci") / unit
    x = np.arange(len(values))

    # Barras apiladas: cómputo abajo, comunicación encima
    ax.bar(x, computing, width=BAR_WIDTH, color=COLORS["computing"],
           label=LABELS["computing"], zorder=2)
    ax.bar(x, communication, bottom=computing, width=BAR_WIDTH,
           color=COLORS["communication"], label=LABELS["communication"], zorder=2)

    # Intervalos de confianza (media ± semi-anchura, calculados sobre las semillas)
    err_kw = dict(fmt="none", ecolor="black", elinewidth=1.2,
                  capsize=4, capthick=1.2, zorder=5)
    if CI_ON_TOTAL:
        ax.errorbar(x, total, yerr=total_ci, **err_kw)
    if CI_ON_COMPUTING:
        ax.errorbar(x, computing, yerr=computing_ci, **err_kw)

    if ANNOTATE_MEANS:
        for xi, m, c in zip(x, total, total_ci):
            if np.isnan(m):
                continue
            top = m + (0.0 if np.isnan(c) else c)
            ax.annotate(MEAN_LABEL_FMT.format(m), (xi, top), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 2)

    # Rótulos del eje X: `ticklabels` si el experimento los define, si no `values`
    ticklabels = exp.get("ticklabels")
    if ticklabels is None:
        ticklabels = values
    else:
        ticklabels = [str(t) for t in _per_dataset(ticklabels, dataset)]
    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels)
    ax.set_xlabel(_per_dataset(exp["xlabel"], dataset))
    if SHOW_FIXED_PARAMS:
        ax.set_title(fixed_label(exp_name, dataset), fontsize=FONT_SIZE - 1)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Margen superior para que la leyenda no pise las barras (con sharey el
    # límite final es el mayor de todos los paneles).
    top = np.nanmax(total + np.nan_to_num(total_ci)) * Y_HEADROOM
    ax.set_ylim(0, max(top, ax.get_ylim()[1]))


def _save(fig, stem: str) -> None:
    """Guarda la figura en una subcarpeta por formato: figures/.../png, .../pdf"""
    for fmt in FIGURE_FORMATS:
        out_dir = FIGURES_DIR / fmt
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_figure(fig_name: str, dataset: str, summary: pd.DataFrame) -> None:
    """Figura `fig_name` del dataset: un panel por experimento, eje Y compartido."""
    spec = FIGURES[fig_name]
    legend_loc = spec.get("legend_loc", "best")
    exps = [e for e in spec["panels"]
            if ((summary.experiment == e) & (summary.dataset == dataset)).any()]
    if not exps:
        print(f"[WARN] figura '{fig_name}' de {DATASETS[dataset]}: sin datos, no se dibuja.")
        return

    # Figura conjunta
    fig, axes = plt.subplots(1, len(exps), sharey=True, squeeze=False,
                             figsize=(PANEL_SIZE[0] * len(exps), PANEL_SIZE[1]))
    for ax, exp_name in zip(axes[0], exps):
        _draw_panel(ax, exp_name, dataset, summary)
    axes[0][0].set_ylabel(YLABEL)
    axes[0][0].legend(loc=legend_loc)
    if SHOW_TITLE:
        fig.suptitle(DATASETS[dataset], fontsize=FONT_SIZE + 2)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    _save(fig, f"training_time_{dataset}_{fig_name}")

    # Paneles individuales (sólo si la figura tiene más de uno)
    if SAVE_INDIVIDUAL_PANELS and len(exps) > 1:
        for exp_name in exps:
            fig, ax = plt.subplots(figsize=PANEL_SIZE)
            _draw_panel(ax, exp_name, dataset, summary)
            ax.set_ylabel(YLABEL)
            ax.legend(loc=legend_loc)
            if SHOW_TITLE:
                title = DATASETS[dataset]
                if SHOW_FIXED_PARAMS:
                    title += f"  ({fixed_label(exp_name, dataset)})"
                ax.set_title(title, fontsize=FONT_SIZE + 1)
            fig.tight_layout()
            _save(fig, f"training_time_{dataset}_{exp_name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    plt.rcParams.update({"font.size": FONT_SIZE})
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not TRACES_DIR.is_dir():
        print(f"[ERROR] carpeta de trazas no encontrada: {TRACES_DIR}")
        return 1

    # Experimentos necesarios = paneles de todas las figuras (sin repetir)
    experiments_to_run = list(dict.fromkeys(
        e for spec in FIGURES.values() for e in spec["panels"]))

    # 1) Tiempo total (y desglose) por semilla, para todos los experimentos
    frames = []
    for exp_name in experiments_to_run:
        print(f"\n=== Experimento '{exp_name}' ===")
        df = collect_per_seed(exp_name)
        if df.empty:
            print("[WARN] ningún fichero leído para este experimento.")
        else:
            frames.append(df)
    if not frames:
        print("[ERROR] no se ha leído ningún fichero.")
        return 1
    per_seed = pd.concat(frames, ignore_index=True)

    # Aviso si dentro de un dataset se están sumando distinto nº de rounds
    # según la configuración (la comparación no sería justa).
    for dataset, grp in per_seed.groupby("dataset"):
        counts = sorted(int(c) for c in grp["n_rounds_equiv"].unique())
        if len(counts) > 1:
            print(f"[WARN] {DATASETS[dataset]}: los totales representan distinto nº de "
                  f"rounds {counts}; revisa MAX_ROUNDS / ROUND_EXTRAPOLATION['{dataset}'].")

    # 2) Media e IC entre semillas
    summary = summarize(per_seed)

    # 3) Figuras: cada entrada de FIGURES, para cada dataset
    for fig_name in FIGURES:
        for dataset in DATASETS:
            plot_figure(fig_name, dataset, summary)

    # 4) Resultados a disco y resumen por pantalla
    per_seed.to_csv(FIGURES_DIR / "training_time_per_seed.csv", index=False)
    summary.to_csv(FIGURES_DIR / "training_time_summary.csv", index=False)

    # 5) Verificación: configuraciones que deben dar el mismo total
    check_equivalences(per_seed)

    ci_pct = int(round(CONFIDENCE_LEVEL * 100))
    print(f"\nResumen (media ± IC {ci_pct} %, {len(SEEDS)} semillas; "
          f"rounds = sumados/en fichero [xfactor -> rounds equivalentes]):")
    for _, r in summary.iterrows():
        exp = EXPERIMENTS[r.experiment]
        varying = f"{exp.get('short', exp['xlabel'])} = {r.value}"
        rounds = f"{r.n_rounds}/{r.n_rounds_file}"
        if r.scale != 1.0:
            rounds += f" [x{r.scale:.2f} -> {r.n_rounds_equiv}]"
        print(f"  {DATASETS[r.dataset]:<12} {varying:<16} ({r.fixed:<16})  "
              f"clients = {r.n_clients:<3} rounds = {rounds:<22} n = {r.n_seeds}"
              f"  computing = {r.computing_mean:9.2f}"
              f"  communication = {r.communication_mean:9.2f}"
              f"  total = {r.total_mean:9.2f} ± {r.total_ci:.2f} s")
    print(f"\nFiguras y CSV guardados en: {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())