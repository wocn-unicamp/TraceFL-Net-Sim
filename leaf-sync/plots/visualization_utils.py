"""Helper to visualize metrics (robust to missing baseline_constants, headers and column name variants)."""
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal

# =============================================================================
# 1) Importar baseline_constants; si no existe, usar stub
# =============================================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from baseline_constants import (
        ACCURACY_KEY,
        BYTES_READ_KEY,
        BYTES_WRITTEN_KEY,
        CLIENT_ID_KEY,
        LOCAL_COMPUTATIONS_KEY,
        NUM_ROUND_KEY,
        NUM_SAMPLES_KEY,
    )
except Exception:
    # ---- STUB por ausencia de baseline_constants.py ----
    ACCURACY_KEY           = "accuracy"
    BYTES_READ_KEY         = "bytes_read"
    BYTES_WRITTEN_KEY      = "bytes_written"
    CLIENT_ID_KEY          = "client_id"
    LOCAL_COMPUTATIONS_KEY = "local_computations"
    NUM_ROUND_KEY          = "round_number"
    NUM_SAMPLES_KEY        = "num_samples"

# =============================================================================
# 2) Auto-normalización de nombres
# =============================================================================

# Candidatos (subcadenas aceptadas, en minúsculas) -> nombre destino
CANDIDATES = {
    ACCURACY_KEY:           ["accuracy", "acc"],
    BYTES_READ_KEY:         ["bytes_read", "read_bytes", "bytesreceived", "rx_bytes"],
    BYTES_WRITTEN_KEY:      ["bytes_written", "write_bytes", "bytessent", "tx_bytes"],
    CLIENT_ID_KEY:          ["client_id", "client", "user_id", "cid"],
    LOCAL_COMPUTATIONS_KEY: ["local_computations", "flops", "local_flops", "compute"],
    NUM_ROUND_KEY:          ["round_number", "round", "num_round", "global_round", "ronda"],
    NUM_SAMPLES_KEY:        ["num_samples", "samples", "test_samples", "n_samples"],
}

# Requisitos mínimos (separados por uso para evitar falsos warnings)
STAT_REQUIRED = [NUM_ROUND_KEY, CLIENT_ID_KEY, NUM_SAMPLES_KEY, ACCURACY_KEY]
SYS_REQUIRED_BYTES_MIN = [NUM_ROUND_KEY, CLIENT_ID_KEY, BYTES_WRITTEN_KEY]
SYS_REQUIRED_FLOPS     = [NUM_ROUND_KEY, CLIENT_ID_KEY, LOCAL_COMPUTATIONS_KEY]

def _find_col(cols_lower, needles):
    for c in cols_lower:
        for n in needles:
            if n in c:
                return c
    return None

def _auto_rename(df: pd.DataFrame, required_keys):
    """Renombra columnas por coincidencia de subcadenas para que coincidan con required_keys."""
    if df is None: 
        return df
    cols = list(df.columns)
    cols_lower = {c: str(c).lower() for c in cols}
    inv = {v: k for k, v in cols_lower.items()}  # lower->real

    rename_map = {}
    for target in required_keys:
        if target in df.columns:
            continue
        needles = CANDIDATES.get(target, [target])
        hit = _find_col(list(cols_lower.values()), needles)
        if hit is not None:
            real = inv[hit]
            rename_map[real] = target

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ajuste adicional común: 'round' -> 'round_number' si sigue faltando
    if NUM_ROUND_KEY in required_keys and NUM_ROUND_KEY not in df.columns and 'round' in df.columns:
        df = df.rename(columns={'round': NUM_ROUND_KEY})

    return df

def _coerce_numeric(df, cols):
    """Convierte columnas a numéricas (coerce) si existen."""
    if df is None: 
        return df
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =============================================================================
# 3) Carga de datos con corrección de headers faltantes
# =============================================================================
def load_data(stat_metrics_file='stat_metrics.csv', sys_metrics_file='sys_metrics.csv'):
    """
    Lee los CSV, detecta si los encabezados son inválidos (cuando la primera fila son datos),
    vuelve a leer con header=None y asigna headers esperados; luego normaliza nombres y dtypes.
    """
    def _looks_like_bad_header(cols):
        # Si muchos encabezados parecen datos (números, 'train', 'test', 'Unnamed'), se considera malo
        bad_tokens = {"train", "test"}
        bad = 0
        for c in cols:
            s = str(c)
            if s.lower().startswith("unnamed"):
                bad += 1;  continue
            try:
                float(s)
                bad += 1;  continue
            except:
                pass
            if s.lower() in bad_tokens:
                bad += 1
        return bad >= max(2, len(cols) // 3)

    def _assign_expected_names(df, kind):
        """
        Asigna encabezados “base”. Para sys_metrics hay caso especial: si el archivo tiene 7 columnas,
        se asume que la última es FLOPs (local_computations) y NO hay bytes_read.
        """
        if df is None:
            return df
        n = df.shape[1]
        if kind == 'stat':
            base7 = ['client_id','round_number','hierarchy','num_samples','set','accuracy','loss']
            if n <= 7:
                df.columns = base7[:n]
            else:
                df.columns = base7 + [f"c{i}" for i in range(7, n)]
            return df

        # kind == 'sys'
        base8 = ['client_id','round_number','hierarchy','num_samples','set','bytes_written','bytes_read','local_computations']
        base7_flops_last = ['client_id','round_number','hierarchy','num_samples','set','bytes_written','local_computations']
        if n == 7:
            df.columns = base7_flops_last
        elif n <= 8:
            df.columns = base8[:n]
        else:
            df.columns = base8 + [f"c{i}" for i in range(8, n)]
        return df

    def _read_with_header_fix(path, kind):
        if not path or not os.path.isfile(path):
            return None
        df = pd.read_csv(path)
        if _looks_like_bad_header(list(df.columns)):
            df = pd.read_csv(path, header=None)
            empty_cols = [c for c in df.columns if df[c].isna().all()]
            if empty_cols:
                df = df.drop(columns=empty_cols)
            df = _assign_expected_names(df, kind)
        return df

    # 1) Leer con fix de header
    stat_metrics = _read_with_header_fix(stat_metrics_file, kind='stat') if stat_metrics_file else None
    sys_metrics  = _read_with_header_fix(sys_metrics_file,  kind='sys')  if sys_metrics_file  else None

    # 2) Normalizar nombres esperados
    if stat_metrics is not None:
        stat_metrics = _auto_rename(stat_metrics, STAT_REQUIRED)
        # Tipos
        stat_metrics = _coerce_numeric(stat_metrics, [NUM_ROUND_KEY, NUM_SAMPLES_KEY, ACCURACY_KEY, "loss"])
        if NUM_ROUND_KEY in stat_metrics.columns:
            stat_metrics = stat_metrics.sort_values(by=NUM_ROUND_KEY).reset_index(drop=True)

    if sys_metrics is not None:
        # Renombrar para bytes y FLOPs (cada uno con su set mínimo)
        sys_metrics = _auto_rename(sys_metrics, SYS_REQUIRED_BYTES_MIN)
        sys_metrics = _auto_rename(sys_metrics, SYS_REQUIRED_FLOPS)

        # === FIX adicional: si aún falta local_computations, tomar la ÚLTIMA columna como FLOPs ===
        if LOCAL_COMPUTATIONS_KEY not in sys_metrics.columns:
            if len(sys_metrics.columns) > 0:
                last_col = sys_metrics.columns[-1]
                if last_col != NUM_ROUND_KEY:  # evita pisar la ronda por accidente
                    sys_metrics = sys_metrics.rename(columns={last_col: LOCAL_COMPUTATIONS_KEY})
                else:
                    print("[WARN] sys_metrics: última columna es round_number; no reasigno a FLOPs.")

        # Tipos
        sys_metrics = _coerce_numeric(sys_metrics, [NUM_ROUND_KEY, BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY])
        if NUM_ROUND_KEY in sys_metrics.columns:
            sys_metrics = sys_metrics.sort_values(by=NUM_ROUND_KEY).reset_index(drop=True)

    # 3) Avisos suaves (separados)
    def _warn_missing(df, name, required):
        if df is None:
            return
        miss = [k for k in required if k not in df.columns]
        if miss:
            print(f"[WARN] {name}: faltan columnas {miss}. Presentes: {list(df.columns)}")

    _warn_missing(stat_metrics, "stat_metrics", STAT_REQUIRED)
    _warn_missing(sys_metrics,  "sys_metrics(bytes)", SYS_REQUIRED_BYTES_MIN)
    _warn_missing(sys_metrics,  "sys_metrics(flops)", SYS_REQUIRED_FLOPS)

    return stat_metrics, sys_metrics

# =============================================================================
# 4) Helpers de plot
# =============================================================================
def _set_plot_properties(properties):
    if 'xlim' in properties: plt.xlim(properties['xlim'])
    if 'ylim' in properties: plt.ylim(properties['ylim'])
    if 'xlabel' in properties: plt.xlabel(properties['xlabel'])
    if 'ylabel' in properties: plt.ylabel(properties['ylabel'])

def _weighted_mean(df, metric_name, weight_name):
    d = df[metric_name]; w = df[weight_name]
    try: return (w * d).sum() / w.sum()
    except ZeroDivisionError: return np.nan

def _weighted_std(df, metric_name, weight_name):
    d = df[metric_name]; w = df[weight_name]
    try:
        m = (w * d).sum() / w.sum()
        var = (w * ((d - m) ** 2)).sum() / w.sum()
        return np.sqrt(var)
    except ZeroDivisionError:
        return np.nan

# =============================================================================
# 5) Plots
# =============================================================================
def plot_accuracy_vs_round_number(stat_metrics, weighted=False, plot_stds=False,
        figsize=(10, 8), title_fontsize=16, **kwargs):
    """Media de accuracy vs ronda (con/bajo ponderación por #samples)."""
    if stat_metrics is None or NUM_ROUND_KEY not in stat_metrics or ACCURACY_KEY not in stat_metrics:
        print("[WARN] No se puede trazar Accuracy vs Round: faltan columnas necesarias.")
        return

    plt.figure(figsize=figsize)
    title_weighted = 'Weighted' if weighted else 'Unweighted'
    plt.title('Accuracy vs Round Number (%s)' % title_weighted, fontsize=title_fontsize)

    if weighted and NUM_SAMPLES_KEY in stat_metrics.columns:
        accuracies = stat_metrics.groupby(NUM_ROUND_KEY).apply(
            _weighted_mean, ACCURACY_KEY, NUM_SAMPLES_KEY).reset_index(name=ACCURACY_KEY)
        stds = stat_metrics.groupby(NUM_ROUND_KEY).apply(
            _weighted_std, ACCURACY_KEY, NUM_SAMPLES_KEY).reset_index(name=ACCURACY_KEY)
    else:
        accuracies = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False)[ACCURACY_KEY].mean()
        stds       = stat_metrics.groupby(NUM_ROUND_KEY, as_index=False)[ACCURACY_KEY].std()

    if plot_stds:
        plt.errorbar(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY], stds[ACCURACY_KEY], fmt='-o', linewidth=1)
    else:
        plt.plot(accuracies[NUM_ROUND_KEY], accuracies[ACCURACY_KEY], linewidth=2)

    p10 = stat_metrics.groupby(NUM_ROUND_KEY)[ACCURACY_KEY].quantile(0.10).reset_index()
    p90 = stat_metrics.groupby(NUM_ROUND_KEY)[ACCURACY_KEY].quantile(0.90).reset_index()
    plt.plot(p10[NUM_ROUND_KEY], p10[ACCURACY_KEY], linestyle=':', linewidth=1)
    plt.plot(p90[NUM_ROUND_KEY], p90[ACCURACY_KEY], linestyle=':', linewidth=1)

    plt.legend(['Mean', '10th percentile', '90th percentile'], loc='upper left')
    plt.ylabel('Accuracy'); plt.xlabel('Round Number')
    _set_plot_properties(kwargs); plt.tight_layout(); plt.show()


def plot_accuracy_vs_round_number_per_client(
        stat_metrics, sys_metrics, max_num_clients, figsize=(15, 12), title_fontsize=16, max_name_len=10, **kwargs):
    """Accuracy por cliente vs ronda (puntos cuando entrenó)."""
    if stat_metrics is None or CLIENT_ID_KEY not in stat_metrics or NUM_ROUND_KEY not in stat_metrics or ACCURACY_KEY not in stat_metrics:
        print("[WARN] No se puede trazar per-client: faltan columnas necesarias en stat_metrics.")
        return

    clients = stat_metrics[CLIENT_ID_KEY].dropna().unique()[:max_num_clients]
    cmap = plt.get_cmap('jet_r')
    plt.figure(figsize=figsize)

    for i, c in enumerate(clients):
        color = cmap(float(i) / max(1, len(clients)))
        c_accuracies = stat_metrics.loc[stat_metrics[CLIENT_ID_KEY] == c]
        plt.plot(c_accuracies[NUM_ROUND_KEY], c_accuracies[ACCURACY_KEY], color=color)

    plt.suptitle('Accuracy vs Round Number (Per Client)', fontsize=title_fontsize)
    plt.title('(Dots indicate that client was trained at that round)')
    plt.xlabel('Round Number'); plt.ylabel('Accuracy')

    if NUM_SAMPLES_KEY in stat_metrics.columns:
        labels = stat_metrics[[CLIENT_ID_KEY, NUM_SAMPLES_KEY]].drop_duplicates()
        labels = labels.loc[labels[CLIENT_ID_KEY].isin(clients)]
        labels = [f"{str(row[CLIENT_ID_KEY])[:max_name_len]}, {int(row[NUM_SAMPLES_KEY])}" for _, row in labels.iterrows()]
        if len(labels) == len(clients):
            plt.legend(labels, title='client id, num_samples', loc='upper left')

    # Dots donde el cliente realmente entrenó (si hay sys_metrics)
    if sys_metrics is not None and CLIENT_ID_KEY in sys_metrics and NUM_ROUND_KEY in sys_metrics:
        for i, c in enumerate(clients):
            color = cmap(float(i) / max(1, len(clients)))
            c_accuracies = stat_metrics.loc[stat_metrics[CLIENT_ID_KEY] == c, [NUM_ROUND_KEY, ACCURACY_KEY]]
            c_comp       = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c, [NUM_ROUND_KEY]].drop_duplicates()
            c_join = pd.merge(c_accuracies, c_comp, on=NUM_ROUND_KEY, how='inner')
            if not c_join.empty:
                plt.plot(c_join[NUM_ROUND_KEY], c_join[ACCURACY_KEY],
                         linestyle='None', marker='.', color=color, markersize=12)

    _set_plot_properties(kwargs); plt.tight_layout(); plt.show()


def plot_bytes_written_and_read(sys_metrics, rolling_window=10, figsize=(10, 8), title_fontsize=16, **kwargs):
    """Rolling sum de bytes escritos/leídos por ronda (soporta que 'bytes_read' no exista)."""
    if sys_metrics is None or NUM_ROUND_KEY not in sys_metrics or BYTES_WRITTEN_KEY not in sys_metrics:
        print("[WARN] No se puede trazar bytes: faltan columnas mínimas en sys_metrics.")
        return

    has_read = BYTES_READ_KEY in sys_metrics.columns

    # Solo las columnas necesarias y numéricas
    cols = [NUM_ROUND_KEY, BYTES_WRITTEN_KEY] + ([BYTES_READ_KEY] if has_read else [])
    df = sys_metrics[cols].copy()
    df = _coerce_numeric(df, cols)
    df = df.sort_values(NUM_ROUND_KEY)

    # Sumar por ronda
    agg_map = {BYTES_WRITTEN_KEY: 'sum'}
    if has_read:
        agg_map[BYTES_READ_KEY] = 'sum'
    server_metrics = df.groupby(NUM_ROUND_KEY, as_index=False).agg(agg_map).sort_values(NUM_ROUND_KEY)

    # Rolling sobre series
    roll = int(rolling_window) if rolling_window and rolling_window > 1 else 1
    server_metrics["bytes_written_roll"] = server_metrics[BYTES_WRITTEN_KEY].rolling(roll, min_periods=1).sum()
    if has_read:
        server_metrics["bytes_read_roll"] = server_metrics[BYTES_READ_KEY].rolling(roll, min_periods=1).sum()

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(server_metrics[NUM_ROUND_KEY], server_metrics["bytes_written_roll"], alpha=0.9, linewidth=2, label='Bytes Written (rolling)')
    if has_read:
        plt.plot(server_metrics[NUM_ROUND_KEY], server_metrics["bytes_read_roll"],    alpha=0.9, linewidth=2, label='Bytes Read (rolling)')

    title_suffix = "" if has_read else " (read missing)"
    plt.title('Bytes by Server vs. Round Number' + title_suffix, fontsize=title_fontsize)
    plt.xlabel('Round Number'); plt.ylabel('Bytes')
    plt.legend(loc='upper left')
    _set_plot_properties(kwargs)
    plt.tight_layout(); plt.show()


def _round_axis(sys_metrics):
    """Devuelve (rounds_sorted, index_map) para manejar rondas no consecutivas o que empiezan en 1."""
    rounds_sorted = sorted(pd.unique(sys_metrics[NUM_ROUND_KEY]))
    idx = {r: i for i, r in enumerate(rounds_sorted)}
    return rounds_sorted, idx


def plot_client_computations_vs_round_number(
        sys_metrics,
        aggregate_window=20,
        max_num_clients=20,
        figsize=(25, 15),
        title_fontsize=16,
        max_name_len=10,
        range_rounds=None):
    """FLOPs locales agregados por ventanas de rounds. (Si falta local_computations, se omite.)"""
    if sys_metrics is None or CLIENT_ID_KEY not in sys_metrics or \
       NUM_ROUND_KEY not in sys_metrics or LOCAL_COMPUTATIONS_KEY not in sys_metrics:
        print("[WARN] No se puede trazar FLOPs: faltan columnas necesarias en sys_metrics.")
        return

    plt.figure(figsize=figsize)

    # Asegurar numeric
    sys_metrics = _coerce_numeric(sys_metrics, [NUM_ROUND_KEY, LOCAL_COMPUTATIONS_KEY])

    rounds_sorted, r_index = _round_axis(sys_metrics)
    num_rounds = len(rounds_sorted)
    clients = sys_metrics[CLIENT_ID_KEY].dropna().unique()[:max_num_clients]

    comp_matrix = []
    matrix_keys = [str(c)[:max_name_len] for c in clients]

    for c in clients:
        client_rows = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
        client_rows = client_rows.groupby(NUM_ROUND_KEY, as_index=False).sum(numeric_only=True)
        c_comp = [0] * num_rounds
        for _, row in client_rows[[NUM_ROUND_KEY, LOCAL_COMPUTATIONS_KEY]].iterrows():
            c_comp[r_index[row[NUM_ROUND_KEY]]] = row[LOCAL_COMPUTATIONS_KEY]
        comp_matrix.append(c_comp)

    if range_rounds:
        a, b = range_rounds
        assert 0 <= a < b <= num_rounds and (b - a) >= aggregate_window
        comp_matrix = [row[a:b] for row in comp_matrix]
        num_rounds = b - a

    agg_comp_matrix = []
    for row in comp_matrix:
        agg = []
        for i in range(max(1, num_rounds // aggregate_window)):
            start = i * aggregate_window
            end   = min((i + 1) * aggregate_window, num_rounds)
            agg.append(int(np.sum(row[start:end])))
        agg_comp_matrix.append(agg)

    plt.title(f'Total Client Computations (FLOPs) vs. Round Number (x {aggregate_window})',
              fontsize=title_fontsize)
    im = plt.imshow(agg_comp_matrix, aspect='auto')
    plt.yticks(range(len(matrix_keys)), matrix_keys)
    plt.colorbar(im, fraction=0.02, pad=0.01)
    plt.tight_layout(); plt.show()


def get_longest_flops_path(sys_metrics):
    """Mayor suma de FLOPs tomando el máximo por ronda (si falta, retorna N/A)."""
    if sys_metrics is None or NUM_ROUND_KEY not in sys_metrics or LOCAL_COMPUTATIONS_KEY not in sys_metrics:
        return "N/A"

    sys_metrics = _coerce_numeric(sys_metrics, [NUM_ROUND_KEY, LOCAL_COMPUTATIONS_KEY])

    rounds_sorted, r_index = _round_axis(sys_metrics)
    num_rounds = len(rounds_sorted)
    clients = sys_metrics[CLIENT_ID_KEY].dropna().unique()

    comp_matrix = []
    for c in clients:
        client_rows = sys_metrics.loc[sys_metrics[CLIENT_ID_KEY] == c]
        client_rows = client_rows.groupby(NUM_ROUND_KEY, as_index=False).sum(numeric_only=True)
        c_comp = [0] * num_rounds
        for _, row in client_rows[[NUM_ROUND_KEY, LOCAL_COMPUTATIONS_KEY]].iterrows():
            c_comp[r_index[row[NUM_ROUND_KEY]]] = row[LOCAL_COMPUTATIONS_KEY]
        comp_matrix.append(c_comp)

    comp_matrix = np.asarray(comp_matrix) if comp_matrix else np.zeros((1, num_rounds))
    num_flops = np.sum(np.max(comp_matrix, axis=0)) if comp_matrix.size else 0
    return '%.2E' % Decimal(float(num_flops))
