#!/usr/bin/env python3
"""
CDFs agrupadas del tiempo de computo de los traces unidos (net_join/).

FedAvg    -> una figura por dataset, una curva por numero de clientes.
Minibatch -> una figura por dataset y numero de clientes, una curva por mb.

Uso:  python plot_cdf_computation_time.py
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
NET_JOIN = AQUI / "net_join"
GROUPED_DIR = AQUI / "figures" / "net_grouped_cdfs"
SAVE_PDF = False

GROUPED_DIR.mkdir(parents=True, exist_ok=True)


def read_cdf(csv_file):
    times = pd.read_csv(csv_file, usecols=["computation-time"])["computation-time"]
    times = pd.to_numeric(times, errors="coerce").dropna().to_numpy()
    times = np.sort(times)
    cdf = np.arange(1, len(times) + 1) / len(times) * 100
    return times, cdf


def file_info(csv_file):
    name = csv_file.stem.lower()
    dataset = "femnist" if "femnist" in name else "shakespeare"
    algorithm = "fedavg" if "fedavg" in name else "minibatch"
    clients = int(re.search(r"_c_(\d+)", name).group(1))
    mb_match = re.search(r"_mb_([\d.]+)", name)
    minibatch = float(mb_match.group(1)) if mb_match else None
    return dataset, algorithm, clients, minibatch


def save_figure(output):
    plt.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        plt.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


# 1) Agrupar las curvas.
groups = {}
for csv_file in sorted(NET_JOIN.glob("metrics_network_*.csv")):
    times, cdf = read_cdf(csv_file)
    if len(times) == 0:
        continue

    dataset, algorithm, clients, minibatch = file_info(csv_file)

    if algorithm == "fedavg":
        group_key = (dataset, algorithm)
        orden = clients                 # para ordenar la leyenda
        label = f"c={clients}"
    else:
        group_key = (dataset, algorithm, clients)
        orden = minibatch
        label = f"mb={minibatch:g}"

    groups.setdefault(group_key, []).append((orden, label, times, cdf))

# 2) Dibujar una figura por grupo.
for group_key, curves in groups.items():
    dataset = group_key[0]
    algorithm = group_key[1]

    plt.figure(figsize=(6, 4))
    for orden, label, times, cdf in sorted(curves, key=lambda x: x[0]):
        plt.step(times, cdf, where="post", label=label)
    plt.xlabel("Computation time (s)")
    plt.ylabel("Clients per training round (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if algorithm == "fedavg":
        output_name = f"cdf_time_{dataset}_fedavg"
    else:
        clients = group_key[2]
        output_name = f"cdf_time_{dataset}_minibatch_c_{clients}"
    save_figure(GROUPED_DIR / output_name)
    print(f"{output_name}.png  ({len(curves)} curvas)")

print(f"\nGrouped figures: {GROUPED_DIR}")