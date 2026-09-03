from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BIMODAL_DIR = Path("sys_bimodal")
GROUPED_DIR = Path("figures/bimodal_cdfs")

SAVE_PDF = False

GROUPED_DIR.mkdir(parents=True, exist_ok=True)


def read_cdf(csv_file):
    # columna 9 = time (segundos). Las 8 primeras son las originales,
    # la 8 es capacity_gflops.
    times = pd.read_csv(csv_file, header=None, usecols=[9]).iloc[:, 0]
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
    minibatch = mb_match.group(1) if mb_match else None

    return dataset, algorithm, clients, minibatch


def save_figure(output):
    plt.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")

    if SAVE_PDF:
        plt.savefig(output.with_suffix(".pdf"), bbox_inches="tight")

    plt.close()


groups = {}


for csv_file in sorted(BIMODAL_DIR.glob("sys_metrics_*.csv")):
    times, cdf = read_cdf(csv_file)

    if len(times) == 0:
        continue

    dataset, algorithm, clients, minibatch = file_info(csv_file)

    # FedAvg: one figure per dataset, varying c.
    if algorithm == "fedavg":
        group_key = (dataset, algorithm)
        label = f"c={clients}"

    # Minibatch: one figure per dataset and number of clients.
    else:
        group_key = (dataset, algorithm, clients)
        label = f"mb={minibatch}"

    groups.setdefault(group_key, []).append((label, times, cdf))


# Grouped figures.
for group_key, curves in groups.items():
    dataset = group_key[0]
    algorithm = group_key[1]

    plt.figure(figsize=(6, 4))

    for label, times, cdf in curves:
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


print(f"Grouped figures:    {GROUPED_DIR}")