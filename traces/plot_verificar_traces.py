from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYS_DIR = Path("sys")
INDIVIDUAL_DIR = Path("figures/individual_cdfs")
GROUPED_DIR = Path("figures/grouped_cdfs")

SAVE_PDF = False

INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
GROUPED_DIR.mkdir(parents=True, exist_ok=True)


def read_cdf(csv_file):
    flops = pd.read_csv(csv_file, header=None, usecols=[7]).iloc[:, 0]
    flops = pd.to_numeric(flops, errors="coerce").dropna().to_numpy()
    flops = np.sort(flops / 1e9)

    cdf = np.arange(1, len(flops) + 1) / len(flops) * 100
    return flops, cdf


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


for csv_file in sorted(SYS_DIR.glob("sys_metrics_*.csv")):
    flops, cdf = read_cdf(csv_file)

    if len(flops) == 0:
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

    groups.setdefault(group_key, []).append((label, flops, cdf))

    # Individual figure.
    plt.figure(figsize=(6, 4))
    plt.step(flops, cdf, where="post")
    plt.xlabel("Computational demand (GFLOPs)")
    plt.ylabel("Clients per training round (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_figure(INDIVIDUAL_DIR / f"cdf_{csv_file.stem}")


# Grouped figures.
for group_key, curves in groups.items():
    dataset = group_key[0]
    algorithm = group_key[1]

    plt.figure(figsize=(6, 4))

    for label, flops, cdf in curves:
        plt.step(flops, cdf, where="post", label=label)

    plt.xlabel("Computational demand (GFLOPs)")
    plt.ylabel("Clients per training round (%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if algorithm == "fedavg":
        output_name = f"cdf_{dataset}_fedavg"
    else:
        clients = group_key[2]
        output_name = f"cdf_{dataset}_minibatch_c_{clients}"

    save_figure(GROUPED_DIR / output_name)


print(f"Individual figures: {INDIVIDUAL_DIR}")
print(f"Grouped figures:    {GROUPED_DIR}")