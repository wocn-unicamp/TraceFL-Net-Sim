import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ================================ Config =====================================

folder_path = "../results_backup/stat/"

file_bases_1 = [
    "stat_metrics_fedavg_c_50_e_1",
    "stat_metrics_fedavg_c_30_e_1",
    "stat_metrics_fedavg_c_20_e_1",
    "stat_metrics_fedavg_c_10_e_1",
    "stat_metrics_fedavg_c_5_e_1",
    "stat_metrics_fedavg_c_3_e_1",
]

file_bases_2 = [
    "stat_metrics_minibatch_c_20_mb_0.9",
    "stat_metrics_minibatch_c_20_mb_0.8",
    "stat_metrics_minibatch_c_20_mb_0.6",
    "stat_metrics_minibatch_c_20_mb_0.5",
    "stat_metrics_minibatch_c_20_mb_0.4",
    "stat_metrics_minibatch_c_20_mb_0.2",
]

file_bases_3 = [
    "stat_metrics_shakespeare_fedavg_c_3_e_1",
    "stat_metrics_shakespeare_fedavg_c_4_e_1",
    "stat_metrics_shakespeare_fedavg_c_5_e_1",
    "stat_metrics_shakespeare_fedavg_c_8_e_1",
    "stat_metrics_shakespeare_fedavg_c_10_e_1",
    "stat_metrics_shakespeare_fedavg_c_20_e_1"
]

file_bases_4 = [
    "stat_metrics_shakespeare_minibatch_c_10_mb_0.9",
    "stat_metrics_shakespeare_minibatch_c_10_mb_0.8",
    "stat_metrics_shakespeare_minibatch_c_10_mb_0.6",
    "stat_metrics_shakespeare_minibatch_c_10_mb_0.5",
    "stat_metrics_shakespeare_minibatch_c_10_mb_0.4",
    "stat_metrics_shakespeare_minibatch_c_10_mb_0.2"
]

file_groups = [
    ("fedavg_clients", file_bases_1),
    ("minibatch_p", file_bases_2),
    ("shakespeare_fedavg", file_bases_3),
    ("shakespeare_minibatch", file_bases_4),
]

# Límites por grupo (solo accuracy)
limits_by_group = {
    "fedavg_clients":      {"acc_test": {"x": (50, 1000), "y": (10, 80)}},
    "minibatch_p":         {"acc_test": {"x": (50, 1000), "y": (10, 80)}},
    "shakespeare_fedavg":  {"acc_test": {"x": (0, 50),    "y": (15, 55)}},
    "shakespeare_minibatch":{"acc_test":{"x": (0, 50),    "y": (15, 55)}},
}

# Ticks mayores por grupo (solo accuracy)
ticks_by_group = {
    "fedavg_clients":      {"acc_test": {"x_major": 100, "y_major": 10}},
    "minibatch_p":         {"acc_test": {"x_major": 100, "y_major": 10}},
    "shakespeare_fedavg":  {"acc_test": {"x_major": 5,   "y_major": 5}},
    "shakespeare_minibatch":{"acc_test":{"x_major": 5,   "y_major": 5}},
}

# Tipografía
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

linewidth = 2.2
grid_alpha = 0.3

# ================================ Utils ======================================

def ensure_csv(path_base: str) -> str:
    return path_base if path_base.endswith(".csv") else path_base + ".csv"

def short_label(name: str) -> str:
    base = os.path.basename(name).replace(".csv", "")
    base = re.sub(r"^stat_metrics_(shakespeare_)?(minibatch_|fedavg_)?", "", base)
    return base

def smart_label(name: str) -> str:
    """
    Prefiere etiqueta de minibatch cuando exista.
    - '...mb_0.5'  -> 'Minibatch = 50%'
    - si no, '...c_10' -> '10 Clients'
    """
    base = os.path.basename(name)
    m_mb = re.search(r"mb_([0-9.]+)", base)
    if m_mb:
        val = m_mb.group(1)
        if "." in val:
            val = val.rstrip("0").rstrip(".")
            val = int(float(val) * 100)
        return f"Minibatch = {val/100}"
    m_c = re.search(r"c_(\d+)", base)
    if m_c:
        return f"{int(m_c.group(1))} Clients"
    return short_label(base)

def maybe_to_percent(s: pd.Series) -> pd.Series:
    if s.max() is not None and s.max() <= 1.001:
        return s * 100.0
    return s

def unique_legend(handles, labels):
    """Mantiene la primera ocurrencia de cada etiqueta, preservando el orden."""
    seen = set()
    h_out, l_out = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h_out.append(h)
            l_out.append(l)
    return h_out, l_out

# Solo columnas necesarias para accuracy
expected_cols_req = {"set", "round_number", "accuracy"}

# ================================ Plot =======================================

out_dir = "figures/accuracy_vs_rounds"
os.makedirs(out_dir, exist_ok=True)

for group_name, file_bases in file_groups:
    if not file_bases:
        continue

    # ÚNICA figura: Test Accuracy
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), sharex=False)

    for base in file_bases:
        file_path = os.path.join(folder_path, ensure_csv(base))
        if not os.path.exists(file_path):
            print(f"[warn] file not found: {file_path} — skipping.")
            continue

        df = pd.read_csv(file_path)
        missing = expected_cols_req - set(df.columns)
        if missing:
            print(f"[warn] missing columns {missing} in {file_path} — skipping.")
            continue

        # Tipos
        df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df = df.dropna(subset=["round_number"])

        df_test = df[df["set"] == "test"]
        acc_test = df_test.groupby("round_number")["accuracy"].mean().sort_index()

        label = smart_label(file_path)

        if not acc_test.empty:
            acc_plot = maybe_to_percent(acc_test)
            ax.plot(acc_plot.index, acc_plot.values, linewidth=linewidth, label=label)
        else:
            print(f"[warn] no test/accuracy data in {file_path}.")

    # Título y ejes
    ax.set_title("Test Accuracy")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy (%)")

    # Límites
    acc_lim = limits_by_group.get(group_name, {}).get("acc_test", {})
    if "x" in acc_lim: ax.set_xlim(*acc_lim["x"])
    if "y" in acc_lim: ax.set_ylim(*acc_lim["y"])

    # Ticks mayores
    acc_ticks = ticks_by_group.get(group_name, {}).get("acc_test", {})
    if "x_major" in acc_ticks:
        ax.xaxis.set_major_locator(MultipleLocator(acc_ticks["x_major"]))
    if "y_major" in acc_ticks:
        ax.yaxis.set_major_locator(MultipleLocator(acc_ticks["y_major"]))

    # Grid
    ax.grid(True, alpha=grid_alpha)

    # Leyenda
    h, l = ax.get_legend_handles_labels()
    handles, labels = unique_legend(h, l)
    if handles:
        leg = ax.legend(handles, labels, loc="upper center", ncol=3,
                        frameon=True, bbox_to_anchor=(0.5, 1.3))
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(out_dir, f"plot_testAcc_{group_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # plt.show()
    print(f"[ok] figure saved at: {out_path}")
