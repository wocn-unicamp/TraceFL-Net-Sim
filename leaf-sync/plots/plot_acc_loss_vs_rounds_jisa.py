import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ================================ Config =====================================

folder_path = "../results/stat/"

file_bases_1 = [
    "stat_metrics_fedavg_c_50_e_1",
    "stat_metrics_fedavg_c_30_e_1",
    "stat_metrics_fedavg_c_20_e_1",
    "stat_metrics_fedavg_c_10_e_1",
    "stat_metrics_fedavg_c_5_e_1",
    "stat_metrics_fedavg_c_3_e_1",
]

file_bases_2 = [
    # "stat_metrics_minibatch_c_20_mb_1",
    "stat_metrics_minibatch_c_20_mb_0.9",
    "stat_metrics_minibatch_c_20_mb_0.8",
    "stat_metrics_minibatch_c_20_mb_0.6",
    "stat_metrics_minibatch_c_20_mb_0.5",
    "stat_metrics_minibatch_c_20_mb_0.4",
    "stat_metrics_minibatch_c_20_mb_0.2",
]

file_bases_3 = [
    # "stat_metrics_shakespeare_fedavg_c_2_e_1",
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

# Per-group subplot limits
limits_by_group = {
    "fedavg_clients": {
        "acc_test":  {"x": (50, 1000), "y": (10, 80)},
        "loss_train":{"x": (50, 1000), "y": (0.5, 4.0)},
    },
    "minibatch_p": {
        "acc_test":  {"x": (50, 1000), "y": (10, 80)},
        "loss_train":{"x": (50, 1000), "y": (0.5, 4.0)},
    },
    "shakespeare_fedavg": {
        "acc_test":  {"x": (0, 50), "y": (15, 55)},
        "loss_train":{"x": (0, 50), "y": (1.5, 4.0)},
    },
    "shakespeare_minibatch": {
        "acc_test":  {"x": (0, 50), "y": (15, 55)},
        "loss_train":{"x": (0, 50), "y": (1.5, 4.0)},
    },
}

# Major tick steps per group/subplot
ticks_by_group = {
    "fedavg_clients": {
        "acc_test":  {"x_major": 100, "y_major": 10},
        "loss_train":{"x_major": 100, "y_major": 0.5}
    },
    "minibatch_p": {
        "acc_test":  {"x_major": 100, "y_major": 10},
        "loss_train":{"x_major": 100, "y_major": 0.5}
    },
    "shakespeare_fedavg": {
        "acc_test":  {"x_major": 5, "y_major": 5},
        "loss_train":{"x_major": 5, "y_major": 0.5}
    },
    "shakespeare_minibatch": {
        "acc_test":  {"x_major": 5, "y_major": 5},
        "loss_train":{"x_major": 5, "y_major": 0.5}
    },
}

# Typography
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
    Prefer minibatch label when both mb_* and c_* exist.
    - '...mb_0.5'  -> 'mb=0.5'
    - else if '...c_10' -> '10 Clients'
    - else fallback to short file tag
    """
    base = os.path.basename(name)
    m_mb = re.search(r"mb_([0-9.]+)", base)
    if m_mb:
        val = m_mb.group(1)
        if "." in val:
            val = val.rstrip("0").rstrip(".")
            val = int(float(val)*100)
        return f"Minibatch = {val}%"
    m_c = re.search(r"c_(\d+)", base)
    if m_c:
        return f"{int(m_c.group(1))} Clients"
    return short_label(base)

def maybe_to_percent(s: pd.Series) -> pd.Series:
    if s.max() is not None and s.max() <= 1.001:
        return s * 100.0
    return s

def unique_legend(handles, labels):
    """Keep first occurrence of each label, preserving order."""
    seen = set()
    h_out, l_out = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h_out.append(h)
            l_out.append(l)
    return h_out, l_out

expected_cols = {"set", "round_number", "accuracy", "loss"}

# ================================ Plot =======================================

os.makedirs("figures", exist_ok=True)

for group_name, file_bases in file_groups:
    if not file_bases:
        continue

    # [0]=Training Loss (LEFT) | [1]=Test Accuracy (RIGHT)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

    for base in file_bases:
        file_path = os.path.join(folder_path, ensure_csv(base))
        if not os.path.exists(file_path):
            print(f"[warn] file not found: {file_path} — skipping.")
            continue

        df = pd.read_csv(file_path)
        missing = expected_cols - set(df.columns)
        if missing:
            print(f"[warn] missing columns {missing} in {file_path} — skipping.")
            continue

        # Types
        df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df = df.dropna(subset=["round_number"])

        df_train = df[df["set"] == "train"]
        df_test  = df[df["set"] == "test"]

        # Per-round means
        acc_test   = df_test.groupby("round_number")["accuracy"].mean().sort_index()
        loss_train = df_train.groupby("round_number")["loss"].mean().sort_index()

        label = smart_label(file_path)

        # Training Loss (LEFT)
        if not loss_train.empty:
            axs[0].plot(loss_train.index, loss_train.values, linewidth=linewidth, label=label)
        else:
            print(f"[warn] no train/loss data in {file_path}.")

        # Test Accuracy (RIGHT) in %
        if not acc_test.empty:
            acc_plot = maybe_to_percent(acc_test)
            axs[1].plot(acc_plot.index, acc_plot.values, linewidth=linewidth, label=label)
        else:
            print(f"[warn] no test/accuracy data in {file_path}.")

    # Titles & labels
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Rounds")
    axs[0].set_ylabel("Loss")

    axs[1].set_title("Test Accuracy")
    axs[1].set_xlabel("Rounds")
    axs[1].set_ylabel("Accuracy (%)")

    # Limits
    cfg_lim = limits_by_group.get(group_name, {})
    acc_lim  = cfg_lim.get("acc_test", {})
    loss_lim = cfg_lim.get("loss_train", {})
    if "x" in loss_lim:  axs[0].set_xlim(*loss_lim["x"])
    if "y" in loss_lim:  axs[0].set_ylim(*loss_lim["y"])
    if "x" in acc_lim:   axs[1].set_xlim(*acc_lim["x"])
    if "y" in acc_lim:   axs[1].set_ylim(*acc_lim["y"])

    # Major ticks
    cfg_ticks = ticks_by_group.get(group_name, {})
    acc_ticks  = cfg_ticks.get("acc_test", {})
    loss_ticks = cfg_ticks.get("loss_train", {})
    if "x_major" in loss_ticks:
        axs[0].xaxis.set_major_locator(MultipleLocator(loss_ticks["x_major"]))
    if "y_major" in loss_ticks:
        axs[0].yaxis.set_major_locator(MultipleLocator(loss_ticks["y_major"]))
    if "x_major" in acc_ticks:
        axs[1].xaxis.set_major_locator(MultipleLocator(acc_ticks["x_major"]))
    if "y_major" in acc_ticks:
        axs[1].yaxis.set_major_locator(MultipleLocator(acc_ticks["y_major"]))

    # Grid
    for ax in axs.ravel():
        ax.grid(True, alpha=grid_alpha)

    # Common legend (top, 2 rows × 3 columns)
    h0, l0 = axs[0].get_legend_handles_labels()
    h1, l1 = axs[1].get_legend_handles_labels()
    handles, labels = unique_legend(h0 + h1, l0 + l1)

    for ax in axs:
        ax.legend_.remove() if ax.get_legend() else None

    leg = fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,             # 3 columns → with 6 lines you get 2 rows
        frameon=True,
        bbox_to_anchor=(0.5, 1.05),
    )
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#cccccc")

    # Space for legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    base_tag = re.sub(r"\.csv$", "", os.path.basename(ensure_csv(file_bases[0])))
    out_path = os.path.join("figures", f"plot_trainLoss_testAcc_{group_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # plt.show()
    print(f"[ok] figure saved at: {out_path}")
