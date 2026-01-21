import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ================================ Config =====================================

folder_path = "../results_2026/stat/"

file_bases_1 = [
    "stat_metrics_fedavg_c_64_e_1",
    "stat_metrics_fedavg_c_64_e_2",
    "stat_metrics_fedavg_c_64_e_3",
]

file_groups = [
    ("fedavg_clients", file_bases_1),
]

limits_by_group = {
    "fedavg_clients": {
        "acc_test":  {"x": (0, 1000), "y": (10, 80)},
        "loss_train":{"x": (0, 1000), "y": (0.5, 4.0)},
    },
}

ticks_by_group = {
    "fedavg_clients": {
        "acc_test":  {"x_major": 100, "y_major": 10},
        "loss_train":{"x_major": 100, "y_major": 0.5}
    },
}

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

def maybe_to_percent(s: pd.Series) -> pd.Series:
    # If accuracy is in [0,1], convert to %
    mx = s.max()
    if pd.notna(mx) and mx <= 1.001:
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

def label_fedavg_c_fixed_e_var(name: str, c_fixed: int = 64) -> str:
    """
    Your filenames: stat_metrics_fedavg_c_64_e_1, etc.
    Since c is fixed and e varies, legend should be: 'E = 1', 'E = 2', ...
    """
    base = os.path.basename(name)

    m_e = re.search(r"e_(\d+)", base)
    e_val = m_e.group(1) if m_e else None

    # Optional sanity check for c
    m_c = re.search(r"c_(\d+)", base)
    c_val = int(m_c.group(1)) if m_c else None

    if e_val is not None:
        # If you want to also show c in legend, use: f"C={c_val}, E={e_val}"
        return f"E = {int(e_val)}"

    # Fallback label
    if c_val is not None:
        return f"C = {c_val}"
    return base.replace(".csv", "")

expected_cols = {"set", "round_number", "accuracy", "loss"}

# ================================ Plot =======================================

out_dir = os.path.join("figures", "acc_loss_vs_rounds")
os.makedirs(out_dir, exist_ok=True)

for group_name, file_bases in file_groups:
    if not file_bases:
        continue

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

        df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df = df.dropna(subset=["round_number"])

        df_train = df[df["set"] == "train"]
        df_test  = df[df["set"] == "test"]

        acc_test   = df_test.groupby("round_number")["accuracy"].mean().sort_index()
        loss_train = df_train.groupby("round_number")["loss"].mean().sort_index()

        label = label_fedavg_c_fixed_e_var(file_path, c_fixed=64)

        if not loss_train.empty:
            axs[0].plot(loss_train.index, loss_train.values, linewidth=linewidth, label=label)
        else:
            print(f"[warn] no train/loss data in {file_path}.")

        if not acc_test.empty:
            acc_plot = maybe_to_percent(acc_test)
            axs[1].plot(acc_plot.index, acc_plot.values, linewidth=linewidth, label=label)
        else:
            print(f"[warn] no test/accuracy data in {file_path}.")

    # Titles & labels (include that C is fixed)
    axs[0].set_title("Training Loss (C = 64)")
    axs[0].set_xlabel("Rounds")
    axs[0].set_ylabel("Loss")

    axs[1].set_title("Test Accuracy (C = 64)")
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

    # Common legend (top)
    h0, l0 = axs[0].get_legend_handles_labels()
    h1, l1 = axs[1].get_legend_handles_labels()
    handles, labels = unique_legend(h0 + h1, l0 + l1)

    for ax in axs:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    leg = fig.legend(
        handles, labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.05),
        title="Local epochs (E)"
    )
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = os.path.join(out_dir, f"plot_trainLoss_testAcc_{group_name}_c64_varE.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[ok] figure saved at: {out_path}")
