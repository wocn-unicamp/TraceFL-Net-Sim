#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

# ===================== Config =====================
PATH_FOLDER = "../results/sys/"
NAME_FILE   = "sys_metrics_fedavg_c_50_e_1"
EXT_FILE    = ".csv"

BINS, SMOOTH_WINDOW = 256, 7
OUT_FIG_MODEL   = "figures/mix_lognorm_shift_fit.png"
OUT_FIG_COMPARE = "figures/mix_lognorm_shift_fit_and_sim.png"
OUT_TXT         = "figures/mix_lognorm_shift_params.txt"

RANDOM_STATE = 42
N_SIM_MIN    = 10000  # minimum simulated samples for comparison

# ===================== Utils ======================
def ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True); return path

def load_data(sys_metrics_file):
    if not os.path.isfile(sys_metrics_file):
        raise FileNotFoundError(f"File not found: {sys_metrics_file}")
    df = pd.read_csv(sys_metrics_file)
    df.columns = ["client_id","round_number","idk","samples","set",
                  "bytes_read","bytes_written","FLOPs"]
    df.index.name = "index"
    return df

def convert_FLOPs_to_time(df):
    df = df.copy()
    df["time"] = (df["FLOPs"] / 1e9) * 1e3  # ms
    return df

def ecdf_xy(vals):
    x = np.sort(vals); n = x.size
    y = np.arange(1, n+1, dtype=float) / n
    return x, y

def smoothed_cdf_from_hist(vals, bins=256, smooth_window=7):
    counts, edges = np.histogram(vals, bins=bins, density=True)
    widths = np.diff(edges)
    if smooth_window and smooth_window > 1:
        k = int(smooth_window); kernel = np.ones(k)/k
        counts = np.convolve(counts, kernel, mode="same")
    cdf = np.cumsum(counts * widths)
    if cdf[-1] > 0: cdf /= cdf[-1]
    x = edges[1:]
    return x, cdf

# ======= Mixture of shifted lognormals (EM + grid over loc) =======
def mixture_cdf_lognorm_shift(x, w, mu, sigma, loc):
    x = np.asarray(x, dtype=float)
    F = np.zeros_like(x, dtype=float)
    mask = x > loc
    z = np.log(np.clip(x[mask] - loc, 1e-12, None))
    for wi, m, s in zip(w, mu, sigma):
        F[mask] += wi * stats.norm.cdf((z - m) / s)
    return np.clip(F, 0.0, 1.0)

def fit_mix_lognorm_shift_to_smoothed(v, k_range=(1,2,3), loc_grid=None,
                                      bins=BINS, smooth_window=SMOOTH_WINDOW,
                                      random_state=RANDOM_STATE, n_init=5):
    x_s, y_s = smoothed_cdf_from_hist(v, bins=bins, smooth_window=smooth_window)

    vmin = float(np.min(v))
    if loc_grid is None:
        # search a shift (loc) below the minimum (moves support)
        lo = vmin - 800.0; hi = vmin - 100.0
        loc_grid = np.linspace(lo, hi, 16)

    best = None
    records = []

    for loc in loc_grid:
        u = v - loc
        if np.any(u <= 0):  # invalid loc
            continue
        ylog = np.log(u).reshape(-1,1)

        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                  random_state=random_state, n_init=n_init)
            gmm.fit(ylog)
            w = gmm.weights_.ravel()
            mu = gmm.means_.ravel()
            sigma = np.sqrt(gmm.covariances_.ravel())

            yhat = mixture_cdf_lognorm_shift(x_s, w, mu, sigma, loc)
            sse  = float(np.sum((yhat - y_s)**2))
            bic  = float(gmm.bic(ylog))

            rec = dict(model=f"mixlognorm(k={k})", loc=loc, sse=sse, bic=bic,
                       w=w, mu=mu, sigma=sigma)
            records.append(rec)
            if (best is None) or (sse < best["sse"]):
                best = rec

    summary = pd.DataFrame(records).sort_values(["sse","bic"]).reset_index(drop=True)
    return (x_s, y_s), summary, best

# ============== Sampling from the fitted mixture =================
def sample_shifted_lognorm_mixture(n, loc, w, mu, sigma, random_state=None):
    """
    Draw n samples X = loc + exp(Z_J), J~Categorical(w), Z_J~N(mu_J, sigma_J^2)
    """
    rng = np.random.default_rng(random_state)
    w   = np.asarray(w, float); w = w / w.sum()
    mu  = np.asarray(mu, float)
    sig = np.asarray(sigma, float)
    k   = len(w)

    comp = rng.choice(k, size=n, p=w)
    z    = rng.normal(loc=mu[comp], scale=sig[comp])
    x    = loc + np.exp(z)
    return x

# ===================== Plots ======================
def plot_model(v, x_sm, y_sm, best, out_path=OUT_FIG_MODEL):
    ensure_dir(os.path.dirname(out_path) or "figures")
    x_ecdf, y_ecdf = ecdf_xy(v)
    xfit = np.linspace(v.min(), v.max(), 1200)
    yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])

    plt.figure(figsize=(11,5))
    plt.step(x_ecdf, y_ecdf, where="post", lw=1.8, label="Empirical ECDF")
    plt.plot(x_sm, y_sm, lw=2.0, label=f"Smoothed CDF (bins={BINS}, win={SMOOTH_WINDOW})")
    plt.plot(xfit, yfit, "--", lw=2.2, label=f"{best['model']}, loc={best['loc']:.1f}")
    plt.ylim(0,1)
    plt.xlabel("Time (ms)")
    plt.ylabel("Cumulative Probability")
    plt.title("Smoothed CDF Fit — Shifted Lognormal Mixture")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

def plot_fit_and_simulation(
    v, x_sm, y_sm, best, n_sim,
    out_path="figures/mix_lognorm_shift_fit_and_sim.png",
    clip_to_real_xlim=True,
    truncate_and_renormalize=False,
):
    ensure_dir(os.path.dirname(out_path) or "figures")

    # ECDF (real)
    x_ecdf, y_ecdf = ecdf_xy(v)
    xmin, xmax = float(x_ecdf.min()), float(x_ecdf.max())

    # Simulated
    sim = sample_shifted_lognorm_mixture(
        n_sim, best["loc"], best["w"], best["mu"], best["sigma"], random_state=RANDOM_STATE
    )

    if truncate_and_renormalize:
        # Conditioned ECDF on [xmin, xmax] (truncated view)
        sim_vis = sim[(sim >= xmin) & (sim <= xmax)]
        x_ecdf_sim, y_ecdf_sim = ecdf_xy(sim_vis)
    else:
        x_ecdf_sim, y_ecdf_sim = ecdf_xy(sim)

    # Model CDF for reference on the left panel
    xfit = np.linspace(xmin, xmax if clip_to_real_xlim else max(xmax, sim.max()), 1500)
    yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])

    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2, sharey=True)

    # Left: ECDF + Smoothed + Model
    ax1.step(x_ecdf, y_ecdf, where="post", lw=1.8, label="Empirical ECDF")
    ax1.plot(x_sm, y_sm, lw=2.0, label=f"Smoothed CDF (bins={BINS}, win={SMOOTH_WINDOW})")
    ax1.plot(xfit, yfit, "--", lw=2.2, label=f"{best['model']}")
    ax1.set_title("Fit on real dataset")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Cumulative Probability")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")
    if clip_to_real_xlim:
        ax1.set_xlim(xmin, xmax)

    # Right: Real ECDF vs Simulated ECDF
    ax2.step(x_ecdf, y_ecdf, where="post", lw=1.8, label="Empirical ECDF")
    ax2.step(x_ecdf_sim, y_ecdf_sim, where="post", lw=1.8,
             label=("Simulated ECDF (truncated)" if truncate_and_renormalize
                    else f"Simulated ECDF (n={n_sim})"))
    ax2.set_title("Validation with simulated data")
    ax2.set_xlabel("Time (ms)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    if clip_to_real_xlim:
        ax2.set_xlim(xmin, xmax)
        ax2.axvline(xmax, ls=":", lw=1.2, color="k", alpha=0.4)

    # >>> Make Y ticks visible on both panels at 0,0.2,...,1 <<<
    yticks = np.linspace(0, 1, 6)
    for ax in (ax1, ax2):
        ax.set_ylim(0, 1)
        ax.set_yticks(yticks)
    # labels left only on ax1, right only on ax2
    ax1.tick_params(axis="y", labelleft=True, labelright=False)
    ax2.tick_params(axis="y", labelleft=False, labelright=True)

    fig.suptitle("Mixture (shifted lognormal): fit and validation", fontsize=14)
    plt.savefig(out_path, dpi=150)
    plt.close()

# ===================== Main ======================
def main():
    sys_metrics_file = os.path.join(PATH_FOLDER, NAME_FILE + EXT_FILE)
    ensure_dir("figures")

    df = load_data(sys_metrics_file)
    df = convert_FLOPs_to_time(df)
    v = df["time"].to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]

    (x_sm, y_sm), summary, best = fit_mix_lognorm_shift_to_smoothed(
        v, k_range=(1,2,3), loc_grid=None,
        bins=BINS, smooth_window=SMOOTH_WINDOW,
        random_state=RANDOM_STATE, n_init=5
    )

    # Report
    print("\n=== Best models (sorted by SSE) ===")
    print(summary.head(10).to_string(index=False))

    # Human-readable parameters
    with open(OUT_TXT, "w") as f:
        f.write(f"Best: {best['model']}  loc={best['loc']}\n")
        for i, (wi, mi, si) in enumerate(zip(best["w"], best["mu"], best["sigma"]), 1):
            scale_i = np.exp(mi)
            f.write(f" comp{i}: w={wi:.6f}, mu_log={mi:.6f}, sigma={si:.6f}, scale=exp(mu)={scale_i:.6f}\n")
        f.write(f"SSE={best['sse']:.6f}, BIC={best['bic']:.2f}\n")
    print(f"\nParameters saved to: {OUT_TXT}")

    # Model figure
    plot_model(v, x_sm, y_sm, best, out_path=OUT_FIG_MODEL)
    print(f"Figure saved to:      {OUT_FIG_MODEL}")

    # --- Simulation-based validation ---
    n_sim = max(N_SIM_MIN, 5 * len(v))
    n_sim = 1000
    plot_fit_and_simulation(v, x_sm, y_sm, best, n_sim=n_sim,
                            out_path=OUT_FIG_COMPARE,
                            clip_to_real_xlim=True,
                            truncate_and_renormalize=False)
    print(f"Comparison figure saved to: {OUT_FIG_COMPARE}")

    # Quick stats
    sim = sample_shifted_lognorm_mixture(n_sim, best["loc"], best["w"], best["mu"], best["sigma"],
                                         random_state=RANDOM_STATE)
    print("\n=== Summary statistics ===")
    print(f"Real:     mean={np.mean(v):.1f} ms, median={np.median(v):.1f} ms, std={np.std(v, ddof=1):.1f} ms")
    print(f"Simulated: mean={np.mean(sim):.1f} ms, median={np.median(sim):.1f} ms, std={np.std(sim, ddof=1):.1f} ms")

if __name__ == "__main__":
    main()
