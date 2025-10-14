#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

# ===================== Config =====================
SYS_FILE         = "../results/sys/sys_metrics_fedavg_c_50_e_1.csv"
# SYS_FILE       = "../results/sys/sys_metrics_minibatch_c_20_mb_0.6.csv"
OUT_FIG_HIST     = "figures/fit/hist_smoothed.png"
OUT_FIG_ECDF     = "figures/fit/ecdf_validation.png"
OUT_FIG_MIXED    = "figures/fit/mix_hist_ecdf.png"
OUT_TXT          = "params/mix_lognorm_shift_params.txt"

RANDOM_STATE = 42
N_SIM_MIN    = 1000
FLOPS_PER_SEC = 1e9
MS2FLOPS = FLOPS_PER_SEC/1e3
X_LABEL = "FLOPs"
HIST_BINS = 64           # same bins in both plots to look comparable
SMOOTH_WINDOW = 7        # moving-average window for histogram smoothing
XLIM_MAX = 5e9           # X-axis cap requested for mix_hist_ecdf (replicated in hist_smoothed)

# --- New: tail mass to cut when defining an upper truncation (1 - Fu) ---
TAIL_MASS_CUT = 0.016    # -> Fu = 1 - 0.016 = 0.984 (similar to your prior examples)
TRUNC_ROUND_TO = None    # e.g., 100.0 to round to nearest 100; keep None to not round

FIG_SIZE = (8, 4)   # inches (width, height)
DPI = 300

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

def ecdf_xy(vals):
    x = np.sort(vals); n = x.size
    y = np.arange(1, n+1, dtype=float) / n
    return x, y

def smoothed_hist(vals, bins=HIST_BINS, smooth_window=SMOOTH_WINDOW):
    """
    Returns (edges, smoothed_density) with total area = 1.
    """
    dens, edges = np.histogram(vals, bins=bins, density=True)
    if smooth_window and smooth_window > 1:
        k = int(smooth_window); kernel = np.ones(k)/k
        dens = np.convolve(dens, kernel, mode="same")
    widths = np.diff(edges)
    area = float(np.sum(dens * widths))
    if area > 0:
        dens = dens / area  # ensure ∑ dens_i * width_i = 1
    return edges, dens

# ======= Mixture of shifted lognormals (CDF) =======
def mixture_cdf_lognorm_shift(x, w, mu, sigma, loc):
    x = np.asarray(x, dtype=float)
    F = np.zeros_like(x, dtype=float)
    mask = x > loc
    z = np.log(np.clip(x[mask] - loc, 1e-12, None))
    for wi, m, s in zip(w, mu, sigma):
        F[mask] += wi * stats.norm.cdf((z - m) / s)
    return np.clip(F, 0.0, 1.0)

# --- New: simple inverse CDF via bisection (monotonic CDF) ---
def inv_cdf_mixture(Ftarget, w, mu, sigma, loc, data_max, tol=1e-6, maxit=100):
    """
    Find x such that mixture_cdf_lognorm_shift(x)=Ftarget using bisection.
    Expands the upper bound if needed until the target is bracketed.
    """
    # Lower bound just above loc
    low = float(max(loc + 1e-9, 0.0))
    # Start high from max observed or XLIM_MAX, whichever larger
    high = float(max(data_max, XLIM_MAX))
    # Expand high until we bracket Ftarget or cap the expansion
    for _ in range(60):
        if mixture_cdf_lognorm_shift([high], w, mu, sigma, loc)[0] >= Ftarget:
            break
        high *= 1.5
        if high > 1e13:
            # give up expansion; caller should fallback to empirical quantile
            return None

    # Bisection
    for _ in range(maxit):
        mid = 0.5 * (low + high)
        Fmid = mixture_cdf_lognorm_shift([mid], w, mu, sigma, loc)[0]
        if abs(Fmid - Ftarget) <= tol * max(1.0, Ftarget):
            return mid
        if Fmid < Ftarget:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

def compute_upper_truncation(best, data, tail_mass_cut=TAIL_MASS_CUT):
    """
    Compute upper truncation 'u*' such that F(u*) = Fu = 1 - tail_mass_cut.
    Falls back to empirical quantile if the model-based inverse fails.
    """
    Fu = float(1.0 - tail_mass_cut)
    w, mu, sigma, loc = best["w"], best["mu"], best["sigma"], best["loc"]
    u_star = inv_cdf_mixture(Fu, w, mu, sigma, loc, data_max=float(np.nanmax(data)))
    if u_star is None or not np.isfinite(u_star):
        # Fallback: empirical Fu-quantile from data
        u_star = float(np.quantile(data, Fu))

    if TRUNC_ROUND_TO and TRUNC_ROUND_TO > 0:
        u_star = float(np.round(u_star / TRUNC_ROUND_TO) * TRUNC_ROUND_TO)

    return u_star, Fu

# ======= Fit (ECDF) =======
def fit_mix_lognorm_shift_ecdf(v, k_range=(1,2,3), loc_grid=None,
                               random_state=RANDOM_STATE, n_init=5):
    x_s, y_s = ecdf_xy(v)
    vmin = float(np.min(v))
    if loc_grid is None:
        lo = vmin - 800.0*MS2FLOPS; hi = vmin - 100.0*MS2FLOPS
        loc_grid = np.linspace(lo, hi, 16)

    best, records = None, []
    for loc in loc_grid:
        u = v - loc
        if np.any(u <= 0):
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
            rec = dict(model="Mix Lognorm Function", loc=loc, sse=sse, bic=bic,
                       w=w, mu=mu, sigma=sigma)
            records.append(rec)
            if (best is None) or (sse < best["sse"]):
                best = rec

    summary = pd.DataFrame(records).sort_values(["sse","bic"]).reset_index(drop=True)
    return (x_s, y_s), summary, best

# ============== Sampling =================
def sample_shifted_lognorm_mixture(n, loc, w, mu, sigma, random_state=None):
    rng = np.random.default_rng(random_state)
    w   = np.asarray(w, float); w = w / w.sum()
    mu  = np.asarray(mu, float)
    sig = np.asarray(sigma, float)
    comp = rng.choice(len(w), size=n, p=w)
    z    = rng.normal(loc=mu[comp], scale=sig[comp])
    return loc + np.exp(z)

def sample_with_components(n, loc, w, mu, sigma, random_state=None):
    rng  = np.random.default_rng(random_state)
    w    = np.asarray(w, float); w = w / w.sum()
    mu   = np.asarray(mu, float)
    sig  = np.asarray(sigma, float)
    comp = rng.choice(len(w), size=n, p=w)
    z    = rng.normal(loc=mu[comp], scale=sig[comp])
    x    = loc + np.exp(z)
    return x, comp

# ===================== Plots (separados) ======================
def plot_histogram_smoothed(v, out_path=OUT_FIG_HIST, clip_xlim=True):
    ensure_dir(os.path.dirname(out_path) or "figures")
    edges, dens_sm = smoothed_hist(v, bins=HIST_BINS, smooth_window=SMOOTH_WINDOW)
    widths = np.diff(edges)
    area = float(np.sum(dens_sm * widths))
    print(f"[CHECK] Smoothed-hist area (should ≈ 1): {area:.6f}")

    # Bars (empirical smoothed PDF) + step outline
    plt.figure(figsize=FIG_SIZE, layout="constrained")
    plt.bar(edges[:-1], dens_sm, width=widths, align='edge',
            alpha=0.85, edgecolor="none",
            label="Empirical histogram")
    plt.step(edges, np.r_[dens_sm, dens_sm[-1]], where="post",
             lw=1.2, ls="--", color="k", alpha=0.7,
             label="Empirical PDF")

    plt.title("Empirical distribution")
    plt.xlabel(X_LABEL); plt.ylabel("Density")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")

    if clip_xlim:
        xmin = float(edges[0])
        plt.xlim(xmin, XLIM_MAX)

    plt.tight_layout(); plt.savefig(out_path, dpi=DPI); plt.close()

def plot_ecdf_validation(v, best, n_sim=N_SIM_MIN, out_path=OUT_FIG_ECDF, clip_xlim=True):
    ensure_dir(os.path.dirname(out_path) or "figures")
    x_ecdf, y_ecdf = ecdf_xy(v)
    xmin, xmax = float(x_ecdf.min()), float(x_ecdf.max())

    sim = sample_shifted_lognorm_mixture(n_sim, best["loc"], best["w"], best["mu"], best["sigma"],
                                         random_state=RANDOM_STATE)
    x_ecdf_sim, y_ecdf_sim = ecdf_xy(sim)
    xfit = np.linspace(xmin, xmax if clip_xlim else max(xmax, sim.max()), 1500)
    yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])

    plt.figure(figsize=(7,4.5))
    plt.step(x_ecdf, y_ecdf, where="post", lw=1.8, label="Empirical ECDF")
    plt.step(x_ecdf_sim, y_ecdf_sim, where="post", lw=1.8, label=f"Simulated ECDF (n={n_sim})")
    plt.plot(xfit, yfit, "--", lw=2.0, label="Model CDF")

    # Mark upper truncation if present
    if "upper_trunc" in best and np.isfinite(best["upper_trunc"]):
        plt.axvline(best["upper_trunc"], ls=":", lw=1.6, color="crimson", alpha=0.8,
                    label=f"Upper trunc (F={best.get('Fu', np.nan):.3f})")

    plt.title("Validation with simulated data")
    plt.xlabel(X_LABEL); plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    if clip_xlim:
        plt.xlim(xmin, xmax)
        plt.axvline(xmax, ls=":", lw=1.2, color="k", alpha=0.4)
    plt.ylim(0,1)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_mixture_hist_with_ecdf(x, comp, bins=HIST_BINS, mode="stacked", show_points=False,
                                out_path=OUT_FIG_MIXED):
    """
    Component bars (stacked/grouped) + simulated ECDF on secondary axis.
    """
    ensure_dir(os.path.dirname(out_path) or "figures")

    colors = ["C0", "C1", "C2"]
    labels = ["Component 1", "Component 2", "Component 3"]

    fig, ax1 = plt.subplots(figsize=FIG_SIZE, layout="constrained")
    edges = np.histogram_bin_edges(x, bins=bins)
    widths = np.diff(edges)
    B = len(widths)
    lefts = edges[:-1]

    # Component densities (sum to total)
    K = int(np.max(comp)) + 1
    n = x.size
    dens = []
    for k in range(K):
        cnt, _ = np.histogram(x[comp == k], bins=edges)
        dens.append(cnt / (n * widths))
    dens = np.vstack(dens)

    if mode == "stacked":
        bottom = np.zeros(B)
        for k, (c, lbl) in enumerate(zip(colors, labels)):
            if k >= dens.shape[0]: break
            ax1.bar(lefts, dens[k], width=widths, align="edge",
                    bottom=bottom, color=c, alpha=0.6, edgecolor="none",
                    label=lbl)
            bottom += dens[k]
        total = bottom
    elif mode == "grouped":
        g = len(labels)
        for k, (c, lbl) in enumerate(zip(colors, labels)):
            if k >= dens.shape[0]: break
            ax1.bar(lefts + (k / g) * widths, dens[k], width=widths / g,
                    align="edge", color=c, alpha=0.8, edgecolor="none",
                    label=lbl)
        total = dens.sum(axis=0)
    else:
        raise ValueError("mode must be 'stacked' or 'grouped'.")

    # Total density outline
    ax1.step(edges, np.r_[total, total[-1]], where="post",
             lw=1.2, ls="--", color="k", alpha=0.7, label="Simulated PDF")

    ax1.set_xlabel(X_LABEL)
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3, zorder=0)

    # === Simulated ECDF with halo ===
    ax2 = ax1.twinx()
    xs_all = np.sort(x)
    ys_all = np.arange(1, x.size + 1) / x.size

    # 1) White halo
    ax2.step(xs_all, ys_all, where="post",
             lw=3.0, color="1.0", alpha=1.0, zorder=1)
    # 2) Main dashed line
    ax2.step(xs_all, ys_all, where="post",
             lw=1.3, ls=(0, (4, 2)), color="0.15", alpha=1.0, zorder=2,
             label="Simulated CDF")

    if show_points:
        order = np.argsort(x)
        comp_sorted = comp[order]
        idx = np.linspace(0, x.size - 1, min(400, x.size), dtype=int)
        ax2.scatter(xs_all[idx], ys_all[idx], s=14,
                    c=np.take(colors, comp_sorted[idx], mode='wrap'),
                    alpha=0.9, edgecolors="white", linewidths=0.4, zorder=3)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Cumulative Probability")
    ax2.tick_params(axis="y", colors="0.15")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    if h2:
        ax1.legend(h1 + [h2[0]], l1 + [l2[0]], loc="best")
    else:
        ax1.legend(loc="best")

    # X limit up to XLIM_MAX (as requested)
    xmin_plot = float(edges[0])
    ax1.set_xlim(xmin_plot, XLIM_MAX)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI)
    plt.close()

# ===================== Main ======================
def main():
    ensure_dir("figures"); ensure_dir(os.path.dirname(OUT_TXT) or ".")
    df = load_data(SYS_FILE)
    v = df["FLOPs"].to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]

    (x_ecdf, y_ecdf), summary, best = fit_mix_lognorm_shift_ecdf(
        v, k_range=(1,2,3), loc_grid=None, random_state=RANDOM_STATE, n_init=5
    )

    # --- Compute upper truncation point (model-based) ---
    upper_trunc, Fu = compute_upper_truncation(best, v, tail_mass_cut=TAIL_MASS_CUT)
    best["upper_trunc"] = upper_trunc
    best["Fu"] = Fu

    # KS (model vs ECDF)
    Fm_ecdf = mixture_cdf_lognorm_shift(x_ecdf, best["w"], best["mu"], best["sigma"], best["loc"])
    ks_ecdf = float(np.max(np.abs(Fm_ecdf - y_ecdf)))

    print("\n=== Best models (sorted by SSE) ===")
    print(summary.head(10).to_string(index=False))

    with open(OUT_TXT, "w") as f:
        f.write(f"Best: {best['model']}  loc={best['loc']}\n")
        for i, (wi, mi, si) in enumerate(zip(best["w"], best["mu"], best["sigma"]), 1):
            scale_i = np.exp(mi)
            f.write(f" comp{i}: w={wi:.6f}, mu_log={mi:.6f}, sigma={si:.6f}, scale=exp(mu)={scale_i:.6f}\n")
        # --- Save truncation metadata ---
        f.write(f"upper_trunc={upper_trunc:.6f}  Fu={Fu:.6f}  tail_mass_cut={TAIL_MASS_CUT:.6f}\n")
        f.write(f"SSE={best['sse']:.6f}, BIC={best['bic']:.2f}\n")
        f.write(f"KS_ECDF={ks_ecdf:.6f}\n")

    print(f"\nParameters saved to: {OUT_TXT}")
    print(f"upper_trunc={upper_trunc:.1f}, Fu={Fu:.6f}, tail_mass_cut={TAIL_MASS_CUT:.6f}")

    # Separate figures
    plot_histogram_smoothed(v, out_path=OUT_FIG_HIST, clip_xlim=True)
    print(f"Histogram figure saved to: {OUT_FIG_HIST}")

    plot_ecdf_validation(v, best, n_sim=N_SIM_MIN, out_path=OUT_FIG_ECDF, clip_xlim=True)
    print(f"ECDF validation figure saved to: {OUT_FIG_ECDF}")

    # Component bars + simulated ECDF
    x_sim, comp_sim = sample_with_components(N_SIM_MIN, best["loc"], best["w"], best["mu"], best["sigma"],
                                             random_state=RANDOM_STATE)
    plot_mixture_hist_with_ecdf(x_sim, comp_sim, bins=HIST_BINS, mode="stacked", show_points=False,
                                out_path=OUT_FIG_MIXED)
    print(f"Mixture histogram + ECDF figure saved to: {OUT_FIG_MIXED}")

    # Stats
    sim = sample_shifted_lognorm_mixture(N_SIM_MIN, best["loc"], best["w"], best["mu"], best["sigma"],
                                         random_state=RANDOM_STATE)
    print("\n=== Summary statistics ===")
    print(f"Real:      mean={np.mean(v):.1f}, median={np.median(v):.1f}, std={np.std(v, ddof=1):.1f}")
    print(f"Simulated: mean={np.mean(sim):.1f}, median={np.median(sim):.1f}, std={np.std(sim, ddof=1):.1f}")
    print(f"(n_real={len(v)}, n_sim={N_SIM_MIN})")
    print(f"KS_ECDF={ks_ecdf:.6f}")

if __name__ == "__main__":
    main()
