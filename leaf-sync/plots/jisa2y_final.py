#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit a 3-component shifted lognormal mixture to per-round FLOPs, validate with ECDF,
and save figures + parameter files for multiple datasets.

Now with progress bars (tqdm) and a fast 2-phase fitting strategy:
- Coarse search over 'loc' using a subsample + few inits.
- Final refit on full data at the best 'loc' with more inits.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# -------- tqdm (fallback si no está instalado) --------
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))

# ===================== Config =====================
BASE_RESULTS_DIR = "../results/sys"
DATASETS = [
    "sys_metrics_fedavg_c_50_e_1",
    "sys_metrics_minibatch_c_20_mb_0.9",
    "sys_metrics_minibatch_c_20_mb_0.8",
    "sys_metrics_minibatch_c_20_mb_0.6",
    "sys_metrics_minibatch_c_20_mb_0.5",
    "sys_metrics_minibatch_c_20_mb_0.4",
    "sys_metrics_minibatch_c_20_mb_0.2",
]

FIG_DIR   = "figures/fit"
PARAM_DIR = "params"

RANDOM_STATE      = 42
HIST_BINS         = 64
N_SIM_VALIDATE    = 1000
TAIL_MASS_CUT     = 0.016      # keep 1 - TAIL_MASS_CUT mass; truncate the rest
SIGMA_MIN         = 0.02       # anti-spike floor in log-space
REG_COVAR         = 1e-3       # GMM covariance regularization

# ---- Estrategia dos fases (rápida y robusta) ----
LOC_GRID_POINTS   = 32         # puntos del grid de 'loc' (coarse)
FIT_SUBSAMPLE     = 20000      # 0 o None para desactivar; recomendado 20k
N_INIT_COARSE     = 5          # inits en fase coarse
N_INIT_FINAL      = 12         # inits en refit final (dataset completo)
MAX_ITER_GMM      = 200        # max_iter de EM (ligeramente mayor al default)

# ---- Restricción sobre loc ----

LOC_ONLY_POSITIVE = True     # fuerza loc >= 0
LOC_MARGIN_FRAC   = 0.05     # loc <= (1 - 0.05) * min(data)
LOC_GRID_POINTS   = 32       # ya lo tienes; lo reusamos


# ===================== Utils =====================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_flops(csv_path: str) -> np.ndarray:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "FLOPs" not in df.columns:
        df.columns = list(df.columns[:-1]) + ["FLOPs"]
    v = df["FLOPs"].to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    return v

def ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
    return xs, ys

# ===================== Mixture model (shifted lognormal) =====================
def mixture_cdf_shifted_lognorm(x, w, mu, sigma, loc):
    x = np.asarray(x, dtype=float)
    F = np.zeros_like(x, dtype=float)
    mask = x > loc
    if np.any(mask):
        z = np.log(np.clip(x[mask] - loc, 1e-12, None))
        for wi, m, s in zip(w, mu, sigma):
            F[mask] += wi * norm.cdf((z - m) / s)
    return np.clip(F, 0.0, 1.0)

def inv_cdf_mixture(Ftarget, w, mu, sigma, loc, data_max, tol=1e-6, maxit=100):
    lo = max(loc + 1e-9, 0.0)
    hi = float(max(data_max, lo + 1.0))
    for _ in range(60):
        if mixture_cdf_shifted_lognorm([hi], w, mu, sigma, loc)[0] >= Ftarget:
            break
        hi *= 1.5
        if hi > 1e13:
            return None
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        Fmid = mixture_cdf_shifted_lognorm([mid], w, mu, sigma, loc)[0]
        if abs(Fmid - Ftarget) <= tol * max(1.0, Ftarget):
            return mid
        if Fmid < Ftarget:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def compute_upper_truncation(best, data, tail_mass_cut=TAIL_MASS_CUT):
    Fu = float(1.0 - tail_mass_cut)
    w, mu, sigma, loc = best["w"], best["mu"], best["sigma"], best["loc"]
    u_star = inv_cdf_mixture(Fu, w, mu, sigma, loc, data_max=float(np.max(data)))
    if u_star is None or not np.isfinite(u_star):
        u_star = float(np.quantile(data, Fu))
    return u_star, Fu

def cdf_truncated(x, w, mu, sigma, loc, u_star):
    Fx = mixture_cdf_shifted_lognorm(x, w, mu, sigma, loc)
    Fu = mixture_cdf_shifted_lognorm([u_star], w, mu, sigma, loc)[0]
    Fu = max(Fu, 1e-12)
    z = np.clip(Fx / Fu, 0.0, 1.0)
    z = np.where(np.asarray(x) > u_star, 1.0, z)
    return z

def sample_shifted_lognorm_mixture(n, loc, w, mu, sigma, rng):
    w = np.asarray(w, float)
    w = w / w.sum()
    comp = rng.choice(len(w), size=n, p=w)
    z = rng.normal(loc=mu[comp], scale=sigma[comp], size=n)
    return loc + np.exp(z)

# ===================== Fitting =====================
def _make_loc_grid(v: np.ndarray,
                   n: int = LOC_GRID_POINTS,
                   margin_frac: float = LOC_MARGIN_FRAC) -> np.ndarray:
    """
    Genera un grid de 'loc' tal que:
      - si LOC_ONLY_POSITIVE:  0 <= loc < (1 - margin_frac) * min(v)
      - siempre estrictamente por debajo de min(v) para garantizar v - loc > 0
    """
    v = np.asarray(v, float)
    vmin = float(np.min(v))
    if vmin <= 0:
        raise ValueError("All FLOPs must be > 0 to fit a shifted lognormal.")

    # límite superior permitido para loc, siempre < vmin
    hi = (1.0 - margin_frac) * vmin
    # por seguridad si el margen aprieta demasiado
    if hi <= 0:
        hi = 0.9 * vmin

    if LOC_ONLY_POSITIVE:
        lo = 0.0
    else:
        # si alguna vez quieres permitir negativos, baja este 'lo'
        lo = -3.0 * (float(np.max(v)) - vmin)

    # grid básico y corte final para evitar que loc toque vmin
    grid = np.linspace(lo, hi, int(max(3, n)))
    grid = grid[(grid >= 0.0) & (grid < vmin * (1.0 - 1e-12))]  # fuerza positividad y separación de vmin
    if grid.size == 0:
        # fallback: unos cuantos valores positivos bien dentro del rango
        grid = np.linspace(0.0, 0.9 * vmin, 8)

    return np.unique(grid)


def _fit_gmm_log_data(ylog, n_init, random_state):
    return GaussianMixture(
        n_components=3,
        covariance_type="diag",
        random_state=random_state,
        n_init=n_init,
        reg_covar=REG_COVAR,
        max_iter=MAX_ITER_GMM,
        tol=1e-3,
    ).fit(ylog)

def fit_shifted_lognorm_k3(v: np.ndarray, random_state: int = RANDOM_STATE):
    """
    Two-phase fitting:
      (A) Coarse: subsample + few inits across a loc grid -> pick best loc by KS.
      (B) Final: refit on full data at best loc with more inits -> return final model.
    """
    rng = np.random.RandomState(random_state)

    # ----- A) COARSE -----
    grid = _make_loc_grid(v)
    if FIT_SUBSAMPLE and v.size > FIT_SUBSAMPLE:
        idx = rng.choice(v.size, FIT_SUBSAMPLE, replace=False)
        v_fit = v[idx]
    else:
        v_fit = v

    x_fit, y_fit = ecdf(v_fit)
    best = None
    records = []

    print(f"[COARSE] loc grid size={grid.size}, n_init={N_INIT_COARSE}, subsample={v_fit.size}", flush=True)

    for loc in tqdm(grid, desc="Coarse loc search", total=grid.size):
        u = v_fit - loc
        if np.any(u <= 0.0):
            continue
        ylog = np.log(u).reshape(-1, 1)

        try:
            gmm = _fit_gmm_log_data(ylog, n_init=N_INIT_COARSE, random_state=random_state)
        except Exception:
            continue

        w = gmm.weights_.ravel()
        mu = gmm.means_.ravel()
        sigma = np.sqrt(gmm.covariances_.ravel())
        if np.any(sigma < SIGMA_MIN):
            continue

        y_hat = mixture_cdf_shifted_lognorm(x_fit, w, mu, sigma, loc)
        ks = float(np.max(np.abs(y_hat - y_fit)))
        rec = dict(model="MixShiftedLogNorm(k=3)", loc=float(loc), w=w, mu=mu, sigma=sigma,
                   ks=ks, bic=float(gmm.bic(ylog)))
        records.append(rec)
        if (best is None) or (ks < best["ks"]):
            best = rec

    if best is None:
        raise RuntimeError("No valid mixture found in coarse phase. Check units or relax SIGMA_MIN/REG_COVAR.")

    # ----- B) FINAL REFIT en TODO el dataset -----
    loc_star = best["loc"]
    u_full = v - loc_star
    if np.any(u_full <= 0.0):
        raise RuntimeError("Final refit: invalid loc (produces nonpositive shifts).")

    ylog_full = np.log(u_full).reshape(-1, 1)
    print(f"[FINAL] Refit at best loc={loc_star:.3e} with n_init={N_INIT_FINAL} on full data (n={v.size})", flush=True)
    gmm_final = _fit_gmm_log_data(ylog_full, n_init=N_INIT_FINAL, random_state=random_state)

    w = gmm_final.weights_.ravel()
    mu = gmm_final.means_.ravel()
    sigma = np.sqrt(gmm_final.covariances_.ravel())

    # KS sobre TODO el dataset
    x_full, y_full = ecdf(v)
    y_hat_full = mixture_cdf_shifted_lognorm(x_full, w, mu, sigma, loc_star)
    ks_full = float(np.max(np.abs(y_hat_full - y_full)))

    best_final = dict(
        model="MixShiftedLogNorm(k=3)",
        loc=float(loc_star),
        w=w, mu=mu, sigma=sigma,
        ks=ks_full,
        bic=float(gmm_final.bic(ylog_full))
    )

    print(f"[INFO] Best loc={best_final['loc']:.3e}, KS(full)={best_final['ks']:.6f}, "
          f"BIC={best_final['bic']:.1f}, sigma_min={np.min(sigma):.3f}", flush=True)

    summary = pd.DataFrame(records).sort_values(["ks", "bic"]).reset_index(drop=True)
    return (x_full, y_full), summary, best_final

# ===================== Plotting =====================
def plot_hist_with_ecdf(v: np.ndarray, out_png: str):
    ensure_dir(os.path.dirname(out_png) or ".")
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    counts, edges = np.histogram(v, bins=HIST_BINS, density=True)
    if HIST_BINS >= 5:
        k = 7
        kernel = np.ones(k) / k
        counts = np.convolve(counts, kernel, mode="same")
    ax1.bar(edges[:-1], counts, width=np.diff(edges), align="edge", alpha=0.8, edgecolor="none", label="Estimated PDF")
    ax1.set_ylabel("Estimated PDF")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    xr, yr = ecdf(v)
    ax2.step(xr, yr, where="post", lw=1.6, ls="--", color="black", label="Empirical ECDF")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Cumulative Probability")

    ax1.set_xlabel("FLOPs")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def plot_ecdf_validation(v: np.ndarray, best: dict, out_png: str, n_sim: int = N_SIM_VALIDATE):
    ensure_dir(os.path.dirname(out_png) or ".")
    rng = np.random.RandomState(RANDOM_STATE)

    xr, yr = ecdf(v)
    x_grid = np.linspace(np.min(v), np.max(v), 1500)
    y_model = mixture_cdf_shifted_lognorm(x_grid, best["w"], best["mu"], best["sigma"], best["loc"])

    u_star, Fu = compute_upper_truncation(best, v, tail_mass_cut=TAIL_MASS_CUT)
    y_model_trunc = cdf_truncated(x_grid, best["w"], best["mu"], best["sigma"], best["loc"], u_star)

    x_sim = sample_shifted_lognorm_mixture(n_sim, best["loc"], best["w"], best["mu"], best["sigma"], rng=rng)
    xs, ys = ecdf(x_sim)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.step(xr, yr, where="post", lw=1.8, label="Empirical ECDF")
    ax.step(xs, ys, where="post", lw=1.6, ls="--", label=f"Simulated ECDF (n={n_sim})")
    ax.plot(x_grid, y_model, ":", lw=1.8, label="Model CDF")
    ax.plot(x_grid, y_model_trunc, "-", lw=1.2, alpha=0.8, label=f"Model CDF (trunc @ F={Fu:.3f})")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# def plot_combined_ecdf(curves: list, out_png: str, xmax_gflops: float = 6.0):
#     """
#     ECDF real, simulada y modelo (múltiples experimentos) en **GFLOPs**,
#     con dos leyendas y límite máximo en 5 GFLOPs (por defecto).
#     """
#     ensure_dir(os.path.dirname(out_png) or ".")
#     from matplotlib.lines import Line2D
#     from matplotlib.ticker import FixedLocator

#     # Paleta armónica
#     colors = plt.get_cmap("Set2").colors

#     # Figura ancha para dejar espacio a las leyendas
#     fig, ax = plt.subplots(figsize=(13, 5.8), dpi=300)

#     color_handles = []
#     seen_labels = set()

#     # Mapeo de nombres
#     name_map = {
#         "fedavg_c_50_e_1": "Batch 100%",
#         "minibatch_c_20_mb_0.9": "Batch 90%",
#         "minibatch_c_20_mb_0.8": "Batch 80%",
#         "minibatch_c_20_mb_0.6": "Batch 60%",
#         "minibatch_c_20_mb_0.5": "Batch 50%",
#         "minibatch_c_20_mb_0.4": "Batch 40%",
#         "minibatch_c_20_mb_0.2": "Batch 20%",
#     }

#     # --- Curvas (conversión a GFLOPs) ---
#     G = 1e-9  # FLOPs -> GFLOPs
#     for i, cur in enumerate(curves):
#         lbl = name_map.get(cur["label"], cur["label"])
#         color = colors[i % len(colors)]

#         xr = cur["x_real"]  * G
#         xs = cur["x_sim"]   * G
#         xm = cur["x_model"] * G

#         ax.step(xr, cur["y_real"], where="post", lw=2.0, color=color, alpha=0.9)
#         # ax.step(xs, cur["y_sim"],  where="post", lw=1.6, ls="--", color=color, alpha=0.9)
#         ax.plot(xm, cur["y_model"], lw=1.8, ls="--", color=color, alpha=0.9)

#         if lbl not in seen_labels:
#             color_handles.append(Line2D([0], [0], color=color, lw=2.8, label=lbl))
#             seen_labels.add(lbl)

#     # --- Ejes y estilo ---
#     ax.set_xlabel("Computational demand (GFLOPs)", fontsize=15, labelpad=6)
#     ax.set_ylabel("Clients (%)", fontsize=15, labelpad=6)
#     # ticks: X=13, Y=15
#     ax.tick_params(axis="x", labelsize=13, length=5, width=1)
#     ax.tick_params(axis="y", labelsize=15, length=5, width=1)
        


#     ax.set_ylim(0, 1)
#     ax.set_xlim(0, float(xmax_gflops))

#     # Ticks limpios en 0..5
#     ax.xaxis.set_major_locator(FixedLocator(np.arange(0, float(xmax_gflops) + 1e-9, 1.0)))

#     ax.grid(True, alpha=0.3)

#     # --- Leyenda 1: tipo de curva ---
#     style_handles = [
#         Line2D([0], [0], color="black", lw=2.0, linestyle="-",  label="Empirical CDF"),
#         # Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="Sim (ECDF)"),
#         Line2D([0], [0], color="black", lw=2.0, linestyle=":",  label="Model CDF"),
#     ]
#     leg_style = ax.legend(
#         handles=style_handles,
#         title="Curve Type",
#         title_fontsize=13,
#         fontsize=12,
#         loc="lower right",
#         bbox_to_anchor=(0.98, 0.5),
#         frameon=True,
#         fancybox=True,
#         edgecolor="0.6",
#         ncol=1,
#     )
#     ax.add_artist(leg_style)

#     # --- Leyenda 2: configuraciones / colores ---
#     leg_colors = ax.legend(
#         handles=color_handles,
#         title="Configuration",
#         title_fontsize=13,
#         fontsize=12,
#         loc="lower right",
#         bbox_to_anchor=(0.98, 0.0),
#         frameon=True,
#         fancybox=True,
#         edgecolor="0.6",
#     )
#     fig.add_artist(leg_colors)

#     # Márgenes para dejar espacio a leyendas
#     plt.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.12)

#     fig.savefig(out_png, bbox_inches="tight")
#     plt.close(fig)

def plot_combined_ecdf(curves: list, out_png: str, xmax_gflops: float = 6.0):
    """
    ECDF (real) y CDF (modelo) en GFLOPs, con eje Y en porcentaje (0..100).
    """
    ensure_dir(os.path.dirname(out_png) or ".")
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    colors = plt.get_cmap("Set2").colors
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=300)

    color_handles, seen_labels = [], set()

    name_map = {
        "fedavg_c_50_e_1": "Batch 100%",
        "minibatch_c_20_mb_0.9": "Batch 90%",
        "minibatch_c_20_mb_0.8": "Batch 80%",
        "minibatch_c_20_mb_0.6": "Batch 60%",
        "minibatch_c_20_mb_0.5": "Batch 50%",
        "minibatch_c_20_mb_0.4": "Batch 40%",
        "minibatch_c_20_mb_0.2": "Batch 20%",
    }

    # --- Curvas (GFLOPs en X, % en Y) ---
    G = 1e-9
    for i, cur in enumerate(curves):
        lbl = name_map.get(cur["label"], cur["label"])
        color = colors[i % len(colors)]

        xr = cur["x_real"]  * G
        xm = cur["x_model"] * G

        # y -> porcentaje
        y_real_pct  = cur["y_real"]  * 100.0
        y_model_pct = cur["y_model"] * 100.0

        ax.step(xr, y_real_pct, where="post", lw=2.0, color=color, alpha=0.9)
        ax.plot(xm, y_model_pct, lw=1.8, ls="--", color=color, alpha=0.9)

        if lbl not in seen_labels:
            color_handles.append(Line2D([0], [0], color=color, lw=2.8, label=lbl))
            seen_labels.add(lbl)

    # --- Ejes y estilo ---
    ax.set_xlabel("Computational demand (GFLOPs)", fontsize=15, labelpad=6)
    ax.set_ylabel("Clients (%)", fontsize=15, labelpad=6)

    # ticks: X=13, Y=15
    ax.tick_params(axis="x", labelsize=13, length=5, width=1)
    ax.tick_params(axis="y", labelsize=15, length=5, width=1)

    # Y en 0..100 %
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

    # X en 0..xmax_gflops con ticks enteros
    ax.set_xlim(0, float(xmax_gflops))
    ax.xaxis.set_major_locator(FixedLocator(np.arange(0, float(xmax_gflops) + 1e-9, 1.0)))

    ax.grid(True, alpha=0.3)

    # --- Leyenda 1: tipo de curva ---
    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-",  label="Empirical CDF"),
        Line2D([0], [0], color="black", lw=2.0, linestyle=":",  label="Model CDF"),
    ]
    leg_style = ax.legend(
        handles=style_handles,
        title="Curve Type",
        title_fontsize=13,
        fontsize=12,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
        fancybox=True,
        edgecolor="0.6",
        ncol=1,
    )
    ax.add_artist(leg_style)

    # --- Leyenda 2: configuraciones / colores ---
    leg_colors = ax.legend(
        handles=color_handles,
        title="Configuration",
        title_fontsize=13,
        fontsize=12,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.0),
        frameon=True,
        fancybox=True,
        edgecolor="0.6",
    )
    fig.add_artist(leg_colors)

    plt.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.12)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ===================== Main =====================
def main():
    ensure_dir(FIG_DIR)
    ensure_dir(PARAM_DIR)

    all_curves = []

    print(f"Datasets: {len(DATASETS)}  |  FIG_DIR={FIG_DIR}  PARAM_DIR={PARAM_DIR}", flush=True)

    for base in tqdm(DATASETS, desc="Datasets"):
        print("\n" + "=" * 80, flush=True)
        print(f"Processing: {base}", flush=True)
        print("=" * 80, flush=True)

        csv_path = os.path.join(BASE_RESULTS_DIR, f"{base}.csv")
        tag = base.replace("sys_metrics_", "")
        out_hist = os.path.join(FIG_DIR, f"{tag}_hist_ecdf.png")
        out_ecdf = os.path.join(FIG_DIR, f"{tag}_ecdf_validation.png")
        out_txt  = os.path.join(PARAM_DIR, f"{tag}_mix_params.txt")

        v = load_flops(csv_path)
        print(f"Loaded {len(v)} valid FLOPs", flush=True)

        (x_ecdf_v, y_ecdf_v), summary, best = fit_shifted_lognorm_k3(v, random_state=RANDOM_STATE)

        # Upper truncation (for reporting)
        u_star, Fu = compute_upper_truncation(best, v, tail_mass_cut=TAIL_MASS_CUT)

        # Save params
        with open(out_txt, "w") as f:
            f.write(f"{best['model']} | loc={best['loc']:.6e}\n")
            for i, (wi, mi, si) in enumerate(zip(best["w"], best["mu"], best["sigma"]), 1):
                f.write(f" comp{i}: w={wi:.6f}, mu_log={mi:.6f}, sigma={si:.6f}\n")
            f.write(f"BIC={best['bic']:.2f}  KS={best['ks']:.6f}\n")
            f.write(f"upper_trunc={u_star:.6e}  Fu={Fu:.6f}  tail_mass_cut={TAIL_MASS_CUT:.6f}\n")

        # Figures
        plot_hist_with_ecdf(v, out_png=out_hist)
        plot_ecdf_validation(v, best, out_png=out_ecdf, n_sim=N_SIM_VALIDATE)

        # Curves for combined figure
        x_model = np.linspace(np.min(v), np.max(v), 1500)
        y_model = mixture_cdf_shifted_lognorm(x_model, best["w"], best["mu"], best["sigma"], best["loc"])


        # CDF del modelo SIN truncar (opcional si la quieres guardar)
        y_model_full = mixture_cdf_shifted_lognorm(
            x_model, best["w"], best["mu"], best["sigma"], best["loc"]
        )
        y_model_trunc = cdf_truncated(
            x_model, best["w"], best["mu"], best["sigma"], best["loc"], u_star
        )   

        # Simulada (si la usas en otro lugar)
        rng = np.random.RandomState(RANDOM_STATE)
        x_sim = sample_shifted_lognorm_mixture(N_SIM_VALIDATE, best["loc"], best["w"], best["mu"], best["sigma"], rng)
        x_sim.sort()
        y_sim = np.arange(1, x_sim.size + 1) / x_sim.size

        all_curves.append({
            "label": tag,
            "x_real": x_ecdf_v, "y_real": y_ecdf_v,
            "x_sim": x_sim,     "y_sim": y_sim,
            "x_model": x_model, "y_model": y_model_trunc,   # <<--- usar TRUNCADA
            "u_star": u_star,   # (opcional) por si luego quieres anotarlo
        })

        # rng = np.random.RandomState(RANDOM_STATE)
        # x_sim = sample_shifted_lognorm_mixture(N_SIM_VALIDATE, best["loc"], best["w"], best["mu"], best["sigma"], rng)
        # x_sim.sort()
        # y_sim = np.arange(1, x_sim.size + 1) / x_sim.size

        # all_curves.append({
        #     "label": tag,
        #     "x_real": x_ecdf_v, "y_real": y_ecdf_v,
        #     "x_sim": x_sim,     "y_sim": y_sim,
        #     "x_model": x_model, "y_model": y_model,
        # })

        print(f"✅ {tag}: KS={best['ks']:.5f}  →  params: {out_txt}", flush=True)

    out_all = os.path.join(FIG_DIR, "ALL_ecdf_real_sim_model.png")
    plot_combined_ecdf(all_curves, out_png=out_all)
    print(f"\n📊 Combined ECDF figure saved to: {out_all}", flush=True)

if __name__ == "__main__":
    main()
