#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

# ===================== Config =====================
PATH_FOLDER = "../results/sys/"
NAME_FILE   = "sys_metrics_minibatch_c_20_mb_0.2"
EXT_FILE    = ".csv"

BINS, SMOOTH_WINDOW = 256, 7
OUT_FIG  = "figures/mix_lognorm_shift_fit.png"
OUT_TXT  = "figures/mix_lognorm_shift_params.txt"

# ===================== Utils ======================
def ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True); return path

def load_data(sys_metrics_file):
    if not os.path.isfile(sys_metrics_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {sys_metrics_file}")
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
                                      random_state=42, n_init=5):
    x_s, y_s = smoothed_cdf_from_hist(v, bins=bins, smooth_window=smooth_window)

    vmin = float(np.min(v)); vmax = float(np.max(v))
    if loc_grid is None:
        # busca loc por debajo del mínimo (desplaza soporte)
        lo = vmin - 800.0; hi = vmin - 100.0
        loc_grid = np.linspace(lo, hi, 16)

    best = None
    records = []

    for loc in loc_grid:
        # datos en log-espacio con shift
        u = v - loc
        if np.any(u <= 0):  # loc no válido
            continue
        ylog = np.log(u).reshape(-1,1)

        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                  random_state=random_state, n_init=n_init)
            gmm.fit(ylog)
            w = gmm.weights_.ravel()
            mu = gmm.means_.ravel()
            sigma = np.sqrt(gmm.covariances_.ravel())

            # CDF de mezcla sobre los puntos de la Smoothed CDF
            yhat = mixture_cdf_lognorm_shift(x_s, w, mu, sigma, loc)
            sse  = float(np.sum((yhat - y_s)**2))
            bic  = float(gmm.bic(ylog))

            rec = dict(model=f"mixlognorm(k={k})", loc=loc, sse=sse, bic=bic,
                       w=w, mu=mu, sigma=sigma)
            records.append(rec)

            if (best is None) or (sse < best["sse"]):
                best = rec

    # ordenar por SSE y devolver mejor
    summary = pd.DataFrame(records).sort_values(["sse","bic"]).reset_index(drop=True)
    return (x_s, y_s), summary, best

# ===================== Plot ======================
def plot_all(v, x_sm, y_sm, best, out_path=OUT_FIG):
    ensure_dir(os.path.dirname(out_path) or "figures")
    x_ecdf, y_ecdf = ecdf_xy(v)
    xfit = np.linspace(v.min(), v.max(), 1200)
    yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])

    plt.figure(figsize=(11,5))
    plt.step(x_ecdf, y_ecdf, where="post", lw=1.8, label="ECDF (empírica)")
    plt.plot(x_sm, y_sm, lw=2.0, label=f"Smoothed CDF (bins={BINS}, win={SMOOTH_WINDOW})")
    plt.plot(xfit, yfit, "--", lw=2.2, label=f"CDF mezcla — {best['model']}, loc={best['loc']:.1f}")
    plt.ylim(0,1); plt.xlabel("Client Computation Time (ms)"); plt.ylabel("Cumulative Probability")
    plt.title("Smoothed CDF fit — Shifted Lognormal Mixture")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

# ===================== Main ======================
def main():
    sys_metrics_file = os.path.join(PATH_FOLDER, NAME_FILE + EXT_FILE)
    ensure_dir("figures")

    df = load_data(sys_metrics_file)
    df = convert_FLOPs_to_time(df)
    v = df["time"].to_numpy(float)
    v = v[np.isfinite(v) & (v > 0)]

    (x_sm, y_sm), summary, best = fit_mix_lognorm_shift_to_smoothed(v,
        k_range=(1,2,3),
        loc_grid=None,  # usa el default basado en min(v)
        bins=BINS, smooth_window=SMOOTH_WINDOW,
        random_state=42, n_init=5
    )

    # Reporte
    print("\n=== Mejores modelos (orden por SSE) ===")
    print(summary.head(10).to_string(index=False))

    # Parámetros legibles
    with open(OUT_TXT, "w") as f:
        f.write(f"Best: {best['model']}  loc={best['loc']}\n")
        for i, (wi, mi, si) in enumerate(zip(best["w"], best["mu"], best["sigma"]), 1):
            scale_i = np.exp(mi)  # escala en ms del componente i
            f.write(f" comp{i}: w={wi:.6f}, mu_log={mi:.6f}, sigma={si:.6f}, scale=exp(mu)={scale_i:.6f}\n")
        f.write(f"SSE={best['sse']:.6f}, BIC={best['bic']:.2f}\n")

    plot_all(v, x_sm, y_sm, best, out_path=OUT_FIG)
    print(f"\nParámetros guardados en: {OUT_TXT}")
    print(f"Figura guardada en:      {OUT_FIG}")

if __name__ == "__main__":
    main()
