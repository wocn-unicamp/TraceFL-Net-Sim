#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

# ===================== Config =====================
PATH_FOLDER = "../results/sys/"
EXT_FILE    = ".csv"


# --- NUEVO: umbral mínimo de pesos y fuerza de penalización ---
W_MIN      = 0.20      # ningún componente debe quedar por debajo de 20%
LAMBDA_W   = 0.1     # fuerza de la penalización (ajústalo según tu caso)

file_bases_2 = [
    "sys_metrics_minibatch_c_20_mb_0.2",
    "sys_metrics_minibatch_c_20_mb_1",
    "sys_metrics_minibatch_c_20_mb_0.9",
    "sys_metrics_minibatch_c_20_mb_0.8",
    "sys_metrics_minibatch_c_20_mb_0.6",
    "sys_metrics_minibatch_c_20_mb_0.5",
    "sys_metrics_minibatch_c_20_mb_0.4",
]


# Ajuste / plots
BINS, SMOOTH_WINDOW = 100, 10
RANDOM_STATE = 42
N_SIM_MIN    = 1000
OUT_DIR      = "fit"

# ===================== Utils ======================
def ensure_dir(path=OUT_DIR):
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

def smoothed_cdf_from_hist(vals, bins=256, smooth_window=7):
    counts, edges = np.histogram(vals, bins=bins, density=True)
    if smooth_window and smooth_window > 1:
        k = int(smooth_window)
        kernel = np.ones(k)/k
        counts = np.convolve(counts, kernel, mode="same")
    widths = np.diff(edges)
    cdf = np.cumsum(counts * widths)
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    x = edges[1:]
    return x, cdf

def empirical_truncation_point(v, method="smoothed", q=1.0, eps=1e-4, x_sm=None, y_sm=None):
    v = np.asarray(v, float)
    if method == "max":
        return float(np.max(v))
    elif method == "quantile":
        return float(np.quantile(v, q))
    elif method == "smoothed":
        if x_sm is None or y_sm is None:
            return float(np.max(v))
        idx = np.where(y_sm >= 1.0 - eps)[0]
        return float(x_sm[idx[0]]) if idx.size else float(np.max(v))
    else:
        raise ValueError("method debe ser 'max' | 'quantile' | 'smoothed'")

# ======= Mixture of shifted lognormals =======
def mixture_cdf_lognorm_shift(x, w, mu, sigma, loc):
    x = np.asarray(x, dtype=float)
    F = np.zeros_like(x, dtype=float)
    mask = x > loc
    z = np.log(np.clip(x[mask] - loc, 1e-12, None))
    for wi, m, s in zip(w, mu, sigma):
        F[mask] += wi * stats.norm.cdf((z - m) / s)
    return np.clip(F, 0.0, 1.0)

def mixture_cdf_lognorm_shift_trunc(x, w, mu, sigma, loc, upper=None):
    F = mixture_cdf_lognorm_shift(x, w, mu, sigma, loc)
    if upper is None:
        return F
    Fu = mixture_cdf_lognorm_shift(np.array([upper]), w, mu, sigma, loc)[0]
    if Fu <= 0.0:
        return np.where(np.asarray(x) <= upper, 0.0, 1.0)
    Ft = np.where(np.asarray(x) <= upper, F / Fu, 1.0)
    return np.clip(Ft, 0.0, 1.0)

# ---------- Búsqueda de loc (gruesa + fina), k=3 ----------
def loc_grid_coarse(v, n_points=41):
    v = np.asarray(v, float)
    vmin = float(v.min())
    q80  = float(np.quantile(v, 0.80))
    spread = max(q80 - vmin, 1e-6)
    lo = vmin - 0.5 * spread
    hi = vmin - 0.02 * spread
    if hi >= vmin:
        hi = vmin - 1e-3 * max(vmin, 1.0)
    if lo >= hi:
        lo = hi - max(0.01 * max(vmin, 1.0), spread * 0.1)
    return np.linspace(lo, hi, n_points)

def loc_grid_refine(loc_best, v, factor=10.0, n_points=51):
    vmin = float(np.min(v))
    rad = max(abs(vmin) * factor * 0.01, 1e-6 * max(abs(vmin), 1.0))  # ±0.1% |vmin|
    lo = loc_best - rad
    hi = loc_best + rad
    return np.linspace(lo, hi, n_points)

# --------- Fit k=3: AJUSTE SOLO CONTRA LA ECDF ---------
def fit_mix_lognorm_shift_k3_empirical(
    v, loc_grid=None,
    bins=BINS, smooth_window=SMOOTH_WINDOW,
    random_state=RANDOM_STATE, n_init=20,
    truncate_to_empirical=True,
    trunc_method="smoothed", trunc_q=1.0, trunc_eps=1e-4
):
    # CDFs de referencia...
    x_sm, y_sm = smoothed_cdf_from_hist(v, bins=bins, smooth_window=smooth_window)
    x_ecdf, y_ecdf = ecdf_xy(v)

    if loc_grid is None:
        loc_grid = loc_grid_coarse(v, n_points=41)

    upper_trunc = empirical_truncation_point(
        v, method=trunc_method, q=trunc_q, eps=trunc_eps, x_sm=x_sm, y_sm=y_sm
    )

    def eval_loc_grid(locs):
        best_local = None
        records = []
        for loc in locs:
            u = v - loc
            if np.any(u <= 0):
                continue
            ylog = np.log(np.clip(u, 1e-12, None)).reshape(-1,1)

            gmm = GaussianMixture(
                n_components=3, covariance_type="diag",
                random_state=random_state, n_init=n_init,
                max_iter=500, reg_covar=1e-6, init_params="kmeans"
            )
            gmm.fit(ylog)
            w = gmm.weights_.ravel()
            mu = gmm.means_.ravel()
            sigma = np.maximum(np.sqrt(gmm.covariances_.ravel()), 1e-8)

            # Ordenar por media
            order = np.argsort(mu)
            w, mu, sigma = w[order], mu[order], sigma[order]

            # CDF del modelo en ECDF
            if truncate_to_empirical:
                yhat_e = mixture_cdf_lognorm_shift_trunc(x_ecdf, w, mu, sigma, loc, upper=upper_trunc)
            else:
                yhat_e = mixture_cdf_lognorm_shift(x_ecdf, w, mu, sigma, loc)

            # Métricas sobre ECDF
            mse_e = float(np.mean((yhat_e - y_ecdf)**2))
            rmse_e = float(np.sqrt(mse_e))
            ks    = float(np.max(np.abs(yhat_e - y_ecdf)))

            # --------- NUEVO: penalización por pesos pequeños ----------
            # penaliza linealmente el déficit por debajo de W_MIN
            deficit = np.clip(W_MIN - w, 0.0, None)   # vector >= 0
            pen_w   = float(LAMBDA_W * np.sum(deficit))
            score_unpen = mse_e
            score = mse_e + pen_w  # objetivo con penalización
            # -----------------------------------------------------------

            bic = float(gmm.bic(ylog))
            Fu  = mixture_cdf_lognorm_shift(np.array([upper_trunc]), w, mu, sigma, loc)[0]
            tail = max(0.0, 1.0 - float(Fu))

            rec = dict(
                model="mixlognorm(k=3)", loc=float(loc),
                sse_ecdf=mse_e * x_ecdf.size, mse_ecdf=mse_e, rmse_ecdf=rmse_e,
                ks=ks, score=score, score_unpenalized=score_unpen, penalty_w=pen_w,
                bic=bic,
                w=w, mu=mu, sigma=sigma,
                upper_trunc=float(upper_trunc), tail_mass=float(tail), Fu=float(Fu)
            )
            records.append(rec)

            # Selección: score penalizado → KS → BIC
            if (best_local is None
                or (score < best_local["score"] - 1e-12)
                or (abs(score - best_local["score"]) <= 1e-12 and ks < best_local["ks"] - 1e-12)
                or (abs(score - best_local["score"]) <= 1e-12 and abs(ks - best_local["ks"]) <= 1e-12 and bic < best_local["bic"])):
                best_local = rec
        return records, best_local

    # 1) Barrido grueso
    records1, best1 = eval_loc_grid(loc_grid)
    # 2) Refinamiento fino
    loc_fine = loc_grid_refine(best1["loc"], v, factor=10.0, n_points=51)
    records2, best2 = eval_loc_grid(loc_fine)

    summary = pd.DataFrame(records1 + records2).sort_values(["score","ks","bic"]).reset_index(drop=True)

    # Selección final (con score penalizado)
    if (best2["score"] < best1["score"] or
        (abs(best2["score"] - best1["score"]) <= 1e-12 and best2["ks"] <= best1["ks"])):
        best = best2
    else:
        best = best1

    return (x_sm, y_sm, x_ecdf, y_ecdf), summary, best

# ============== Sampling ==============
def sample_shifted_lognorm_mixture(n, loc, w, mu, sigma, random_state=None):
    rng = np.random.default_rng(random_state)
    w   = np.asarray(w, float); w = w / w.sum()
    mu  = np.asarray(mu, float)
    sig = np.asarray(sigma, float)
    k   = len(w)
    comp = rng.choice(k, size=n, p=w)
    z    = rng.normal(loc=mu[comp], scale=sig[comp])
    x    = loc + np.exp(z)
    return x

# ===================== Plot: Fit + Sim (2 subplots) ======================
def plot_fit_and_simulation(
    v, x_sm, y_sm, x_ecdf, y_ecdf, best, n_sim,
    out_path, clip_to_real_xlim=True, truncate_and_renormalize=True,
):
    ensure_dir(os.path.dirname(out_path) or OUT_DIR)

    xmin, xmax = float(x_ecdf.min()), float(x_ecdf.max())

    sim = sample_shifted_lognorm_mixture(
        n_sim, best["loc"], best["w"], best["mu"], best["sigma"], random_state=RANDOM_STATE
    )
    upper = best.get("upper_trunc", None)
    if upper is not None:
        sim = sim[sim <= upper]

    if truncate_and_renormalize:
        sim_vis = sim[(sim >= xmin) & (sim <= xmax)]
        x_ecdf_sim, y_ecdf_sim = ecdf_xy(sim_vis)
    else:
        x_ecdf_sim, y_ecdf_sim = ecdf_xy(sim)

    xfit = np.linspace(xmin, xmax if clip_to_real_xlim else max(xmax, sim.max()), 1500)
    yfit = mixture_cdf_lognorm_shift_trunc(
        xfit, best["w"], best["mu"], best["sigma"], best["loc"],
        upper=best.get("upper_trunc", xmax if clip_to_real_xlim else None)
    )

    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2, sharey=True)

    # Left: ECDF + Smoothed + Model
    ax1.step(x_ecdf, y_ecdf, where="post", lw=1.6, label="Empirical ECDF")
    ax1.plot(x_sm, y_sm, lw=1.8, ls="--", label=f"Smoothed CDF (bins={BINS}, win={SMOOTH_WINDOW})")
    ax1.plot(xfit, yfit, "-", lw=2.2, label=f"{best['model']} (trunc)")
    ax1.set_title("Fit on real dataset (k=3) — objective: ECDF")
    ax1.set_xlabel("FLOPs")
    ax1.set_ylabel("Cumulative Probability")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")
    if clip_to_real_xlim:
        ax1.set_xlim(xmin, xmax)

    # Right: Real ECDF vs Simulated ECDF
    ax2.step(x_ecdf, y_ecdf, where="post", lw=1.6, label="Empirical ECDF")
    ax2.step(x_ecdf_sim, y_ecdf_sim, where="post", lw=1.6,
             label=("Simulated ECDF (truncated)" if truncate_and_renormalize
                    else f"Simulated ECDF (n={n_sim})"))
    ax2.set_title("Validation with simulated data")
    ax2.set_xlabel("FLOPs")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    if clip_to_real_xlim:
        ax2.set_xlim(xmin, xmax)
        ax2.axvline(xmax, ls=":", lw=1.2, color="k", alpha=0.4)

    yticks = np.linspace(0, 1, 6)
    for ax in (ax1, ax2):
        ax.set_ylim(0, 1)
        ax.set_yticks(yticks)
    ax1.tick_params(axis="y", labelleft=True, labelright=False)
    ax2.tick_params(axis="y", labelleft=False, labelright=True)

    fig.suptitle("Shifted Lognormal Mixture (k=3): fit (ECDF) and validation", fontsize=14)
    plt.savefig(out_path, dpi=150)
    plt.close()

# ===================== Overlay final: Empírica + Smoothed + FIT ==========
def plot_overlay_emp_smooth_fit(all_items, out_path):
    """
    all_items: lista de tuplas (label, best_dict, v_array)
    Dibuja, por cada trace: ECDF (emp), Smoothed CDF (--) y FIT truncado (—).
    Eje X en **log**.
    """
    ensure_dir(os.path.dirname(out_path) or OUT_DIR)

    # Rango global en X (>0 para escala log)
    global_min = min(float(np.min(v)) for _, _, v in all_items)
    global_max = max(float(np.max(v)) for _, _, v in all_items)
    if global_min <= 0:
        global_min = np.nextafter(0.0, 1.0)
    xgrid = np.logspace(np.log10(global_min), np.log10(global_max), 2000)

    plt.figure(figsize=(12, 6))
    for label, best, v in all_items:
        x_e, y_e = ecdf_xy(v)
        x_sm, y_sm = smoothed_cdf_from_hist(v, bins=BINS, smooth_window=SMOOTH_WINDOW)
        yfit = mixture_cdf_lognorm_shift_trunc(
            xgrid, best["w"], best["mu"], best["sigma"], best["loc"],
            upper=best.get("upper_trunc", None)
        )
        line_fit, = plt.plot(xgrid, yfit, lw=2.2, label=f"{label} (fit)")
        color = line_fit.get_color()
        # plt.plot(x_sm, y_sm, lw=1.8, ls="--", color=color, label=f"{label} (smooth)")
        plt.step(x_e, y_e, where="post", ls="--", color=color, label=f"{label} (emp)")

    # plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("FLOPs (log scale)")
    plt.ylabel("Cumulative Probability")
    plt.title("Empirical ECDF + Smoothed CDF (--) + Truncated FIT — all traces (k=3, fit vs ECDF)")
    plt.grid(True, which="both", alpha=0.3)
    # plt.legend(loc="lower right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ===================== Main ======================
def main():
    ensure_dir(OUT_DIR)

    summary_rows = []
    overlay_items = []

    for base in file_bases_2:
        csv_path = os.path.join(PATH_FOLDER, base + EXT_FILE)
        print(f"\n=== Processing: {csv_path} ===")
        try:
            df = load_data(csv_path)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        v = df["FLOPs"].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size < 10:
            print("  [SKIP] Not enough data points.")
            continue

        # Fit k=3 SOLO respecto a la ECDF
        (x_sm, y_sm, x_ecdf, y_ecdf), summary, best = fit_mix_lognorm_shift_k3_empirical(
            v, loc_grid=None,
            bins=BINS, smooth_window=SMOOTH_WINDOW,
            random_state=RANDOM_STATE, n_init=20,
            truncate_to_empirical=True,
            trunc_method="smoothed", trunc_q=1.0, trunc_eps=1e-4
        )

        # Salidas por trace
        fig2 = os.path.join(OUT_DIR, f"{base}_fit_and_sim.png")
        txtp = os.path.join(OUT_DIR, f"{base}_params.txt")

        # TXT
        with open(txtp, "w") as f:
            f.write(f"Best: {best['model']}  loc={best['loc']}\n")
            if "upper_trunc" in best:
                f.write(f" upper_trunc={best['upper_trunc']:.6f}  Fu={best.get('Fu', float('nan')):.6f}  tail_mass_cut={best.get('tail_mass', float('nan')):.6f}\n")
            for i, (wi, mi, si) in enumerate(zip(best["w"], best["mu"], best["sigma"]), 1):
                scale_i = np.exp(mi)
                flag = "  <-- below W_MIN" if wi < W_MIN else ""
                f.write(f" comp{i}: w={wi:.6f}, mu_log={mi:.6f}, sigma={si:.6f}, scale=exp(mu)={scale_i:.6f}{flag}\n")
            f.write(f"SSE_ECDF={best['sse_ecdf']:.6f}, MSE_ECDF={best['mse_ecdf']:.6f}, RMSE_ECDF={best['rmse_ecdf']:.6f}, ")
            f.write(f"KS={best['ks']:.6f}, SCORE(=MSE+pen)={best['score']:.6f}, MSE_no_pen={best['score_unpenalized']:.6f}, PEN_w={best['penalty_w']:.6f}, BIC={best['bic']:.2f}\n")

        print(f"  Params saved to: {txtp}")

        # Figura (fit + sim)
        plot_fit_and_simulation(
            v, x_sm, y_sm, x_ecdf, y_ecdf, best,
            n_sim=max(N_SIM_MIN, 5 * len(v)),
            out_path=fig2,
            clip_to_real_xlim=True,
            truncate_and_renormalize=True
        )
        print(f"  Figure saved to: {fig2}")

        # Summary row
        # En summary_rows (para CSV global), añade columnas nuevas:
        summary_rows.append({
            "file_base": base,
            "model": best["model"],
            "k": 3,
            "loc": best["loc"],
            "upper_trunc": best.get("upper_trunc", np.nan),
            "Fu": best.get("Fu", np.nan),
            "tail_mass": best.get("tail_mass", np.nan),
            "mse_ecdf": best["mse_ecdf"],
            "rmse_ecdf": best["rmse_ecdf"],
            "ks": best["ks"],
            "score": best["score"],                     # con penalización
            "mse_no_pen": best["score_unpenalized"],    # sin penalización
            "penalty_w": best["penalty_w"],
            "bic": best["bic"],
            "w": json.dumps([float(x) for x in best["w"]]),
            "mu": json.dumps([float(x) for x in best["mu"]]),
            "sigma": json.dumps([float(x) for x in best["sigma"]]),
        })


        # Top-10 (opcional)
        top10_csv = os.path.join(OUT_DIR, f"{base}_summary_stop10.csv")
        summary.head(10).to_csv(top10_csv, index=False)
        print(f"  Top-10 summary saved to: {top10_csv}")

        overlay_items.append((base, best, v))

    # Summary global + overlay final
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        out_csv = os.path.join(OUT_DIR, "fit_summary.csv")
        df_sum.to_csv(out_csv, index=False)
        print(f"\n=== Global summary saved to: {out_csv} ===")

        overlay_png = os.path.join(OUT_DIR, "all_traces_overlay.png")
        plot_overlay_emp_smooth_fit(overlay_items, overlay_png)
        print(f"Overlay figure saved to: {overlay_png}")
    else:
        print("\nNo outputs generated (no valid inputs found).")

if __name__ == "__main__":
    main()
