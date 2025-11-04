#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator
from scipy import stats
from sklearn.mixture import GaussianMixture

# ===================== Config =====================
SYS_FILE         = "../results/sys/sys_metrics_fedavg_c_50_e_1.csv"
OUT_FIG_HIST     = "figures/fit/hist_smoothed.png"
OUT_FIG_ECDF     = "figures/fit/ecdf_validation.png"
OUT_FIG_MIXED    = "figures/fit/mix_hist_ecdf.png"
OUT_TXT          = "params/mix_lognorm_shift_params.txt"

RANDOM_STATE   = 42
N_SIM_MIN      = 1000
X_LABEL        = "GFLOPs"
HIST_BINS      = 64
SMOOTH_WINDOW  = 7
XLIM_MAX       = 5e9
TAIL_MASS_CUT  = 0.016
TRUNC_ROUND_TO = None

# ====== Estilo global ======
FIG_CFG   = {"size": (8,5), "dpi": 300, "tight": True, "grid_alpha": 0.30}
FONT_CFG  = {"title": 20, "axis": 18, "ticks": 18, "legend": 18}
TICK_CFG  = {"len": 4, "width": .8, "xrot": 0, "yrot": 0, "xpad": 4, "ypad": 4}
LEGEND_CFG= {"loc": "best", "frame": True, "ncol": 1}

# ---- Eje X: 0 .. 5e9, step 0.5e9; etiquetas solo en enteros
AX_CFG = {
    "xlim":  (0.15e9, 5.0e9),
    "xticks": np.arange(1.0e9, 5.0e9 + 1.0, 0.5e9),
}

def _x_tick_formatter(val, pos):
    g = val / 1e9
    if abs(g - round(g)) < 1e-9:
        return f"{int(round(g))}"
    return ""

X_MAJOR_LOCATOR = FixedLocator(AX_CFG["xticks"])
X_MAJOR_FORMAT  = FuncFormatter(_x_tick_formatter)

TEXTS = {
    "common": {
        "yl_cumprob": "Cumulative Probability",
        "upper_mark_fmt": "Upper trunc (F={Fu:.3f})",
    },
    "hist":  {
        "title": " ",
        "yl":    "Estimated PDF",
        "labels": {"pdf":"Estimated PDF", "ecdf":"Empirical CDF"},
    },
    "ecdf":  {
        "title": " ",
        "yl":    "Cumulative Probability",
        "labels": {"real":"Empirical ECDF","sim":"Simulated ECDF","model":"Model CDF"}
    },
    "mixed": {
        "title": " ",
        "yl":"Estimated PDF",
        "labels":{"pdf":"Simulated PDF","cdf":"Simulated CDF","c1":"Component 1","c2":"Component 2","c3":"Component 3"}
    }
}

def _figure():
    return plt.subplots(figsize=FIG_CFG["size"], dpi=FIG_CFG["dpi"],
                        layout="constrained" if FIG_CFG["tight"] else None)

def _style_xaxis(ax):
    ax.set_xlim(*AX_CFG["xlim"])
    ax.xaxis.set_major_locator(X_MAJOR_LOCATOR)
    ax.xaxis.set_major_formatter(X_MAJOR_FORMAT)
    ax.set_xlabel(X_LABEL, fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["xpad"])
    ax.tick_params(axis="x", labelsize=FONT_CFG["ticks"],
                   length=TICK_CFG["len"], width=TICK_CFG["width"])
    for t in ax.get_xticklabels(): t.set_rotation(TICK_CFG["xrot"])

def _style_yaxis(ax):
    ax.tick_params(axis="y", labelsize=FONT_CFG["ticks"],
                   length=TICK_CFG["len"], width=TICK_CFG["width"])
    for t in ax.get_yticklabels(): t.set_rotation(TICK_CFG["yrot"])

def _style(ax):
    ax.grid(True, alpha=FIG_CFG["grid_alpha"])
    _style_xaxis(ax); _style_yaxis(ax)

def _legend(ax):
    ax.legend(loc=LEGEND_CFG["loc"], frameon=LEGEND_CFG["frame"],
              ncol=LEGEND_CFG["ncol"], fontsize=FONT_CFG["legend"])

# ===================== Estrictitud de cola =====================
CALIBRATE_QUANTILE           = False
USE_TRUNCATED_FOR_VALIDATION = True

# Pesos base de la pérdida (puedes afinar)
W_SSE   = 1.0
W_KS    = 0.6
W_TAIL  = 0.35   # ⬅️ subimos un poco el peso de la cola
TAIL_GATE = 0.95
QUANT_PEN = [(0.50, 0.2), (0.90, 0.6)]
Q_TARGET  = 0.995

# ===== Mejoras anti-spike y estabilidad del GMM =====
ENFORCE_MIN_SIGMA = 0.02   # ⬅️ sigma mínima aceptable en log-espacio
REG_COVAR = 1e-3           # ⬅️ regularización de covarianza
GMM_N_INIT = 25            # ⬅️ más reinicios aleatorios para estabilidad

# Penalización adicional por sigmas pequeñas (en composite loss)
LAMBDA_SIGMA_PEN = 1e-3    # ⬅️ peso de penalización anti-spike

# ===================== Utils ======================
def ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True); return path

def load_data(sys_metrics_file):
    if not os.path.isfile(sys_metrics_file):
        raise FileNotFoundError(f"File not found: {sys_metrics_file}")
    df = pd.read_csv(sys_metrics_file)
    df.columns = ["client_id","round_number","idk","samples","set","bytes_read","bytes_written","FLOPs"]
    df.index.name = "index"
    return df

def ecdf_xy(vals):
    x = np.sort(vals); n = x.size
    y = np.arange(1, n+1, dtype=float) / n
    return x, y

def smoothed_hist(vals, bins=HIST_BINS, smooth_window=SMOOTH_WINDOW):
    dens, edges = np.histogram(vals, bins=bins, density=True)
    if smooth_window and smooth_window > 1:
        k = int(smooth_window); dens = np.convolve(dens, np.ones(k)/k, mode="same")
    widths = np.diff(edges); area = float(np.sum(dens * widths))
    if area > 0: dens = dens / area
    return edges, dens

def trunc_rand(flops, upper, on=True, rng=None):
    if not (on and upper is not None): return flops
    rng = np.random.default_rng() if rng is None else rng
    u = np.asarray(upper); m = flops > u
    if not np.any(m): return flops
    eps = np.finfo(float).eps
    if np.ndim(u) == 0:
        lo, hi = 0.8*float(u)*(1+eps), float(u); flops[m] = rng.uniform(lo, hi, m.sum())
    else:
        lo, hi = 0.8*u[m]*(1+eps), u[m]; flops[m] = rng.uniform(lo, hi)
    return flops

# ======= Mixture of shifted lognormals (CDF/PDF) =======
def mixture_cdf_lognorm_shift(x, w, mu, sigma, loc):
    x = np.asarray(x, dtype=float); F = np.zeros_like(x, dtype=float)
    mask = x > loc
    if np.any(mask):
        z = np.log(np.clip(x[mask]-loc, 1e-12, None))
        for wi, m, s in zip(w, mu, sigma):
            F[mask] += wi * stats.norm.cdf((z - m) / s)
    return np.clip(F, 0.0, 1.0)

def pdf_shifted_lognorm_mixture(x, w, mu, sigma, loc):
    x = np.asarray(x, float); f = np.zeros_like(x); mask = x > loc
    if np.any(mask):
        z = np.log(np.clip(x[mask]-loc, 1e-300, None))
        for wi, m, s in zip(w, mu, sigma):
            f[mask] += wi * (1.0/((x[mask]-loc)*s*np.sqrt(2*np.pi))) * np.exp(-0.5*((z-m)/s)**2)
    return f

# --- Inverse CDF (bisección) ---
def inv_cdf_mixture(Ftarget, w, mu, sigma, loc, data_max, tol=1e-6, maxit=100):
    low = float(max(loc + 1e-9, 0.0)); high = float(max(data_max, XLIM_MAX))
    for _ in range(60):
        if mixture_cdf_lognorm_shift([high], w, mu, sigma, loc)[0] >= Ftarget: break
        high *= 1.5
        if high > 1e13: return None
    for _ in range(maxit):
        mid = 0.5*(low+high); Fmid = mixture_cdf_lognorm_shift([mid], w, mu, sigma, loc)[0]
        if abs(Fmid - Ftarget) <= tol*max(1.0, Ftarget): return mid
        if Fmid < Ftarget: low = mid
        else:              high = mid
    return 0.5*(low+high)

def compute_upper_truncation(best, data, tail_mass_cut=TAIL_MASS_CUT):
    Fu = float(1.0 - tail_mass_cut)
    w, mu, sigma, loc = best["w"], best["mu"], best["sigma"], best["loc"]
    u_star = inv_cdf_mixture(Fu, w, mu, sigma, loc, data_max=float(np.nanmax(data)))
    if u_star is None or not np.isfinite(u_star): u_star = float(np.quantile(data, Fu))
    if TRUNC_ROUND_TO and TRUNC_ROUND_TO > 0: u_star = float(np.round(u_star/TRUNC_ROUND_TO)*TRUNC_ROUND_TO)
    return u_star, Fu

# === CDF truncada (opcional) ===
def cdf_truncated(x, w, mu, sigma, loc, u_star):
    Fx = mixture_cdf_lognorm_shift(np.asarray(x), w, mu, sigma, loc)
    Fu = mixture_cdf_lognorm_shift(np.array([u_star], float), w, mu, sigma, loc)[0]
    Fu = max(Fu, 1e-12); z = np.clip(Fx/Fu, 0.0, 1.0); z[np.asarray(x) > u_star] = 1.0
    return z

def sample_shifted_lognorm_mixture(n, loc, w, mu, sigma, random_state=None):
    rng = np.random.default_rng(random_state)
    w = np.asarray(w, float); w = w/w.sum()
    mu = np.asarray(mu, float); sig = np.asarray(sigma, float)
    comp = rng.choice(len(w), size=n, p=w); z = rng.normal(loc=mu[comp], scale=sig[comp])
    return loc + np.exp(z)

def sample_shifted_lognorm_truncated(n, loc, w, mu, sigma, u_star, random_state=None):
    rng = np.random.default_rng(random_state)
    x = sample_shifted_lognorm_mixture(n, loc, w, mu, sigma, random_state=rng)
    trunc_rand(x, upper=u_star, on=True, rng=rng); return x

# ======= Pérdida compuesta con penalización anti-spike =====
def _inv_cdf_safe(q, w, mu, sigma, loc, data_max):
    xq = inv_cdf_mixture(q, w, mu, sigma, loc, data_max=data_max)
    return xq if (xq is not None and np.isfinite(xq)) else None

def composite_cdf_loss(x_s, y_s, w, mu, sigma, loc,
                       w_sse=W_SSE, w_ks=W_KS, w_tail=W_TAIL, tail_gate=TAIL_GATE,
                       quant_pen=QUANT_PEN, lambda_sigma=LAMBDA_SIGMA_PEN):
    y_hat = mixture_cdf_lognorm_shift(x_s, w, mu, sigma, loc)

    sse = np.mean((y_hat - y_s)**2)
    ks = np.max(np.abs(y_hat - y_s))

    tail_mask = (y_s >= tail_gate)
    tail_mse = np.mean((y_hat[tail_mask]-y_s[tail_mask])**2) if np.any(tail_mask) else 0.0

    data_max = float(x_s[-1])
    qpen = 0.0
    for q, wq in quant_pen:
        x_emp = float(np.quantile(x_s, q))
        x_mod = _inv_cdf_safe(q, w, mu, sigma, loc, data_max)
        if x_mod is not None:
            qpen += wq * ((x_mod - x_emp)/max(x_emp,1.0))**2

    # ⬅️ Penalización anti-spike: grande si hay sigmas muy pequeñas
    sigma_pen = np.mean(np.maximum(0.0, ENFORCE_MIN_SIGMA - np.asarray(sigma)))
    penalty = lambda_sigma * sigma_pen

    loss = w_sse*sse + w_ks*ks + w_tail*tail_mse + qpen + penalty
    return loss, (sse, ks, tail_mse, qpen, penalty)

# ======= loc_grid reforzado =======
def make_dynamic_loc_grid(v, n_grid=60, widen=3.0, eps_frac=0.01):
    """
    Grade de 'loc' significativamente por debajo do mínimo dos dados.
    """
    v = np.asarray(v, float)
    vmin, vmax = np.min(v), np.max(v)
    span = max(vmax - vmin, 1.0)
    lo = vmin - widen * span
    hi = vmin - eps_frac * span
    grid = np.linspace(lo, hi, n_grid)
    grid = grid[grid < (vmin - 0.05 * span)]
    return grid

# ======= Fit (k=3, regularizado, anti-spike) =======
def fit_mix_lognorm_shift_ecdf(v, loc_grid=None,
                               random_state=RANDOM_STATE, n_init=GMM_N_INIT):
    """
    Ajusta SIEMPRE una mezcla de 3 lognormales desplazadas (k=3),
    con regularización de covarianza, más reinicios y filtros anti-spike.
    """
    x_s, y_s = ecdf_xy(v)
    loc_grid = make_dynamic_loc_grid(v) if loc_grid is None else loc_grid
    best, records = None, []
    vmin = float(np.min(v))

    for loc in loc_grid:
        u = v - loc
        if np.any(u <= 0):
            continue
        if loc >= 0 or loc > vmin * 0.8:
            continue

        ylog = np.log(u).reshape(-1, 1)
        try:
            gmm = GaussianMixture(
                n_components=3,
                covariance_type="diag",
                random_state=random_state,
                n_init=n_init,
                reg_covar=REG_COVAR     # ⬅️ regulariza las covarianzas
            ).fit(ylog)
        except Exception:
            continue

        w = gmm.weights_.ravel()
        mu = gmm.means_.ravel()
        sigma = np.sqrt(gmm.covariances_.ravel())

        # ⬅️ Filtro anti-spike: descartamos soluciones con sigma demasiado pequeñas
        if np.any(sigma < ENFORCE_MIN_SIGMA):
            continue

        loss, parts = composite_cdf_loss(x_s, y_s, w, mu, sigma, loc)
        rec = dict(model="Mix Lognorm Function", loc=loc, w=w, mu=mu, sigma=sigma,
                   sse=parts[0], ks=parts[1], tail_mse=parts[2], qpen=parts[3],
                   penalty=parts[4], loss=loss, bic=float(gmm.bic(ylog)), k=3)
        records.append(rec)

        if (best is None) or (loss < best["loss"]):
            best = rec

    if not records:
        raise RuntimeError("No valid solution: check units (FLOPs vs GFLOPs) or relax ENFORCE_MIN_SIGMA/REG_COVAR.")

    summary = pd.DataFrame(records).sort_values(["loss","bic"]).reset_index(drop=True)
    print(f"[INFO] Best loc={best['loc']:.2e}, k=3, BIC={best['bic']:.2f}, loss={best['loss']:.3e}, "
          f"KS={best['ks']:.4f}, sigma_min={np.min(best['sigma']):.3f}")
    return (x_s, y_s), summary, best

# ===================== Plots ======================
def plot_histogram_smoothed(v, out_path=OUT_FIG_HIST):
    import matplotlib.patheffects as pe
    ensure_dir(os.path.dirname(out_path) or "figures")
    edges, dens_sm = smoothed_hist(v, bins=HIST_BINS, smooth_window=SMOOTH_WINDOW)
    widths = np.diff(edges); area = float(np.sum(dens_sm * widths))
    print(f"[CHECK] Smoothed-hist area (should ≈ 1): {area:.6f}")

    fig, ax = _figure()
    ax.bar(edges[:-1], dens_sm, width=widths, align='edge', alpha=0.85, edgecolor="none")
    pdf_line, = ax.step(edges, np.r_[dens_sm, dens_sm[-1]], where="post",
                        lw=1.2, ls="-.", color="black", alpha=0.7,
                        label=TEXTS["hist"]["labels"]["pdf"])

    x_ecdf, y_ecdf = ecdf_xy(v)
    ax2 = ax.twinx()
    ecdf_line, = ax2.step(x_ecdf, y_ecdf, where="post",
                          lw=1.4, ls="--", color="black", alpha=0.9, zorder=6,
                          label=TEXTS["hist"]["labels"]["ecdf"])
    ecdf_line.set_path_effects([
        pe.Stroke(linewidth=ecdf_line.get_linewidth() + 1.6, foreground="white"),
        pe.Normal()
    ])

    ax2.set_ylim(0, 1)
    ax2.set_ylabel(TEXTS["common"]["yl_cumprob"], fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])
    _style_yaxis(ax2)
    ax2.set_xlim(*AX_CFG["xlim"])
    ax2.xaxis.set_major_locator(X_MAJOR_LOCATOR)
    ax2.xaxis.set_major_formatter(X_MAJOR_FORMAT)

    ax.set_title(TEXTS["hist"]["title"], fontsize=FONT_CFG["title"], pad=6)
    ax.set_ylabel(TEXTS["hist"]["yl"], fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])
    _style(ax)

    ax.legend([ecdf_line, pdf_line],
              [TEXTS["hist"]["labels"]["ecdf"], TEXTS["hist"]["labels"]["pdf"]],
              loc=LEGEND_CFG["loc"], frameon=LEGEND_CFG["frame"],
              ncol=LEGEND_CFG["ncol"], fontsize=FONT_CFG["legend"])
    fig.savefig(out_path); plt.close(fig)

def plot_ecdf_validation(v, best, n_sim=N_SIM_MIN, out_path=OUT_FIG_ECDF):
    ensure_dir(os.path.dirname(out_path) or "figures")
    x_ecdf, y_ecdf = ecdf_xy(v)
    if USE_TRUNCATED_FOR_VALIDATION and "upper_trunc" in best and np.isfinite(best["upper_trunc"]):
        u_star = best["upper_trunc"]
        sim = sample_shifted_lognorm_truncated(n_sim, best["loc"], best["w"], best["mu"], best["sigma"],
                                               u_star=u_star, random_state=RANDOM_STATE)
        xs, ys = ecdf_xy(sim); xfit = np.linspace(*AX_CFG["xlim"], 1500)
        yfit = cdf_truncated(xfit, best["w"], best["mu"], best["sigma"], best["loc"], u_star)
        model_label = TEXTS["ecdf"]["labels"]["model"] + " (trunc)"
        sim_label = f"{TEXTS['ecdf']['labels']['sim']} (trunc, n={n_sim})"
    else:
        sim = sample_shifted_lognorm_mixture(n_sim, best["loc"], best["w"], best["mu"], best["sigma"],
                                             random_state=RANDOM_STATE)
        xs, ys = ecdf_xy(sim); xfit = np.linspace(*AX_CFG["xlim"], 1500)
        yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])
        model_label = TEXTS["ecdf"]["labels"]["model"]
        sim_label = f"{TEXTS['ecdf']['labels']['sim']} (n={n_sim})"

    fig, ax = _figure()
    ax.step(x_ecdf, y_ecdf, where="post", lw=1.8, label=TEXTS["ecdf"]["labels"]["real"])
    ax.step(xs, ys, where="post", lw=1.8, label=sim_label)
    ax.plot(xfit, yfit, "--", lw=2.0, label=model_label)
    ax.set_title(TEXTS["ecdf"]["title"], fontsize=FONT_CFG["title"], pad=6)
    ax.set_ylabel(TEXTS["ecdf"]["yl"], fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])
    _style(ax); ax.set_ylim(0,1)
    _legend(ax); fig.savefig(out_path); plt.close(fig)

def plot_mixture_hist_with_ecdf(x, comp, bins=HIST_BINS, mode="stacked", show_points=False, out_path=OUT_FIG_MIXED):
    ensure_dir(os.path.dirname(out_path) or "figures")
    colors = ["C0","C1","C2"]; labels = [TEXTS["mixed"]["labels"]["c1"], TEXTS["mixed"]["labels"]["c2"], TEXTS["mixed"]["labels"]["c3"]]
    fig, ax1 = _figure()
    edges = np.histogram_bin_edges(x, bins=bins); widths = np.diff(edges); B = len(widths); lefts = edges[:-1]
    K = int(np.max(comp)) + 1; n = x.size; dens = []
    for k in range(K):
        cnt, _ = np.histogram(x[comp == k], bins=edges); dens.append(cnt / (n * widths))
    dens = np.vstack(dens)

    if mode == "stacked":
        bottom = np.zeros(B)
        for k, (c, lbl) in enumerate(zip(colors, labels)):
            if k >= dens.shape[0]: break
            ax1.bar(lefts, dens[k], width=widths, align="edge", bottom=bottom, color=c, alpha=0.6, edgecolor="none", label=lbl)
            bottom += dens[k]
        total = bottom
    elif mode == "grouped":
        g = len(labels)
        for k, (c, lbl) in enumerate(zip(colors, labels)):
            if k >= dens.shape[0]: break
            ax1.bar(lefts + (k/g)*widths, dens[k], width=widths/g, align="edge", color=c, alpha=0.8, edgecolor="none", label=lbl)
        total = dens.sum(axis=0)
    else:
        raise ValueError("mode must be 'stacked' or 'grouped'.")

    ax1.step(edges, np.r_[total, total[-1]], where="post", lw=1.2, ls="--", color="k", alpha=0.7,
             label=TEXTS["mixed"]["labels"]["pdf"])
    ax1.set_title(TEXTS["mixed"]["title"], fontsize=FONT_CFG["title"], pad=6)
    ax1.set_ylabel(TEXTS["mixed"]["yl"], fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])
    _style(ax1)

    ax2 = ax1.twinx()
    xs_all = np.sort(x); ys_all = np.arange(1, x.size + 1) / x.size
    ax2.step(xs_all, ys_all, where="post", lw=3.0, color="1.0", alpha=1.0)
    ax2.step(xs_all, ys_all, where="post", lw=1.3, ls=(0,(4,2)), color="0.15", alpha=1.0, label=TEXTS["mixed"]["labels"]["cdf"])
    if show_points:
        order = np.argsort(x); comp_sorted = comp[order]
        idx = np.linspace(0, x.size - 1, min(400, x.size), dtype=int)
        ax2.scatter(xs_all[idx], ys_all[idx], s=14, c=np.take(colors, comp_sorted[idx], mode='wrap'),
                    alpha=0.9, edgecolors="white", linewidths=0.4)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel(TEXTS["common"]["yl_cumprob"], fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])
    _style_xaxis(ax2); _style_yaxis(ax2)

    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + ([h2[0]] if h2 else []), l1 + ([l2[0]] if l2 else []),
               loc=LEGEND_CFG["loc"], frameon=LEGEND_CFG["frame"],
               ncol=LEGEND_CFG["ncol"], fontsize=FONT_CFG["legend"])
    fig.savefig(out_path); plt.close(fig)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import numpy as np
import os

def plot_ecdf_multi_full(all_curves, out_path):
    """
    ECDF real, simulada e modelo para múltiplos experimentos,
    com escala log, legendas refinadas e paleta harmônica.
    """
    ensure_dir(os.path.dirname(out_path) or "figures")

    # Paleta harmônica (Set2) – 8 cores suaves
    colors = plt.get_cmap("Set2").colors

    # Figura mais larga para dar espaço às legendas
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=300)

    color_handles = []
    seen_labels = set()

    # Mapeia nomes simplificados
    name_map = {
        "fedavg_c_50_e_1": "Batch 100%",
        "minibatch_c_20_mb_0.9": "Batch 90%",
        "minibatch_c_20_mb_0.8": "Batch 80%",
        "minibatch_c_20_mb_0.6": "Batch 60%",
        "minibatch_c_20_mb_0.5": "Batch 50%",
        "minibatch_c_20_mb_0.4": "Batch 40%",
        "minibatch_c_20_mb_0.2": "Batch 20%",
    }

    # --- Curvas principais ---
    for i, cur in enumerate(all_curves):
        lbl = name_map.get(cur["label"], cur["label"])
        color = colors[i % len(colors)]

        ax.step(cur["x_real"], cur["y_real"], where="post",
                lw=2.0, color=color, alpha=0.9)
        ax.step(cur["x_sim"], cur["y_sim"], where="post",
                lw=1.6, ls="--", color=color, alpha=0.9)
        ax.plot(cur["x_model"], cur["y_model"],
                lw=1.8, ls=":", color=color, alpha=0.9)

        if lbl not in seen_labels:
            color_handles.append(Line2D([0], [0], color=color, lw=2.8, label=lbl))
            seen_labels.add(lbl)

    # --- Eixos e estilo ---
    # ax.set_xscale("log")
    ax.set_xlabel("GFLOPs", fontsize=15, labelpad=6)
    ax.set_ylabel("Clients (%)", fontsize=15, labelpad=6)
    ax.tick_params(axis="both", labelsize=13, length=5, width=1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _style_xaxis(ax)

    # --- Legenda 1: tipo de curva (estilos de linha) ---
    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="Real (ECDF)"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="Sim (ECDF)"),
        Line2D([0], [0], color="black", lw=2.0, linestyle=":", label="Model (CDF)"),
    ]
    leg_style = ax.legend(
        handles=style_handles,
        title="Curve Type",
        title_fontsize=13,
        fontsize=12,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.5),  # fora do quadro, empilhada
        frameon=True,
        fancybox=True,
        edgecolor="0.6",
        ncol=1,
    )
    ax.add_artist(leg_style)

    # --- Legenda 2: cores / configuração ---
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

    # --- Margens para espaço confortável ---
    plt.subplots_adjust(left=0.08, right=0.78, top=0.93, bottom=0.12)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



# ===================== Main ======================
def main():
    file_bases = [
        "sys_metrics_fedavg_c_50_e_1",
        "sys_metrics_minibatch_c_20_mb_0.9",
        "sys_metrics_minibatch_c_20_mb_0.8",
        "sys_metrics_minibatch_c_20_mb_0.6",
        "sys_metrics_minibatch_c_20_mb_0.5",
        "sys_metrics_minibatch_c_20_mb_0.4",
        "sys_metrics_minibatch_c_20_mb_0.2",
    ]

    base_results_dir = "../results/sys"
    ensure_dir("figures"); ensure_dir("params")

    all_curves = []

    for base in file_bases:
        print("\n" + "="*80)
        print(f"Processing dataset: {base}")
        print("="*80)

        sys_file = os.path.join(base_results_dir, f"{base}.csv")
        tag = base.replace("sys_metrics_", "")
        out_prefix = os.path.join("figures/fit", tag)
        ensure_dir(os.path.dirname(out_prefix))
        out_hist = f"{out_prefix}_hist_smoothed.png"
        out_ecdf = f"{out_prefix}_ecdf_validation.png"
        out_mix  = f"{out_prefix}_mix_hist_ecdf.png"
        out_txt  = f"params/{tag}_mix_lognorm_shift_params.txt"

        df = load_data(sys_file)
        v = df["FLOPs"].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0)]
        print(f"Loaded {len(v)} valid FLOPs values from {sys_file}")

        (x_ecdf, y_ecdf), summary, best = fit_mix_lognorm_shift_ecdf(
            v, loc_grid=None, random_state=RANDOM_STATE, n_init=GMM_N_INIT
        )
        best_used = dict(best); best_used["calib_ok"] = False

        upper_trunc, Fu = compute_upper_truncation(best_used, v, tail_mass_cut=TAIL_MASS_CUT)
        best_used["upper_trunc"] = upper_trunc; best_used["Fu"] = Fu

        x_real, y_real = ecdf_xy(v)

        x_sim = sample_shifted_lognorm_mixture(N_SIM_MIN, best_used["loc"], best_used["w"],
                                               best_used["mu"], best_used["sigma"],
                                               random_state=RANDOM_STATE)
        x_sim.sort(); y_sim = np.arange(1, len(x_sim)+1)/len(x_sim)

        x_model = np.linspace(*AX_CFG["xlim"], 1500)
        y_model = mixture_cdf_lognorm_shift(x_model, best_used["w"], best_used["mu"],
                                            best_used["sigma"], best_used["loc"])

        plot_histogram_smoothed(v, out_path=out_hist)
        plot_ecdf_validation(v, best_used, n_sim=N_SIM_MIN, out_path=out_ecdf)

        rng_lbl = np.random.default_rng(RANDOM_STATE)
        comp_sim = rng_lbl.choice(len(best_used["w"]), size=x_sim.size, p=np.array(best_used["w"])/np.sum(best_used["w"]))
        plot_mixture_hist_with_ecdf(x_sim, comp_sim, bins=HIST_BINS, mode="stacked",
                                    show_points=False, out_path=out_mix)

        # Guardar parámetros + métricas
        # Fm_ecdf = mixture_cdf_lognorm_shift(x_real, best_used["w"], best_used["mu"], best_used["sigma"], best_used["loc"])
        # KS = float(np.max(np.abs(Fm_ecdf - y_real)))
        
        if USE_TRUNCATED_FOR_VALIDATION and "upper_trunc" in best_used and np.isfinite(best_used["upper_trunc"]):
            Fm_ecdf = cdf_truncated(
                x_real,
                best_used["w"],
                best_used["mu"],
                best_used["sigma"],
                best_used["loc"],
                best_used["upper_trunc"]
            )
        else:
            Fm_ecdf = mixture_cdf_lognorm_shift(
                x_real,
                best_used["w"],
                best_used["mu"],
                best_used["sigma"],
                best_used["loc"]
            )

        KS = float(np.max(np.abs(Fm_ecdf - y_real)))
        SSE_sum = float(np.sum((Fm_ecdf - y_real)**2))
        
        
        SSE_sum = float(np.sum((Fm_ecdf - y_real)**2))

        with open(out_txt, "w") as f:
            f.write(f"Best: {best_used['model']}  loc={best_used['loc']}\n")
            for i, (wi, mi, si) in enumerate(zip(best_used["w"], best_used["mu"], best_used["sigma"]), 1):
                scale_i = float(np.exp(mi))
                f.write(f" comp{i}: w={wi:.6f}, mu_log={mi:.6f}, sigma={si:.6f}, scale=exp(mu)={scale_i:.6f}\n")
            f.write(f"upper_trunc={best_used['upper_trunc']:.6f}  Fu={best_used['Fu']:.6f}  tail_mass_cut={TAIL_MASS_CUT:.6f}\n")
            f.write(f"BIC={best_used['bic']:.2f}\n")
            f.write(f"KS_ECDF={KS:.6f}\n")
            f.write(f"SSE={SSE_sum:.6f}\n")
            f.write(f"anti_spike_penalty={best_used['penalty']:.6e}\n")

        print(f"✅ Finished {tag} | KS={KS:.4f}, SSE={SSE_sum:.2f}")

        all_curves.append({
            "label": tag,
            "x_real": x_real, "y_real": y_real,
            "x_sim": x_sim, "y_sim": y_sim,
            "x_model": x_model, "y_model": y_model
        })

    out_all = "figures/fit/ALL_ecdf_real_sim_model.png"
    plot_ecdf_multi_full(all_curves, out_path=out_all)
    print(f"\n📊 Combined ECDF figure saved to: {out_all}")

if __name__ == "__main__":
    main()
