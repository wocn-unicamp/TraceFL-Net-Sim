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

# ---- Eje X: 1e9 .. 5e9, paso 0.5e9; mostrar etiquetas solo en enteros
AX_CFG = {
    "xlim":  (1.0e9, 5.0e9),
    "xticks": np.arange(1.0e9, 5.0e9 + 1.0, 0.5e9),
}

def _x_tick_formatter(val, pos):
    g = val / 1e9  # en miles de millones
    # si está "casi" en un entero => etiqueta; si es .5 => vacío
    if abs(g - round(g)) < 1e-9:
        return f"{int(round(g))}"
    return ""  # oculta los .5

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
        "labels": {
            "real":"Empirical ECDF",
            "sim":"Simulated ECDF",
            "model":"Model CDF"
            # "model_trunc":"Model CDF (truncated)"
        }
    },
    "mixed": {
        "title": " ",
        "yl":"Estimated PDF",
        "labels":{"pdf":"Model PDF","cdf":"Model CDF","c1":"Component 1","c2":"Component 2","c3":"Component 3"}
    }
}

def _figure():
    return plt.subplots(figsize=FIG_CFG["size"], dpi=FIG_CFG["dpi"],
                        layout="constrained" if FIG_CFG["tight"] else None)

def _style_xaxis(ax):
    # límites + ticks + formateador personalizado
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
USE_TRUNCATED_FOR_VALIDATION = False
W_SSE, W_KS, W_TAIL, TAIL_GATE = 1.0, 0.6, 0.2, 0.95
QUANT_PEN = [(0.50, 0.2), (0.90, 0.6)]
Q_TARGET  = 0.995

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

# ======= Pérdida compuesta y loc_grid dinámico =======
def _inv_cdf_safe(q, w, mu, sigma, loc, data_max):
    xq = inv_cdf_mixture(q, w, mu, sigma, loc, data_max=data_max)
    return xq if (xq is not None and np.isfinite(xq)) else None

def composite_cdf_loss(x_s, y_s, w, mu, sigma, loc,
                       w_sse=W_SSE, w_ks=W_KS, w_tail=W_TAIL, tail_gate=TAIL_GATE,
                       quant_pen=QUANT_PEN):
    y_hat = mixture_cdf_lognorm_shift(x_s, w, mu, sigma, loc)
    sse = np.mean((y_hat - y_s)**2); ks = np.max(np.abs(y_hat - y_s))
    tail_mask = (y_s >= tail_gate); tail_mse = np.mean((y_hat[tail_mask]-y_s[tail_mask])**2) if np.any(tail_mask) else 0.0
    data_max = float(x_s[-1]); qpen = 0.0
    for q, wq in quant_pen:
        x_emp = float(np.quantile(x_s, q)); x_mod = _inv_cdf_safe(q, w, mu, sigma, loc, data_max)
        if x_mod is not None: qpen += wq * ((x_mod - x_emp)/max(x_emp,1.0))**2
    return w_sse*sse + w_ks*ks + w_tail*tail_mse + qpen, (sse, ks, tail_mse, qpen)

def make_dynamic_loc_grid(v, n_grid=40, widen=2.0, eps_frac=0.02):
    q05, q95 = np.quantile(v, [0.05, 0.95]); span = max(q95 - q05, 1.0)
    vmin = float(v.min()); lo = vmin - widen*span; hi = vmin - eps_frac*span
    grid = np.linspace(lo, hi, n_grid); return np.minimum(grid, vmin - 1e-9)

# ======= Fit =======
def fit_mix_lognorm_shift_ecdf(v, k_range=(1,2,3), loc_grid=None,
                               random_state=RANDOM_STATE, n_init=5):
    x_s, y_s = ecdf_xy(v); loc_grid = make_dynamic_loc_grid(v) if loc_grid is None else loc_grid
    best, records = None, []
    for loc in loc_grid:
        u = v - loc
        if np.any(u <= 0): continue
        ylog = np.log(u).reshape(-1,1)
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                  random_state=random_state, n_init=n_init).fit(ylog)
            w = gmm.weights_.ravel(); mu = gmm.means_.ravel(); sigma = np.sqrt(gmm.covariances_.ravel())
            loss, parts = composite_cdf_loss(x_s, y_s, w, mu, sigma, loc)
            rec = dict(model="Mix Lognorm Function", loc=loc, w=w, mu=mu, sigma=sigma,
                       sse=parts[0], ks=parts[1], tail_mse=parts[2], qpen=parts[3],
                       loss=loss, bic=float(gmm.bic(ylog)), k=k)
            records.append(rec)
            if (best is None) or (loss < best["loss"]): best = rec
    summary = pd.DataFrame(records).sort_values(["loss","bic"]).reset_index(drop=True)
    return (x_s, y_s), summary, best

# ======= Calibración opcional =======
def calibrate_quantile_mu_shift(best, x_target, q_target=Q_TARGET,
                                delta_min=-3.0, delta_max=3.0, tol=1e-8, maxit=80):
    w, mu, sigma, loc = best["w"], best["mu"].copy(), best["sigma"], best["loc"]
    def F_with_delta(delta): return mixture_cdf_lognorm_shift(np.array([x_target], float), w, mu + delta, sigma, loc)[0]
    f_lo, f_hi = F_with_delta(delta_min), F_with_delta(delta_max)
    g_lo, g_hi = f_lo - q_target, f_hi - q_target; expand = 0; dmin, dmax = delta_min, delta_max
    while g_lo * g_hi > 0 and expand < 5:
        expand += 1; dmin -= 2.0; dmax += 2.0
        f_lo, f_hi = F_with_delta(dmin), F_with_delta(dmax); g_lo, g_hi = f_lo - q_target, f_hi - q_target
    if g_lo * g_hi > 0: return None, dict(ok=False, msg="No bracket")
    for _ in range(maxit):
        mid = 0.5*(dmin+dmax); fm = F_with_delta(mid) - q_target
        if abs(fm) <= tol*max(1.0, q_target): return mu + mid, dict(ok=True, delta=mid, F_at_target=fm + q_target)
        if fm > 0: dmin = mid
        else:      dmax = mid
    mid = 0.5*(dmin+dmax); return mu + mid, dict(ok=True, delta=mid, F_at_target=F_with_delta(mid))

# ===================== Plots (rango X fijo + labels alternas) ======================
def plot_histogram_smoothed(v, out_path=OUT_FIG_HIST):
    ensure_dir(os.path.dirname(out_path) or "figures")

    # ---- Convertir a GFLOPs ----
    v_g = np.asarray(v, float) * 1e-9

    # Histograma suavizado en GFLOPs (densidad) y convertir a fracción por bin
    edges, dens_sm = smoothed_hist(v_g, bins=HIST_BINS, smooth_window=SMOOTH_WINDOW)  # densidad
    widths = np.diff(edges)
    prob_sm = dens_sm * widths  # -> fracción por bin
    area = float(np.sum(prob_sm))  # ~ 1.0
    print(f"[CHECK] Smoothed-hist mass (should ≈ 1): {area:.6f}")

    # Eje primario (izq) = CDF ; Eje secundario (der) = Histograma (fracción)
    fig, ax_cdf = _figure()
    ax_pdf = ax_cdf.twinx()

    # --- PDF (derecha): barras (fracción por bin) + línea escalón (fracción) ---
    ax_pdf.bar(edges[:-1], prob_sm, width=widths, align='edge', alpha=0.85, edgecolor="none")
    pdf_line, = ax_pdf.step(
        edges, np.r_[prob_sm, prob_sm[-1]],
        where="post",
        lw=2,
        ls=(0, (1, 1)),     # punteado distinto de la CDF
        color="black",
        alpha=0.9,
        zorder=15,          # por encima de todo
        label=TEXTS["hist"]["labels"]["pdf"]
    )

    # --- CDF (izquierda): ECDF en porcentaje (stroke blanco + punteada) ---
    x_ecdf, y_ecdf = ecdf_xy(v_g)
    ax_cdf.step(x_ecdf, y_ecdf, where="post", lw=3.0, color="1.0", alpha=1.0, zorder=3)  # stroke blanco
    ecdf_line, = ax_cdf.step(
        x_ecdf, y_ecdf, where="post",
        lw=2,
        ls=(0, (4, 2)),     # punteado distinto de la PDF
        color="0.15",
        alpha=1.0,
        zorder=6,
        label=TEXTS["hist"]["labels"]["ecdf"]
    )

    # --- Etiquetas de ejes y estilo ---
    ax_cdf.set_title(TEXTS["hist"]["title"], fontsize=FONT_CFG["title"], pad=6)
    ax_cdf.set_xlabel("Computational demand (GFLOPs)", fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["xpad"])

    ax_cdf.set_ylim(0, 1.0)
    ax_cdf.set_ylabel("CDF Clients (%)", fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])
    ax_cdf.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y*100))}"))

    # Eje derecho ahora es fracción (no densidad)
    ax_pdf.set_ylabel("Clients (Histogram)", fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])

    # Límites y ticks de X: 1..5 GFLOPs con enteros
    ax_cdf.set_xlim(1.0, 5.0)
    ax_pdf.set_xlim(1.0, 5.0)
    ax_cdf.xaxis.set_major_locator(FixedLocator(np.arange(1.0, 5.0 + 1e-9, 1.0)))

    # Estética: grid y tamaños de tick (X=13, Y=15)
    ax_cdf.grid(True, alpha=FIG_CFG["grid_alpha"])
    ax_cdf.tick_params(axis="x", labelsize=18, length=TICK_CFG["len"], width=TICK_CFG["width"])
    ax_cdf.tick_params(axis="y", labelsize=18, length=TICK_CFG["len"], width=TICK_CFG["width"])

    ax_pdf.tick_params(axis="y", labelsize=18, length=TICK_CFG["len"], width=TICK_CFG["width"])

    # --- Leyenda a la derecha pero más cerca del eje (no tan afuera) ---
    ax_cdf.legend(
        [ecdf_line, pdf_line],
        [TEXTS["hist"]["labels"]["ecdf"], TEXTS["hist"]["labels"]["pdf"]],
        loc="center right",          # centrada verticalmente en el lado derecho
        # bbox_to_anchor=(1.0, 0.5),  # pegada al borde; más a la izquierda que 1.02
        frameon=LEGEND_CFG["frame"],
        ncol=LEGEND_CFG["ncol"],
        fontsize=FONT_CFG["legend"]-4,
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf_validation(v, best, n_sim=N_SIM_MIN, out_path=OUT_FIG_ECDF):
    ensure_dir(os.path.dirname(out_path) or "figures")
    x_ecdf, y_ecdf = ecdf_xy(v)
    if USE_TRUNCATED_FOR_VALIDATION and "upper_trunc" in best and np.isfinite(best["upper_trunc"]):
        u_star = best["upper_trunc"]
        sim = sample_shifted_lognorm_truncated(n_sim, best["loc"], best["w"], best["mu"], best["sigma"], u_star=u_star, random_state=RANDOM_STATE)
        xs, ys = ecdf_xy(sim); xfit = np.linspace(*AX_CFG["xlim"], 1500)
        yfit = cdf_truncated(xfit, best["w"], best["mu"], best["sigma"], best["loc"], u_star)
        model_label = TEXTS["ecdf"]["labels"]["model_trunc"]; sim_label = f"{TEXTS['ecdf']['labels']['sim']} (trunc, n={n_sim})"
    else:
        sim = sample_shifted_lognorm_mixture(n_sim, best["loc"], best["w"], best["mu"], best["sigma"], random_state=RANDOM_STATE)
        xs, ys = ecdf_xy(sim); xfit = np.linspace(*AX_CFG["xlim"], 1500)
        yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])
        model_label = TEXTS["ecdf"]["labels"]["model"]; sim_label = f"{TEXTS['ecdf']['labels']['sim']} (n={n_sim})"

    fig, ax = _figure()
    ax.step(x_ecdf, y_ecdf, where="post", lw=1.8, label=TEXTS["ecdf"]["labels"]["real"])
    ax.step(xs, ys, where="post", lw=1.8, label=sim_label)
    ax.plot(xfit, yfit, "--", lw=2.0, label=model_label)

    # if "upper_trunc" in best and np.isfinite(best["upper_trunc"]):
        # ax.axvline(best["upper_trunc"], ls=":", lw=1.6, color="crimson", alpha=0.8,
                #    label=TEXTS["common"]["upper_mark_fmt"].format(Fu=best.get("Fu", np.nan)))

    ax.set_title(TEXTS["ecdf"]["title"], fontsize=FONT_CFG["title"], pad=6)
    ax.set_ylabel(TEXTS["ecdf"]["yl"], fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])
    _style(ax); ax.set_ylim(0,1)
    _legend(ax); fig.savefig(out_path); plt.close(fig)

def plot_mixture_hist_with_ecdf(x, comp, bins=HIST_BINS, mode="stacked",
                                show_points=False, out_path=OUT_FIG_MIXED):
    ensure_dir(os.path.dirname(out_path) or "figures")

    colors = ["C0","C1","C2"]
    labels = [TEXTS["mixed"]["labels"]["c1"],
              TEXTS["mixed"]["labels"]["c2"],
              TEXTS["mixed"]["labels"]["c3"]]

    fig, ax_cdf = _figure()
    ax_pdf = ax_cdf.twinx()

    x_g = np.asarray(x, float) * 1e-9
    edges = np.histogram_bin_edges(x_g, bins=bins)
    widths = np.diff(edges); B = len(widths); lefts = edges[:-1]

    K = int(np.max(comp)) + 1
    n = float(x_g.size)
    dens = []
    for k in range(K):
        cnt, _ = np.histogram(x_g[comp == k], bins=edges)
        dens.append(cnt / n)  # <-- fracción por bin (NO densidad)
    dens = np.vstack(dens) if len(dens) > 0 else np.zeros((0, B))

    bar_handles = []
    if mode == "stacked":
        bottom = np.zeros(B)
        for k, (c, lbl) in enumerate(zip(colors, labels)):
            if k >= dens.shape[0]:
                break
            h = ax_pdf.bar(lefts, dens[k], width=widths, align="edge",
                           bottom=bottom, color=c, alpha=0.6, edgecolor="none", label=lbl)
            bar_handles.append(h[0])
            bottom += dens[k]
        total = bottom
    elif mode == "grouped":
        g = len(labels)
        for k, (c, lbl) in enumerate(zip(colors, labels)):
            if k >= dens.shape[0]:
                break
            h = ax_pdf.bar(lefts + (k/g)*widths, dens[k], width=widths/g,
                           align="edge", color=c, alpha=0.8, edgecolor="none", label=lbl)
            bar_handles.append(h[0])
        total = dens.sum(axis=0)
    else:
        raise ValueError("mode must be 'stacked' or 'grouped'.")

    pdf_line, = ax_pdf.step(edges, np.r_[total, total[-1]], where="post",
                            lw=2.0, ls=(0, (1, 1)), color="k", alpha=0.7,
                            label=TEXTS["mixed"]["labels"]["pdf"])

    xs_all = np.sort(x_g); ys_all = np.arange(1, x_g.size + 1) / x_g.size
    ax_cdf.step(xs_all, ys_all, where="post", lw=3.0, color="1.0", alpha=1.0)
    cdf_line, = ax_cdf.step(xs_all, ys_all, where="post",
                            lw=2.0, ls=(0,(4,2)), color="0.15", alpha=1.0,
                            label=TEXTS["mixed"]["labels"]["cdf"])

    if show_points and x_g.size > 0:
        order = np.argsort(x_g); comp_sorted = comp[order]
        idx = np.linspace(0, x_g.size - 1, min(400, x_g.size), dtype=int)
        ax_cdf.scatter(xs_all[idx], ys_all[idx], s=14,
                       c=np.take(colors, comp_sorted[idx], mode='wrap'),
                       alpha=0.9, edgecolors="white", linewidths=0.4)

    ax_cdf.set_title(TEXTS["mixed"]["title"], fontsize=FONT_CFG["title"], pad=6)
    ax_cdf.set_xlabel("Computational demand (GFLOPs)",
                      fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["xpad"])
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_ylabel("CDF Clients (%)",
                      fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])
    ax_cdf.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y*100))}"))

    ax_pdf.set_ylabel("Clients (Histogram)",
                      fontsize=FONT_CFG["axis"], labelpad=TICK_CFG["ypad"])

    ax_cdf.grid(True, alpha=FIG_CFG["grid_alpha"])
    ax_cdf.tick_params(axis="x", labelsize=FONT_CFG["ticks"],
                       length=TICK_CFG["len"], width=TICK_CFG["width"])
    ax_cdf.tick_params(axis="y", labelsize=FONT_CFG["ticks"],
                       length=TICK_CFG["len"], width=TICK_CFG["width"])
    ax_pdf.tick_params(axis="y", labelsize=18,
                   length=TICK_CFG["len"], width=TICK_CFG["width"])


    ax_cdf.set_xlim(1.0, 5.0)
    ax_pdf.set_xlim(1.0, 5.0)
    ax_cdf.xaxis.set_major_locator(FixedLocator(np.arange(0, 5.0 + 1e-9, 1.0)))

    h1, l1 = ax_pdf.get_legend_handles_labels()
    h2, l2 = ax_cdf.get_legend_handles_labels()

    legend_font = max(10, int(FONT_CFG["legend"] * 0.75))
    ax_cdf.legend(
        h1 + ([h2[0]] if h2 else []),
        l1 + ([l2[0]] if l2 else []),
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        frameon=LEGEND_CFG["frame"],
        ncol=LEGEND_CFG["ncol"],
        fontsize=legend_font,
    )

    fig.savefig(out_path); plt.close(fig)



# ===================== Main ======================
def main():
    ensure_dir("figures"); ensure_dir(os.path.dirname(OUT_TXT) or ".")
    df = load_data(SYS_FILE)
    v = df["FLOPs"].to_numpy(float); v = v[np.isfinite(v) & (v > 0)]

    (x_ecdf, y_ecdf), summary, best = fit_mix_lognorm_shift_ecdf(v, k_range=(1,2,3), loc_grid=None,
                                                                 random_state=RANDOM_STATE, n_init=5)

    if CALIBRATE_QUANTILE:
        x_target = float(np.quantile(v, Q_TARGET))
        mu_cal, info = calibrate_quantile_mu_shift(best, x_target, q_target=Q_TARGET)
        if (mu_cal is not None) and info.get("ok", False):
            best_used = dict(best); best_used["mu"] = mu_cal
            best_used["calib_ok"] = True; best_used["delta_mu"] = info["delta"]; best_used["F_at_target_post"] = info["F_at_target"]
        else:
            best_used = dict(best); best_used["calib_ok"] = False
    else:
        best_used = dict(best); best_used["calib_ok"] = False

    upper_trunc, Fu = compute_upper_truncation(best_used, v, tail_mass_cut=TAIL_MASS_CUT)
    best_used["upper_trunc"] = upper_trunc; best_used["Fu"] = Fu

    x_s, y_s = ecdf_xy(v)
    Fm_ecdf = mixture_cdf_lognorm_shift(x_s, best_used["w"], best_used["mu"], best_used["sigma"], best_used["loc"])
    ks_ecdf = float(np.max(np.abs(Fm_ecdf - y_s)))

    print("\n=== Best candidates (sorted by LOSS) ===")
    print(summary.head(10).to_string(index=False))

    SSE_sum = float(np.sum((Fm_ecdf - y_s)**2))
    with open(OUT_TXT, "w") as f:
        f.write(f"Best: {best_used['model']}  loc={best_used['loc']}\n")
        for i, (wi, mi, si) in enumerate(zip(best_used["w"], best_used["mu"], best_used["sigma"]), 1):
            scale_i = float(np.exp(mi))
            f.write(f" comp{i}: w={wi:.6f}, mu_log={mi:.6f}, sigma={si:.6f}, scale=exp(mu)={scale_i:.6f}\n")
        f.write(f"upper_trunc={best_used['upper_trunc']:.6f}  Fu={best_used['Fu']:.6f}  tail_mass_cut={TAIL_MASS_CUT:.6f}\n")
        f.write(f"SSE={SSE_sum:.6f}, BIC={best_used['bic']:.2f}\n")
        f.write(f"KS_ECDF={ks_ecdf:.6f}\n")

    print(f"\nParameters saved to: {OUT_TXT}")
    print(f"upper_trunc={upper_trunc:.1f}, Fu={Fu:.6f}, tail_mass_cut={TAIL_MASS_CUT:.6f}")

    plot_histogram_smoothed(v, out_path=OUT_FIG_HIST)
    print(f"Histogram figure saved to: {OUT_FIG_HIST}")

    plot_ecdf_validation(v, best_used, n_sim=N_SIM_MIN, out_path=OUT_FIG_ECDF)
    print(f"ECDF validation figure saved to: {OUT_FIG_ECDF}")

    x_sim = sample_shifted_lognorm_mixture(N_SIM_MIN, best_used["loc"], best_used["w"], best_used["mu"], best_used["sigma"], random_state=RANDOM_STATE)
    rng_lbl = np.random.default_rng(RANDOM_STATE)
    comp_sim = rng_lbl.choice(len(best_used["w"]), size=x_sim.size, p=np.array(best_used["w"])/np.sum(best_used["w"]))
    plot_mixture_hist_with_ecdf(x_sim, comp_sim, bins=HIST_BINS, mode="stacked", show_points=False, out_path=OUT_FIG_MIXED)
    print(f"Mixture histogram + ECDF figure saved to: {OUT_FIG_MIXED}")

    print("\n=== Summary statistics ===")
    print(f"Real:      mean={np.mean(v):.1f}, median={np.median(v):.1f}, std={np.std(v, ddof=1):.1f}")
    print(f"Simulated: mean={np.mean(x_sim):.1f}, median={np.median(x_sim):.1f}, std={np.std(x_sim, ddof=1):.1f}")
    print(f"(n_real={len(v)}, n_sim={N_SIM_MIN})")
    print(f"KS_ECDF_used={ks_ecdf:.6f}")

if __name__ == "__main__":
    main()

