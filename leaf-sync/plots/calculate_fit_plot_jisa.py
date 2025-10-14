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

RANDOM_STATE   = 42
N_SIM_MIN      = 1500            # ECDF simulada razonablemente suave
X_LABEL        = "FLOPs"
HIST_BINS      = 64
SMOOTH_WINDOW  = 7
XLIM_MAX       = 5e9

# --- Truncamiento superior (1 - Fu): se calcula, pero por defecto NO se usa para renormalizar en las figuras ---
TAIL_MASS_CUT  = 0.016           # Fu = 0.984
TRUNC_ROUND_TO = None            # e.g., 100.0 para redondear u*

FIG_SIZE = (8, 4)
DPI = 300

# ==================== Estrictitud de cola (relajada por defecto) ====================
# Puedes intensificarlo cambiando estos flags a True.
CALIBRATE_QUANTILE          = False   # Si True, ajusta Δμ para clavar q_target (ver Q_TARGET)
USE_TRUNCATED_FOR_VALIDATION= False   # Si True, CDF y simulación se renormalizan en u* en la figura de validación

# Pérdida compuesta (relajada):
W_SSE     = 1.0     # peso SSE global (CDF vs ECDF)
W_KS      = 0.6     # peso KS
W_TAIL    = 0.2     # MSE en cola (bajo, para no forzar)
TAIL_GATE = 0.95    # definición de "cola" en ECDF
# Penalización de cuantiles (más suave, sin 0.99):
QUANT_PEN = [(0.50, 0.2), (0.90, 0.6)]

# Calibración opcional de cuantil (si CALIBRATE_QUANTILE=True):
Q_TARGET  = 0.995   # usar 0.999 si tienes muchos datos y quieres apretar cola


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
    dens, edges = np.histogram(vals, bins=bins, density=True)
    if smooth_window and smooth_window > 1:
        k = int(smooth_window); kernel = np.ones(k)/k
        dens = np.convolve(dens, kernel, mode="same")
    widths = np.diff(edges)
    area = float(np.sum(dens * widths))
    if area > 0:
        dens = dens / area
    return edges, dens


def trunc_rand(flops, upper, on=True, rng=None):
    """
    Reemplaza in-place todo valor > upper por un uniforme en (0.8*upper, upper).
    `upper` puede ser escalar o array broadcastable contra `flops`.
    """
    if not (on and upper is not None):
        return flops
    rng = np.random.default_rng() if rng is None else rng
    u = np.asarray(upper)
    m = flops > u
    if not np.any(m):
        return flops
    # Intervalo abierto por abajo, semiabierto por arriba (como uniform de NumPy)
    eps = np.finfo(float).eps
    if np.ndim(u) == 0:
        lo = 0.8 * float(u) * (1.0 + eps)
        hi = float(u)
        flops[m] = rng.uniform(lo, hi, m.sum())
    else:
        lo = 0.8 * u[m] * (1.0 + eps)
        hi = u[m]
        flops[m] = rng.uniform(lo, hi)
    return flops

# ======= Mixture of shifted lognormals (CDF/PDF) =======
def mixture_cdf_lognorm_shift(x, w, mu, sigma, loc):
    x = np.asarray(x, dtype=float)
    F = np.zeros_like(x, dtype=float)
    mask = x > loc
    if np.any(mask):
        z = np.log(np.clip(x[mask] - loc, 1e-12, None))
        for wi, m, s in zip(w, mu, sigma):
            F[mask] += wi * stats.norm.cdf((z - m) / s)
    return np.clip(F, 0.0, 1.0)

def pdf_shifted_lognorm_mixture(x, w, mu, sigma, loc):
    x = np.asarray(x, float)
    f = np.zeros_like(x)
    mask = x > loc
    if np.any(mask):
        z = np.log(np.clip(x[mask]-loc, 1e-300, None))
        for wi, m, s in zip(w, mu, sigma):
            f[mask] += wi * (1.0/((x[mask]-loc)*s*np.sqrt(2*np.pi))) * np.exp(-0.5*((z-m)/s)**2)
    return f

# --- Inverse CDF via bisección ---
def inv_cdf_mixture(Ftarget, w, mu, sigma, loc, data_max, tol=1e-6, maxit=100):
    low = float(max(loc + 1e-9, 0.0))
    high = float(max(data_max, XLIM_MAX))
    for _ in range(60):
        if mixture_cdf_lognorm_shift([high], w, mu, sigma, loc)[0] >= Ftarget:
            break
        high *= 1.5
        if high > 1e13:
            return None
    for _ in range(maxit):
        mid = 0.5 * (low + high)
        Fmid = mixture_cdf_lognorm_shift([mid], w, mu, sigma, loc)[0]
        if abs(Fmid - Ftarget) <= tol * max(1.0, Ftarget):
            return mid
        if Fmid < Ftarget: low = mid
        else:               high = mid
    return 0.5 * (low + high)

def compute_upper_truncation(best, data, tail_mass_cut=TAIL_MASS_CUT):
    Fu = float(1.0 - tail_mass_cut)
    w, mu, sigma, loc = best["w"], best["mu"], best["sigma"], best["loc"]
    u_star = inv_cdf_mixture(Fu, w, mu, sigma, loc, data_max=float(np.nanmax(data)))
    if u_star is None or not np.isfinite(u_star):
        u_star = float(np.quantile(data, Fu))
    if TRUNC_ROUND_TO and TRUNC_ROUND_TO > 0:
        u_star = float(np.round(u_star / TRUNC_ROUND_TO) * TRUNC_ROUND_TO)
    return u_star, Fu


# === Versión truncada (solo para uso opcional en validación) ===
def cdf_truncated(x, w, mu, sigma, loc, u_star):
    Fx = mixture_cdf_lognorm_shift(np.asarray(x), w, mu, sigma, loc)
    Fu = mixture_cdf_lognorm_shift(np.array([u_star], float), w, mu, sigma, loc)[0]
    Fu = max(Fu, 1e-12)
    z = np.clip(Fx / Fu, 0.0, 1.0)
    z[np.asarray(x) > u_star] = 1.0
    return z

def sample_shifted_lognorm_mixture(n, loc, w, mu, sigma, random_state=None):
    rng = np.random.default_rng(random_state)
    w   = np.asarray(w, float); w = w / w.sum()
    mu  = np.asarray(mu, float)
    sig = np.asarray(sigma, float)
    comp = rng.choice(len(w), size=n, p=w)
    z    = rng.normal(loc=mu[comp], scale=sig[comp])
    return loc + np.exp(z)

# def sample_shifted_lognorm_truncated(n, loc, w, mu, sigma, u_star, random_state=None):
#     rng = np.random.default_rng(random_state)
#     out = np.empty(n, dtype=float); i = 0
#     Fu = mixture_cdf_lognorm_shift(np.array([u_star], float), w, mu, sigma, loc)[0]
#     Fu = max(Fu, 1e-9)
#     while i < n:
#         m = int((n - i) / Fu * 1.2) + 1
#         xs = sample_shifted_lognorm_mixture(m, loc, w, mu, sigma, random_state=rng.integers(1<<30))
#         xs = xs[xs <= u_star]
#         k = min(xs.size, n - i)
#         if k > 0:
#             out[i:i+k] = xs[:k]; i += k
#     return out

def sample_shifted_lognorm_truncated(n, loc, w, mu, sigma, u_star, random_state=None):
    """
    Muestrea n valores de la mezcla shifted lognormal y aplica truncamiento aleatorio:
    todo valor > u_star se reemplaza por U(0.8*u_star, u_star). No usa rechazo.
    """
    rng = np.random.default_rng(random_state)
    x = sample_shifted_lognorm_mixture(n, loc, w, mu, sigma, random_state=rng)
    trunc_rand(x, upper=u_star, on=True, rng=rng)
    return x

# ======= Pérdida compuesta y loc_grid dinámico =======
def _inv_cdf_safe(q, w, mu, sigma, loc, data_max):
    xq = inv_cdf_mixture(q, w, mu, sigma, loc, data_max=data_max)
    return xq if (xq is not None and np.isfinite(xq)) else None

def composite_cdf_loss(x_s, y_s, w, mu, sigma, loc,
                       w_sse=W_SSE, w_ks=W_KS, w_tail=W_TAIL, tail_gate=TAIL_GATE,
                       quant_pen=QUANT_PEN):
    y_hat = mixture_cdf_lognorm_shift(x_s, w, mu, sigma, loc)
    sse   = np.mean((y_hat - y_s)**2)
    ks    = np.max(np.abs(y_hat - y_s))
    tail_mask = (y_s >= tail_gate)
    tail_mse  = np.mean((y_hat[tail_mask] - y_s[tail_mask])**2) if np.any(tail_mask) else 0.0

    # Penalización de cuantiles (suave)
    data_max = float(x_s[-1])
    qpen = 0.0
    for q, wq in quant_pen:
        x_emp = float(np.quantile(x_s, q))
        x_mod = _inv_cdf_safe(q, w, mu, sigma, loc, data_max)
        if x_mod is not None:
            denom = max(x_emp, 1.0)
            qpen += wq * ((x_mod - x_emp)/denom)**2

    loss = w_sse*sse + w_ks*ks + w_tail*tail_mse + qpen
    return loss, (sse, ks, tail_mse, qpen)

def make_dynamic_loc_grid(v, n_grid=40, widen=2.0, eps_frac=0.02):
    q05, q95 = np.quantile(v, [0.05, 0.95])
    span = max(q95 - q05, 1.0)
    vmin = float(v.min())
    lo = vmin - widen*span
    hi = vmin - eps_frac*span
    grid = np.linspace(lo, hi, n_grid)
    grid = np.minimum(grid, vmin - 1e-9)
    return grid


# ======= Fit (ECDF) con pérdida compuesta y loc_grid dinámico =======
def fit_mix_lognorm_shift_ecdf(v, k_range=(1,2,3), loc_grid=None,
                               random_state=RANDOM_STATE, n_init=5):
    x_s, y_s = ecdf_xy(v)
    if loc_grid is None:
        loc_grid = make_dynamic_loc_grid(v)

    best, records = None, []
    for loc in loc_grid:
        u = v - loc
        if np.any(u <= 0):  # requiere v > loc
            continue
        ylog = np.log(u).reshape(-1,1)
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                  random_state=random_state, n_init=n_init)
            gmm.fit(ylog)
            w = gmm.weights_.ravel()
            mu = gmm.means_.ravel()
            sigma = np.sqrt(gmm.covariances_.ravel())

            loss, parts = composite_cdf_loss(x_s, y_s, w, mu, sigma, loc)
            sse, ks, tail_mse, qpen = parts
            bic  = float(gmm.bic(ylog))
            rec = dict(model="Mix Lognorm Function", loc=loc, sse=sse, ks=ks,
                       tail_mse=tail_mse, qpen=qpen, loss=loss, bic=bic, w=w, mu=mu, sigma=sigma, k=k)
            records.append(rec)
            if (best is None) or (loss < best["loss"]):
                best = rec

    summary = pd.DataFrame(records).sort_values(["loss","bic"]).reset_index(drop=True)
    return (x_s, y_s), summary, best


# ======= Calibración opcional: Δμ para cuantil objetivo =======
def calibrate_quantile_mu_shift(best, x_target, q_target=Q_TARGET,
                                delta_min=-3.0, delta_max=3.0,
                                tol=1e-8, maxit=80):
    w, mu, sigma, loc = best["w"], best["mu"].copy(), best["sigma"], best["loc"]
    def F_with_delta(delta):
        return mixture_cdf_lognorm_shift(np.array([x_target], float), w, mu + delta, sigma, loc)[0]
    f_lo = F_with_delta(delta_min); f_hi = F_with_delta(delta_max)
    g_lo, g_hi = f_lo - q_target, f_hi - q_target
    expand = 0; dmin, dmax = delta_min, delta_max
    while g_lo * g_hi > 0 and expand < 5:
        expand += 1; dmin -= 2.0; dmax += 2.0
        f_lo = F_with_delta(dmin); f_hi = F_with_delta(dmax)
        g_lo, g_hi = f_lo - q_target, f_hi - q_target
    if g_lo * g_hi > 0:
        return None, dict(ok=False, msg="No se pudo bracketear Δ para el cuantil objetivo.")
    for _ in range(maxit):
        mid = 0.5*(dmin + dmax); fm = F_with_delta(mid) - q_target
        if abs(fm) <= tol * max(1.0, q_target):
            mu_cal = mu + mid
            return mu_cal, dict(ok=True, delta=mid, F_at_target=fm + q_target)
        if fm > 0: dmin = mid
        else:      dmax = mid
    mu_cal = mu + 0.5*(dmin + dmax)
    return mu_cal, dict(ok=True, delta=0.5*(dmin + dmax), F_at_target=F_with_delta(0.5*(dmin + dmax)))


# ===================== Plots ======================
def plot_histogram_smoothed(v, out_path=OUT_FIG_HIST, clip_xlim=True):
    ensure_dir(os.path.dirname(out_path) or "figures")
    edges, dens_sm = smoothed_hist(v, bins=HIST_BINS, smooth_window=SMOOTH_WINDOW)
    widths = np.diff(edges)
    area = float(np.sum(dens_sm * widths))
    print(f"[CHECK] Smoothed-hist area (should ≈ 1): {area:.6f}")

    plt.figure(figsize=FIG_SIZE, layout="constrained")
    plt.bar(edges[:-1], dens_sm, width=widths, align='edge', alpha=0.85, edgecolor="none", label="Empirical histogram")
    plt.step(edges, np.r_[dens_sm, dens_sm[-1]], where="post", lw=1.2, ls="--", color="k", alpha=0.7, label="Empirical PDF")
    plt.title("Empirical distribution")
    plt.xlabel(X_LABEL); plt.ylabel("Density")
    plt.grid(True, alpha=0.3); plt.legend(loc="best")
    if clip_xlim:
        xmin = float(edges[0]); plt.xlim(xmin, XLIM_MAX)
    plt.tight_layout(); plt.savefig(out_path, dpi=DPI); plt.close()


def plot_ecdf_validation(v, best, n_sim=N_SIM_MIN, out_path=OUT_FIG_ECDF, clip_xlim=True):
    ensure_dir(os.path.dirname(out_path) or "figures")
    x_ecdf, y_ecdf = ecdf_xy(v)
    xmin, xmax = float(x_ecdf.min()), float(x_ecdf.max())

    if USE_TRUNCATED_FOR_VALIDATION and "upper_trunc" in best and np.isfinite(best["upper_trunc"]):
        u_star = best["upper_trunc"]
        sim = sample_shifted_lognorm_truncated(n_sim, best["loc"], best["w"], best["mu"], best["sigma"], u_star=u_star, random_state=RANDOM_STATE)
        x_ecdf_sim, y_ecdf_sim = ecdf_xy(sim)
        xfit = np.linspace(xmin, xmax if clip_xlim else max(xmax, u_star), 1500)
        yfit = cdf_truncated(xfit, best["w"], best["mu"], best["sigma"], best["loc"], u_star)
        model_label = "Model CDF (truncated)"
        sim_label   = f"Simulated ECDF (trunc, n={n_sim})"
    else:
        sim = sample_shifted_lognorm_mixture(n_sim, best["loc"], best["w"], best["mu"], best["sigma"], random_state=RANDOM_STATE)
        x_ecdf_sim, y_ecdf_sim = ecdf_xy(sim)
        xfit = np.linspace(xmin, xmax if clip_xlim else max(xmax, sim.max()), 1500)
        yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])
        model_label = "Model CDF"
        sim_label   = f"Simulated ECDF (n={n_sim})"

    plt.figure(figsize=(7,4.5))
    plt.step(x_ecdf, y_ecdf, where="post", lw=1.8, label="Empirical ECDF")
    plt.step(x_ecdf_sim, y_ecdf_sim, where="post", lw=1.8, label=sim_label)
    plt.plot(xfit, yfit, "--", lw=2.0, label=model_label)

    if "upper_trunc" in best and np.isfinite(best["upper_trunc"]):
        plt.axvline(best["upper_trunc"], ls=":", lw=1.6, color="crimson", alpha=0.8, label=f"Upper trunc (F={best.get('Fu', np.nan):.3f})")

    plt.title("Validation with simulated data")
    plt.xlabel(X_LABEL); plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    if clip_xlim:
        plt.xlim(xmin, xmax); plt.axvline(xmax, ls=":", lw=1.2, color="k", alpha=0.4)
    plt.ylim(0,1)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_mixture_hist_with_ecdf(x, comp, bins=HIST_BINS, mode="stacked", show_points=False,
                                out_path=OUT_FIG_MIXED):
    ensure_dir(os.path.dirname(out_path) or "figures")

    colors = ["C0", "C1", "C2"]
    labels = ["Component 1", "Component 2", "Component 3"]

    fig, ax1 = plt.subplots(figsize=FIG_SIZE, layout="constrained")
    edges = np.histogram_bin_edges(x, bins=bins)
    widths = np.diff(edges); B = len(widths); lefts = edges[:-1]

    K = int(np.max(comp)) + 1; n = x.size
    dens = []
    for k in range(K):
        cnt, _ = np.histogram(x[comp == k], bins=edges)
        dens.append(cnt / (n * widths))
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
            ax1.bar(lefts + (k / g) * widths, dens[k], width=widths / g, align="edge", color=c, alpha=0.8, edgecolor="none", label=lbl)
        total = dens.sum(axis=0)
    else:
        raise ValueError("mode must be 'stacked' or 'grouped'.")

    ax1.step(edges, np.r_[total, total[-1]], where="post", lw=1.2, ls="--", color="k", alpha=0.7, label="Simulated PDF")
    ax1.set_xlabel(X_LABEL); ax1.set_ylabel("Density"); ax1.grid(True, alpha=0.3, zorder=0)

    ax2 = ax1.twinx()
    xs_all = np.sort(x); ys_all = np.arange(1, x.size + 1) / x.size
    ax2.step(xs_all, ys_all, where="post", lw=3.0, color="1.0", alpha=1.0)
    ax2.step(xs_all, ys_all, where="post", lw=1.3, ls=(0, (4, 2)), color="0.15", alpha=1.0, label="Simulated CDF")
    if show_points:
        order = np.argsort(x); comp_sorted = comp[order]
        idx = np.linspace(0, x.size - 1, min(400, x.size), dtype=int)
        ax2.scatter(xs_all[idx], ys_all[idx], s=14, c=np.take(colors, comp_sorted[idx], mode='wrap'),
                    alpha=0.9, edgecolors="white", linewidths=0.4)
    ax2.set_ylim(0, 1); ax2.set_ylabel("Cumulative Probability"); ax2.tick_params(axis="y", colors="0.15")

    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + ([h2[0]] if h2 else []), l1 + ([l2[0]] if l2 else []), loc="best")

    xmin_plot = float(edges[0]); ax1.set_xlim(xmin_plot, XLIM_MAX)
    plt.tight_layout(); plt.savefig(out_path, dpi=DPI); plt.close()


# ===================== Main ======================
def main():
    ensure_dir("figures"); ensure_dir(os.path.dirname(OUT_TXT) or ".")
    df = load_data(SYS_FILE)
    v = df["FLOPs"].to_numpy(float); v = v[np.isfinite(v) & (v > 0)]

    # 1) Fit con pérdida compuesta (relajada) y loc_grid dinámico
    (x_ecdf, y_ecdf), summary, best = fit_mix_lognorm_shift_ecdf(v, k_range=(1,2,3), loc_grid=None,
                                                                 random_state=RANDOM_STATE, n_init=5)

    # 2) Calibración opcional del cuantil objetivo
    if CALIBRATE_QUANTILE:
        x_target = float(np.quantile(v, Q_TARGET))
        mu_cal, info = calibrate_quantile_mu_shift(best, x_target, q_target=Q_TARGET)
        if (mu_cal is not None) and info.get("ok", False):
            best_used = dict(best); best_used["mu"] = mu_cal
            best_used["calib_ok"] = True; best_used["delta_mu"] = info["delta"]
            best_used["F_at_target_post"] = info["F_at_target"]
        else:
            best_used = dict(best); best_used["calib_ok"] = False
    else:
        best_used = dict(best); best_used["calib_ok"] = False
        x_target, info = None, None

    # 3) Truncamiento (se calcula y se muestra la línea; no renormaliza por defecto en validación)
    upper_trunc, Fu = compute_upper_truncation(best_used, v, tail_mass_cut=TAIL_MASS_CUT)
    best_used["upper_trunc"] = upper_trunc; best_used["Fu"] = Fu

    # 4) KS en puntos empíricos (modelo no truncado, como en el fit)
    Fm_ecdf = mixture_cdf_lognorm_shift(x_ecdf, best_used["w"], best_used["mu"], best_used["sigma"], best_used["loc"])
    ks_ecdf = float(np.max(np.abs(Fm_ecdf - y_ecdf)))

    print("\n=== Best candidates (sorted by LOSS) ===")
    print(summary.head(10).to_string(index=False))

    # 5) Guardar TXT
    SSE_sum = float(np.sum((Fm_ecdf - y_ecdf)**2))
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

    # 6) Figuras
    plot_histogram_smoothed(v, out_path=OUT_FIG_HIST, clip_xlim=True)
    print(f"Histogram figure saved to: {OUT_FIG_HIST}")

    plot_ecdf_validation(v, best_used, n_sim=N_SIM_MIN, out_path=OUT_FIG_ECDF, clip_xlim=True)
    print(f"ECDF validation figure saved to: {OUT_FIG_ECDF}")

    # Simulación para figura de componentes (NO truncada — enfoque relajado)
    x_sim = sample_shifted_lognorm_mixture(N_SIM_MIN, best_used["loc"], best_used["w"], best_used["mu"], best_used["sigma"], random_state=RANDOM_STATE)
    rng_lbl = np.random.default_rng(RANDOM_STATE)
    comp_sim = rng_lbl.choice(len(best_used["w"]), size=x_sim.size, p=np.array(best_used["w"])/np.sum(best_used["w"]))
    plot_mixture_hist_with_ecdf(x_sim, comp_sim, bins=HIST_BINS, mode="stacked", show_points=False, out_path=OUT_FIG_MIXED)
    print(f"Mixture histogram + ECDF figure saved to: {OUT_FIG_MIXED}")

    # 7) Stats
    print("\n=== Summary statistics ===")
    print(f"Real:      mean={np.mean(v):.1f}, median={np.median(v):.1f}, std={np.std(v, ddof=1):.1f}")
    print(f"Simulated: mean={np.mean(x_sim):.1f}, median={np.median(x_sim):.1f}, std={np.std(x_sim, ddof=1):.1f}")
    print(f"(n_real={len(v)}, n_sim={N_SIM_MIN})")
    print(f"KS_ECDF_used={ks_ecdf:.6f}")

if __name__ == "__main__":
    main()
