# fit_shifted_lognorm_mixture.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a shifted-lognormal mixture on FLOPs (or time) with flexible normalization.
Saves parameters in both human-readable TXT and machine-readable JSON,
including normalization metadata (a,b) for minmax/minmax_robust or s for scale.

Key features:
- Normalization modes:
  "none" | "median" | "gflops" | "const" | "pXX" (pure scaling v' = v / s)
  "minmax" | "minmax_robust"  (affine v' = (v - a) / b)
- Robust GMM fit in log-space with diagonal covariances, reg_covar, sigma floor
- Small-weights handling: Option A (post-floor), B (penalty), C (refit with k-1)
- Two-stage loc search: coarse absolute ms offsets (converted to current units) + refine
- Selection: penalized SSE (primary) + KS/BIC tie-breakers
- Plots and params saved under figures/fit_log/
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

# ===================== Paths & IO =====================
METRIC = "flops"  # "flops" (supports normalization) or "time"
PATH_FOLDER = "../results/sys/"
NAME_FILE   = "sys_metrics_minibatch_c_20_mb_0.6"
EXT_FILE    = ".csv"

OUT_DIR         = "figures/fit_log"
OUT_FIG_MODEL   = os.path.join(OUT_DIR, "mix_lognorm_shift_fit.png")
OUT_FIG_COMPARE = os.path.join(OUT_DIR, "mix_lognorm_shift_fit_and_sim.png")
OUT_TXT         = os.path.join(OUT_DIR, "mix_lognorm_shift_params.txt")
OUT_JSON        = os.path.join(OUT_DIR, "mix_lognorm_shift_params.json")

# ===================== Plot / ECDF =====================
BINS = 256
SMOOTH_WINDOW = 7
PLOT_LOGX = True  # gets auto-disabled when data ~[0,1] after minmax

# ===================== Simulation =====================
RANDOM_STATE = 42
N_SIM_MIN    = 10_000

# ===================== FLOPs ↔ time =====================
FLOPS_PER_SEC = 5e9  # used for ms offset conversion and for reporting

# ===================== Normalization (FLOPs only) =====================
# Modes: "none" | "median" | "gflops" | "const" | "pXX" | "minmax" | "minmax_robust"
NORMALIZE_MODE = "minmax_robust"
NORM_CONST     = 1e9
ROBUST_P_LO    = 1.0   # [1..49]
ROBUST_P_HI    = 99.0  # [51..99]

# ===================== Mixture / Fit controls =====================
K_RANGE       = (1, 2, 3)
GMM_REG_COVAR = 1e-6
SIGMA_MIN     = 1e-3

# Tiny weights handling
W_MIN_FLOOR      = 0.05
ENABLE_OPTION_A  = True   # post-floor & renormalize
ENABLE_OPTION_B  = True   # penalize small weights in score
ENABLE_OPTION_C  = True   # refit with k-1 if small weights exist
LAMBDA_SMALL_PEN = 0.05
USE_ECDF_FOR_FIT = True   # fit against exact ECDF (smoothed CDF for visualization only)

# loc search (absolute ms offsets, converted to current units)
COARSE_LOC_GRID_POINTS_ABS_MS = 16
COARSE_LOC_OFFSET_LO_ABS_MS   = 800.0   # vmin - 800 ms
COARSE_LOC_OFFSET_HI_ABS_MS   = 100.0   # vmin - 100 ms
REFINE_LOC              = True
REFINE_LOC_POINTS       = 21
REFINE_LOC_SPAN_ABS_MS  = 120.0        # ±120 ms

# ===================== Utils =====================
def ensure_dir(path=OUT_DIR):
    os.makedirs(path, exist_ok=True); return path

def _find_col(df, candidates):
    cols = list(df.columns); low = {c: c.lower() for c in cols}
    for cand in candidates:
        cl = cand.lower()
        for c, lc in low.items():
            if cl in lc: return c
    return None

def load_data(sys_metrics_file):
    """Robust CSV loading with name-based mapping; falls back to legacy order."""
    if not os.path.isfile(sys_metrics_file):
        raise FileNotFoundError(f"File not found: {sys_metrics_file}")
    df = pd.read_csv(sys_metrics_file)
    m = {}
    m["client_id"]     = _find_col(df, ["client_id","clientid","client","cid"])
    m["round_number"]  = _find_col(df, ["round_number","round","round_id"])
    m["samples"]       = _find_col(df, ["samples","num_samples","nsamples"])
    m["set"]           = _find_col(df, ["set"])
    m["bytes_read"]    = _find_col(df, ["bytes_read","bytesin","bytes_in","input_bytes"])
    m["bytes_written"] = _find_col(df, ["bytes_written","bytesout","bytes_out","output_bytes"])
    m["FLOPs"]         = _find_col(df, ["flops","FLOPs","gflops","ops"])
    if any(v is None for v in m.values()):
        df.columns = ["client_id","round_number","idk","samples","set","bytes_read","bytes_written","FLOPs"]
    else:
        norm = {k: df[col] for k, col in m.items() if col is not None}
        norm["idk"] = np.nan
        df = pd.DataFrame(norm)
    df.index.name = "index"
    return df

def convert_FLOPs_to_time(df, flops_per_sec=FLOPS_PER_SEC):
    df = df.copy(); df["time"] = (df["FLOPs"] / float(flops_per_sec)) * 1e3  # ms
    return df

def ecdf_xy(vals):
    x = np.sort(vals); n = x.size
    y = np.arange(1, n+1, dtype=float) / n
    return x, y

def smoothed_cdf_from_hist(vals, bins=BINS, smooth_window=SMOOTH_WINDOW):
    counts, edges = np.histogram(vals, bins=bins, density=True)
    widths = np.diff(edges)
    if smooth_window and smooth_window > 1:
        k = int(smooth_window); kernel = np.ones(k)/k
        counts = np.convolve(counts, kernel, mode="same")
    cdf = np.cumsum(counts * widths)
    if cdf[-1] > 0: cdf /= cdf[-1]
    x = edges[1:]
    return x, cdf

# ---------------- Normalization helpers ----------------
def _compute_scale(v, mode, const_val=1e9):
    mode = str(mode).lower()
    if mode == "median":  return float(np.median(v)), "median"
    if mode == "gflops":  return 1e9, "1e9 (GFLOPs)"
    if mode == "const":   return float(const_val), f"{float(const_val):.3g}"
    if mode.startswith("p") and mode[1:].isdigit():
        q = min(max(int(mode[1:]), 1), 99)
        return float(np.quantile(v, q/100.0)), f"p{q}"
    return float(np.median(v)), "median"

def _compute_affine(v, mode, p_lo=1.0, p_hi=99.0):
    mode = str(mode).lower()
    if mode == "minmax":
        a = float(np.min(v)); vmax = float(np.max(v)); b = max(vmax - a, 1e-12)
        return a, b, "min–max"
    # robust
    p_lo = float(np.clip(p_lo, 0.0, 49.0))
    p_hi = float(np.clip(p_hi, 51.0, 100.0))
    a  = float(np.quantile(v, p_lo/100.0))
    hi = float(np.quantile(v, p_hi/100.0))
    b  = max(hi - a, 1e-12)
    return a, b, f"min–max robust (p{int(p_lo)}–p{int(p_hi)})"

def _build_normalized_vector(v_units, metric, norm_mode):
    """
    Return v_fit (possibly normalized), norm_info dict, x_label and logx flag.
    norm_info:
      - {"type":"none"}
      - {"type":"scale","s":..., "desc":...}
      - {"type":"affine","a":..., "b":..., "desc":...}
    """
    if metric.lower() == "time":
        return v_units, {"type":"none"}, "Time (ms)", PLOT_LOGX

    m = str(norm_mode).lower()
    if m == "none":
        return v_units, {"type":"none"}, "FLOPs", PLOT_LOGX

    if m in ("median","gflops","const") or (m.startswith("p") and m[1:].isdigit()):
        s, desc = _compute_scale(v_units, m, NORM_CONST)
        v = v_units / s
        return v, {"type":"scale","s":s,"desc":desc}, f"FLOPs (scaled by {desc})", (False if m=="median" else PLOT_LOGX)

    if m in ("minmax","minmax_robust"):
        if m == "minmax": a,b,desc = _compute_affine(v_units,"minmax")
        else:             a,b,desc = _compute_affine(v_units,"minmax_robust",ROBUST_P_LO,ROBUST_P_HI)
        v = (v_units - a) / b
        return v, {"type":"affine","a":a,"b":b,"desc":desc}, f"FLOPs ({desc})", False

    return v_units, {"type":"none"}, "FLOPs", PLOT_LOGX

def params_from_norm_to_units(best_norm, norm_info):
    """Convert best params from normalized space to original units."""
    out = dict(best_norm); t = norm_info.get("type","none")
    if t == "scale":
        s = float(norm_info["s"])
        out["loc"]   = s * float(best_norm["loc"])
        out["mu"]    = np.asarray(best_norm["mu"], float) + np.log(s)
    elif t == "affine":
        a = float(norm_info["a"]); b = float(norm_info["b"])
        out["loc"]   = a + b * float(best_norm["loc"])
        out["mu"]    = np.asarray(best_norm["mu"], float) + np.log(b)
    else:
        out["loc"]   = float(best_norm["loc"])
        out["mu"]    = np.asarray(best_norm["mu"], float)
    out["sigma"] = np.asarray(best_norm["sigma"], float)
    out["w"]     = np.asarray(best_norm["w"], float)
    return out

def inv_transform_samples(x_fit, norm_info):
    """Map samples from fitting units back to original units."""
    t = norm_info.get("type","none")
    if t == "scale":  return float(norm_info["s"]) * np.asarray(x_fit, float)
    if t == "affine":
        a = float(norm_info["a"]); b = float(norm_info["b"])
        return a + b * np.asarray(x_fit, float)
    return np.asarray(x_fit, float)

# ======= Mixture of shifted lognormals =======
def mixture_cdf_lognorm_shift(x, w, mu, sigma, loc):
    """
    F(x) = sum_i w_i * Phi( (log(x - loc) - mu_i) / sigma_i ) for x > loc; 0 otherwise.
    """
    x = np.asarray(x, float)
    F = np.zeros_like(x, float)
    mask = x > loc
    if not np.any(mask): return F
    z = np.log(np.clip(x[mask] - loc, 1e-12, None))
    w = np.asarray(w, float); mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    for wi, m, s in zip(w, mu, sigma):
        F[mask] += wi * stats.norm.cdf((z - m) / s)
    return np.clip(F, 0.0, 1.0)

def _evaluate_model_on_reference(x_ref, y_ref, w, mu, sigma, loc):
    yhat = mixture_cdf_lognorm_shift(x_ref, w, mu, sigma, loc)
    r = yhat - y_ref
    return {"yhat": yhat, "sse": float(np.sum(r**2)), "ks": float(np.max(np.abs(r)))}

def _post_floor_weights(w, w_min=W_MIN_FLOOR):
    w = np.asarray(w, float)
    w2 = np.maximum(w, float(w_min)); w2 /= w2.sum()
    return w2

def _fit_gmm_logspace(ylog, k, random_state=RANDOM_STATE, reg_covar=GMM_REG_COVAR):
    gmm = GaussianMixture(n_components=k, covariance_type="diag",
                          random_state=random_state, n_init=5, reg_covar=reg_covar)
    gmm.fit(ylog)
    w = gmm.weights_.ravel()
    mu = gmm.means_.ravel()
    sigma = np.sqrt(gmm.covariances_.ravel())
    sigma = np.maximum(sigma, SIGMA_MIN)
    bic = float(gmm.bic(ylog))
    return w, mu, sigma, bic, gmm

def _penalized_score(base_sse, small_count):
    return float(base_sse + (LAMBDA_SMALL_PEN * float(small_count) if ENABLE_OPTION_B else 0.0))

def _ms_to_units_for_offsets(ms, metric, flops_per_sec):
    if metric.lower() == "time": return float(ms)  # ms
    return float(ms) * float(flops_per_sec) / 1000.0  # FLOPs

def _fit_for_loc(v, loc, k_range=K_RANGE, use_ecdf=USE_ECDF_FOR_FIT):
    u = v - loc
    if np.any(u <= 0): return []
    ylog = np.log(u).reshape(-1,1)
    x_ref,y_ref = ecdf_xy(v) if use_ecdf else smoothed_cdf_from_hist(v, bins=BINS, smooth_window=SMOOTH_WINDOW)
    cands = []
    for k in k_range:
        w, mu, sigma, bic, _ = _fit_gmm_logspace(ylog, k)
        small_count = int(np.sum(w < W_MIN_FLOOR))

        ev = _evaluate_model_on_reference(x_ref, y_ref, w, mu, sigma, loc)
        cands.append(dict(model=f"mixlognorm(k={k})", k=k, loc=loc, w=w, mu=mu, sigma=sigma,
                          bic=bic, sse=ev["sse"], ks=ev["ks"], small_count=small_count,
                          score=_penalized_score(ev["sse"], small_count), note="raw"))

        if ENABLE_OPTION_A and small_count > 0:
            wA = _post_floor_weights(w)
            evA = _evaluate_model_on_reference(x_ref, y_ref, wA, mu, sigma, loc)
            cands.append(dict(model=f"mixlognorm(k={k})", k=k, loc=loc, w=wA, mu=mu, sigma=sigma,
                              bic=bic, sse=evA["sse"], ks=evA["ks"], small_count=0,
                              score=_penalized_score(evA["sse"], 0), note="post_floor_w"))

        if ENABLE_OPTION_C and small_count > 0 and k > 1:
            k2 = k - 1
            w2, mu2, sigma2, bic2, _ = _fit_gmm_logspace(ylog, k2)
            small_count2 = int(np.sum(w2 < W_MIN_FLOOR))
            ev2 = _evaluate_model_on_reference(x_ref, y_ref, w2, mu2, sigma2, loc)
            cands.append(dict(model=f"mixlognorm(k={k2})", k=k2, loc=loc, w=w2, mu=mu2, sigma=sigma2,
                              bic=bic2, sse=ev2["sse"], ks=ev2["ks"], small_count=small_count2,
                              score=_penalized_score(ev2["sse"], small_count2), note="refit_k_minus_1"))
    return cands

def _build_coarse_loc_grid(vmin, metric, flops_per_sec, norm_info):
    off_lo = _ms_to_units_for_offsets(COARSE_LOC_OFFSET_LO_ABS_MS, metric, flops_per_sec)
    off_hi = _ms_to_units_for_offsets(COARSE_LOC_OFFSET_HI_ABS_MS, metric, flops_per_sec)
    div = 1.0
    t = norm_info.get("type","none")
    if t == "scale":  div = float(norm_info["s"])
    elif t == "affine": div = float(norm_info["b"])
    lo = vmin - off_lo / div
    hi = vmin - off_hi / div
    return np.linspace(lo, hi, int(COARSE_LOC_GRID_POINTS_ABS_MS))

def _refine_span(metric, flops_per_sec, norm_info):
    span = _ms_to_units_for_offsets(REFINE_LOC_SPAN_ABS_MS, metric, flops_per_sec)
    t = norm_info.get("type","none")
    if t == "scale":  return span / float(norm_info["s"])
    if t == "affine": return span / float(norm_info["b"])
    return span

def fit_mix_lognorm_shift(v_fit, k_range=K_RANGE, use_ecdf=USE_ECDF_FOR_FIT,
                          metric=METRIC, flops_per_sec=FLOPS_PER_SEC, norm_info=None):
    if norm_info is None: norm_info = {"type":"none"}
    v_fit = np.asarray(v_fit, float); vmin = float(np.min(v_fit))

    all_candidates = []
    for loc in _build_coarse_loc_grid(vmin, metric, flops_per_sec, norm_info):
        all_candidates.extend(_fit_for_loc(v_fit, loc, k_range=k_range, use_ecdf=use_ecdf))
    if not all_candidates:
        raise RuntimeError("No valid candidates found; check loc grid or data.")

    summary_coarse = pd.DataFrame(all_candidates).sort_values(["score","ks","bic","sse"]).reset_index(drop=True)
    best_coarse = summary_coarse.iloc[0].to_dict()

    if REFINE_LOC:
        loc0 = float(best_coarse["loc"]); span = _refine_span(metric, flops_per_sec, norm_info)
        all_candidates_ref = []
        for loc in np.linspace(loc0 - span, loc0 + span, int(REFINE_LOC_POINTS)):
            all_candidates_ref.extend(_fit_for_loc(v_fit, loc, k_range=k_range, use_ecdf=use_ecdf))
        if all_candidates_ref:
            summary_ref = pd.DataFrame(all_candidates_ref).sort_values(["score","ks","bic","sse"]).reset_index(drop=True)
            best_ref = summary_ref.iloc[0].to_dict()
            summary = pd.concat([summary_coarse, summary_ref], ignore_index=True).sort_values(["score","ks","bic","sse"]).reset_index(drop=True)
            best = best_ref
        else:
            summary = summary_coarse; best = best_coarse
    else:
        summary = summary_coarse; best = best_coarse

    x_ref,y_ref = (ecdf_xy(v_fit) if use_ecdf else smoothed_cdf_from_hist(v_fit, bins=BINS, smooth_window=SMOOTH_WINDOW))
    x_sm,y_sm   = smoothed_cdf_from_hist(v_fit, bins=BINS, smooth_window=SMOOTH_WINDOW)
    return (x_ref,y_ref), (x_sm,y_sm), summary, best

# ===================== Plots =====================
def plot_model(v_fit, ref_curve, smoothed_curve, best, out_path=OUT_FIG_MODEL, logx=PLOT_LOGX, x_label="Fitting units"):
    ensure_dir(os.path.dirname(out_path) or OUT_DIR)
    x_ecdf,y_ecdf = ref_curve; x_sm,y_sm = smoothed_curve
    xfit = np.linspace(v_fit.min(), v_fit.max(), 1200)
    yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])

    plt.figure(figsize=(11,5))
    plt.step(x_ecdf,y_ecdf, where="post", lw=1.8, label="Empirical ECDF (reference)")
    plt.plot(x_sm,y_sm, lw=2.0, label=f"Smoothed CDF (bins={BINS}, win={SMOOTH_WINDOW})")
    plt.plot(xfit,yfit, "--", lw=2.2, label=f"{best['model']}, loc={best['loc']:.6g}")
    plt.ylim(0,1); plt.xlabel(x_label); plt.ylabel("Cumulative Probability")
    plt.title("Shifted Lognormal Mixture — Fit vs ECDF"); plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    if logx:
        pos = x_ecdf[x_ecdf > 0]
        if pos.size > 0:
            plt.xscale("log"); plt.xlim(left=float(np.min(pos)), right=float(np.max(x_ecdf)))
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_fit_and_simulation(v_fit, ref_curve, best, n_sim,
                            out_path=OUT_FIG_COMPARE, clip_to_real_xlim=True,
                            truncate_and_renormalize=False, logx=PLOT_LOGX, x_label="Fitting units"):
    ensure_dir(os.path.dirname(out_path) or OUT_DIR)
    x_ecdf,y_ecdf = ref_curve; xmin,xmax = float(x_ecdf.min()), float(x_ecdf.max())
    # simulate in fit units
    rng = np.random.default_rng(RANDOM_STATE)
    comp = rng.choice(len(best["w"]), size=n_sim, p=best["w"]/np.sum(best["w"]))
    z = rng.normal(best["mu"][comp], best["sigma"][comp]); sim = best["loc"] + np.exp(z)
    if truncate_and_renormalize:
        sim_vis = sim[(sim >= xmin) & (sim <= xmax)]
        x_ecdf_sim,y_ecdf_sim = ecdf_xy(sim_vis)
    else:
        x_ecdf_sim,y_ecdf_sim = ecdf_xy(sim)

    xfit = np.linspace(xmin, xmax if clip_to_real_xlim else max(xmax, sim.max()), 1500)
    yfit = mixture_cdf_lognorm_shift(xfit, best["w"], best["mu"], best["sigma"], best["loc"])

    fig = plt.figure(figsize=(12,5), layout="constrained"); ax1,ax2 = fig.subplots(1,2, sharey=True)
    ax1.step(x_ecdf,y_ecdf, where="post", lw=1.8, label="Empirical ECDF"); ax1.plot(xfit,yfit,"--",lw=2.2,label=f"{best['model']}")
    ax1.set_title("Fit on real dataset"); ax1.set_xlabel(x_label); ax1.set_ylabel("Cumulative Probability"); ax1.grid(True, alpha=0.3); ax1.legend(loc="lower right")
    ax2.step(x_ecdf,y_ecdf, where="post", lw=1.8, label="Empirical ECDF"); ax2.step(x_ecdf_sim,y_ecdf_sim, where="post", lw=1.8, label=f"Simulated ECDF (n={n_sim})")
    ax2.set_title("Validation with simulated data"); ax2.set_xlabel(x_label); ax2.grid(True, alpha=0.3); ax2.legend(loc="lower right")

    if clip_to_real_xlim:
        ax1.set_xlim(xmin, xmax); ax2.set_xlim(xmin, xmax); ax2.axvline(xmax, ls=":", lw=1.2, color="k", alpha=0.4)
    yticks = np.linspace(0,1,6)
    for ax in (ax1,ax2): ax.set_ylim(0,1); ax.set_yticks(yticks)
    if logx:
        def _safe_logx(ax):
            pos = x_ecdf[x_ecdf>0]
            if pos.size>0: ax.set_xscale("log"); ax.set_xlim(left=float(np.min(pos)), right=xmax)
        _safe_logx(ax1); _safe_logx(ax2)
    ax1.tick_params(axis="y", labelleft=True, labelright=False)
    ax2.tick_params(axis="y", labelleft=False, labelright=True)
    fig.suptitle("Mixture (shifted lognormal): fit and validation", fontsize=14)
    plt.savefig(out_path, dpi=150); plt.close()

# ===================== Main =====================
def main():
    sys_metrics_file = os.path.join(PATH_FOLDER, NAME_FILE + EXT_FILE)
    ensure_dir(OUT_DIR)
    df = load_data(sys_metrics_file)

    # Build fitting vector
    if METRIC.lower() == "flops":
        v_units = df["FLOPs"].to_numpy(float)
        v_units = v_units[np.isfinite(v_units) & (v_units > 0)]
        if v_units.size < 50: raise ValueError(f"Not enough valid samples: {v_units.size} (<50).")
        v_fit, norm_info, x_label, logx_flag = _build_normalized_vector(v_units, metric="flops", norm_mode=NORMALIZE_MODE)
        metric_for_fit = "flops"
    elif METRIC.lower() == "time":
        df = convert_FLOPs_to_time(df, flops_per_sec=FLOPS_PER_SEC)
        v_units = df["time"].to_numpy(float)
        v_units = v_units[np.isfinite(v_units) & (v_units > 0)]
        if v_units.size < 50: raise ValueError(f"Not enough valid samples: {v_units.size} (<50).")
        v_fit, norm_info, x_label, logx_flag = v_units, {"type":"none"}, "Time (ms)", PLOT_LOGX
        metric_for_fit = "time"
    else:
        raise ValueError("METRIC must be 'flops' or 'time'.")

    # Fit
    ref_curve, smoothed_curve, summary, best_norm = fit_mix_lognorm_shift(
        v_fit, k_range=K_RANGE, use_ecdf=USE_ECDF_FOR_FIT,
        metric=metric_for_fit, flops_per_sec=FLOPS_PER_SEC, norm_info=norm_info
    )

    # Report
    print("\n=== Best candidates (sorted by penalized score, ks, bic, sse) ===")
    cols = ["model","note","k","loc","sse","ks","bic","small_count","score"]
    print(summary[cols].head(12).to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    # Save TXT (normalized, normalization meta, converted)
    with open(OUT_TXT, "w") as f:
        f.write(f"Best (normalized space): {best_norm['model']}  note={best_norm['note']}  loc_norm={best_norm['loc']}\n")
        for i,(wi,mi,si) in enumerate(zip(best_norm["w"], best_norm["mu"], best_norm["sigma"]), 1):
            f.write(f" comp{i}: w={wi:.6f}, mu_log_norm={mi:.6f}, sigma={si:.6f}, exp(mu_norm)={np.exp(mi):.6g}\n")
        f.write("\nNormalization:\n")
        t = norm_info.get("type","none")
        if t == "affine":
            f.write(f" type=affine  desc={norm_info.get('desc','')}\n a={norm_info['a']}\n b={norm_info['b']}\n")
        elif t == "scale":
            f.write(f" type=scale   desc={norm_info.get('desc','')}\n s={norm_info['s']}\n")
        else:
            f.write(" type=none\n")

        best_units = params_from_norm_to_units(best_norm, norm_info)
        f.write("\nConverted to original units:\n")
        f.write(f" loc={best_units['loc']}\n")
        unit_name = "FLOPs" if metric_for_fit == "flops" else "ms"
        for i,(wi,mi,si) in enumerate(zip(best_units["w"], best_units["mu"], best_units["sigma"]), 1):
            f.write(f" comp{i}: w={wi:.6f}, mu_log={mi:.6f}, sigma={si:.6f}, exp(mu)={np.exp(mi):.6g} {unit_name}\n")
    print(f"\nParameters saved to: {OUT_TXT}")

    # Save JSON (machine-readable)
    best_units = params_from_norm_to_units(best_norm, norm_info)
    payload = {
        "metric": METRIC,
        "flops_per_sec": FLOPS_PER_SEC,
        "normalization": norm_info,            # contains type + (a,b) or s
        "best_norm": {
            "k": int(best_norm["k"]),
            "loc": float(best_norm["loc"]),
            "w": np.asarray(best_norm["w"], float).tolist(),
            "mu": np.asarray(best_norm["mu"], float).tolist(),
            "sigma": np.asarray(best_norm["sigma"], float).tolist(),
        },
        "best_units": {
            "loc": float(best_units["loc"]),
            "w": np.asarray(best_units["w"], float).tolist(),
            "mu": np.asarray(best_units["mu"], float).tolist(),
            "sigma": np.asarray(best_units["sigma"], float).tolist(),
        },
    }
    with open(OUT_JSON, "w") as jf:
        json.dump(payload, jf, indent=2)
    print(f"Parameters (JSON) saved to: {OUT_JSON}")

    # Figures (in fit units)
    plot_model(v_fit, ref_curve, smoothed_curve, best_norm, out_path=OUT_FIG_MODEL, logx=logx_flag, x_label=x_label)
    print(f"Figure saved to:      {OUT_FIG_MODEL}")

    n_sim = max(N_SIM_MIN, 5 * len(v_fit))
    plot_fit_and_simulation(v_fit, ref_curve, best_norm, n_sim=n_sim,
                            out_path=OUT_FIG_COMPARE, clip_to_real_xlim=True,
                            truncate_and_renormalize=False, logx=logx_flag, x_label=x_label)
    print(f"Comparison figure saved to: {OUT_FIG_COMPARE}")

    # Summary stats in ORIGINAL units (for interpretability)
    rng = np.random.default_rng(RANDOM_STATE)
    comp = rng.choice(len(best_norm["w"]), size=n_sim, p=best_norm["w"]/np.sum(best_norm["w"]))
    z = rng.normal(best_norm["mu"][comp], best_norm["sigma"][comp])
    sim_fit = best_norm["loc"] + np.exp(z)
    sim_units = inv_transform_samples(sim_fit, norm_info)
    v_units_report = inv_transform_samples(v_fit, norm_info)

    print("\n=== Summary statistics (original units) ===")
    if metric_for_fit == "flops":
        fmt = lambda a: f"{a:.3e}"
        print(f"Real:      mean={fmt(np.mean(v_units_report))} FLOPs, median={fmt(np.median(v_units_report))}, std={fmt(np.std(v_units_report, ddof=1))}")
        print(f"Simulated: mean={fmt(np.mean(sim_units))} FLOPs, median={fmt(np.median(sim_units))}, std={fmt(np.std(sim_units, ddof=1))}")
    else:
        print(f"Real:      mean={np.mean(v_units_report):.1f} ms, median={np.median(v_units_report):.1f} ms, std={np.std(v_units_report, ddof=1):.1f} ms")
        print(f"Simulated: mean={np.mean(sim_units):.1f} ms, median={np.median(sim_units):.1f} ms, std={np.std(sim_units, ddof=1):.1f} ms")

if __name__ == "__main__":
    main()
