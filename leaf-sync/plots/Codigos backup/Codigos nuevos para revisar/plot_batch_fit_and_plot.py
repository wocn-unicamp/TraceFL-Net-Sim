#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch fit, combined plots, combined params TXT, and a global normalized template:
- Per-dataset fit via fit_shifted_lognorm_mixture.py (Mixture of shifted lognormals).
- Three combined figures per normalization mode:
    1) combined_models_normalized.png
    2) combined_models_normalized_with_empirical.png
    3) combined_models_flops_with_empirical.png
- Combined TXT with parameters (normalized & original units).
- Fit a single normalized "template" mixture (shared shape across datasets):
    - template_normalized_params.json / .txt
    - template_normalized_model_with_empirical.png (normalized)
    - template_original_units_with_empirical.png (converted to each dataset's units)
- Optional comparison: run two normalization modes (e.g., minmax vs minmax_robust) and plot a side-by-side panel.

Requirements:
- fit_shifted_lognorm_mixture.py (your earlier module).
- SciPy, NumPy, pandas, matplotlib, scikit-learn.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ndtr  # standard normal CDF
from sklearn.mixture import GaussianMixture

# ---- import your fitter module (previously provided) ----
import fit_shifted_lognorm_mixture as fit

# -------------------- Config --------------------
PATH_FOLDER = "../results/sys/"
file_bases = [
    "sys_metrics_minibatch_c_20_mb_1",
    "sys_metrics_minibatch_c_20_mb_0.9",
    "sys_metrics_minibatch_c_20_mb_0.8",
    "sys_metrics_minibatch_c_20_mb_0.6",
    "sys_metrics_minibatch_c_20_mb_0.5",
    "sys_metrics_minibatch_c_20_mb_0.4",
    "sys_metrics_minibatch_c_20_mb_0.2",
]

# Primary normalization mode and optional comparison set
PRIMARY_NORM_MODE = "minmax"  # baseline
COMPARE_NORM_MODES = True     # set False for a single run
NORM_MODES_TO_COMPARE = ["minmax", "minmax_robust"]

# Template (global normalized model) controls
TEMPLATE_K_RANGE   = (1, 2, 3)
TEMPLATE_REG_COV   = 1e-6
TEMPLATE_SIGMA_MIN = 1e-3
TEMPLATE_RANDOM_STATE = 42
TEMPLATE_MAX_PER_DATASET = None   # e.g., 200000 to cap pooled size; None = no cap

# Paths
OUT_DIR_ROOT = "figures/fit_log"
os.makedirs(OUT_DIR_ROOT, exist_ok=True)

# ----------------- Helpers -----------------
def ecdf_xy(vals):
    x = np.sort(vals)
    n = x.size
    y = np.arange(1, n + 1, dtype=float) / n
    return x, y

def mixture_cdf_lognorm_shift(x, w, mu, sigma, loc):
    """F(x) = sum_i w_i * Phi((log(x - loc) - mu_i) / sigma_i), x>loc; else 0. Vectorized."""
    x = np.asarray(x, dtype=float)
    F = np.zeros_like(x, dtype=float)
    mask = x > loc
    if not np.any(mask):
        return F
    z = np.log(np.clip(x[mask] - loc, 1e-12, None))
    w = np.asarray(w, float); mu = np.asarray(mu, float); sigma = np.asarray(sigma, float)
    t = (z[:, None] - mu[None, :]) / sigma[None, :]
    F[mask] = (w[None, :] * ndtr(t)).sum(axis=1)
    return np.clip(F, 0.0, 1.0)

def to_original_units_from_norm(bn, norm):
    """Convert normalized best params to original units (FLOPs) using fitter's rules."""
    t = norm.get("type", "none")
    loc_norm = float(bn["loc"])
    mu_norm  = np.array(bn["mu"], float)
    w        = np.array(bn["w"], float)
    sigma    = np.array(bn["sigma"], float)
    if t == "affine":
        a, b = float(norm["a"]), float(norm["b"])
        loc = a + b * loc_norm
        mu  = mu_norm + np.log(b)
    elif t == "scale":
        s = float(norm["s"])
        loc = s * loc_norm
        mu  = mu_norm + np.log(s)
    else:
        loc = loc_norm
        mu  = mu_norm
    return float(loc), w, mu, sigma

def apply_same_normalization(v, norm):
    """Map original v (FLOPs) to normalized v' using the normalization metadata from JSON."""
    t = norm.get("type", "none")
    if t == "affine":
        a, b = float(norm["a"]), float(norm["b"])
        return (v - a) / b
    if t == "scale":
        s = float(norm["s"])
        return v / s
    return v

def fit_template_normalized(v_all_norm,
                            k_range=TEMPLATE_K_RANGE,
                            reg_covar=TEMPLATE_REG_COV,
                            sigma_min=TEMPLATE_SIGMA_MIN,
                            random_state=TEMPLATE_RANDOM_STATE):
    """
    Fit a shifted-lognormal mixture on pooled normalized samples.
    - Two hyperparameters are searched: 'loc' over a grid and 'k' in k_range.
    - Selection by SSE vs ECDF reference, tie-break by KS then BIC.
    """
    v = np.asarray(v_all_norm, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 100:
        raise ValueError("Not enough normalized samples to fit the template.")

    x_ref, y_ref = ecdf_xy(v)
    vmin = float(v.min())
    # Search loc below vmin to ensure x-loc > 0; range ~[vmin-1, vmin-0.05]
    loc_grid = np.linspace(vmin - 1.0, vmin - 0.05, 16)

    best = None
    records = []

    for loc in loc_grid:
        u = v - loc
        if np.any(u <= 0):
            continue
        ylog = np.log(u).reshape(-1, 1)

        for k in k_range:
            gmm = GaussianMixture(
                n_components=k, covariance_type="diag",
                random_state=random_state, n_init=5, reg_covar=reg_covar
            ).fit(ylog)

            w = gmm.weights_.ravel()
            mu = gmm.means_.ravel()
            sg = np.sqrt(gmm.covariances_.ravel())
            sg = np.maximum(sg, sigma_min)

            Fm = mixture_cdf_lognorm_shift(x_ref, w, mu, sg, loc)
            res = Fm - y_ref
            sse = float(np.sum(res**2))
            ks  = float(np.max(np.abs(res)))
            bic = float(gmm.bic(ylog))

            rec = dict(loc=loc, w=w, mu=mu, sigma=sg, k=k, sse=sse, ks=ks, bic=bic)
            records.append(rec)
            if (best is None) or (sse < best["sse"]) or (sse == best["sse"] and (ks < best["ks"] or (ks == best["ks"] and bic < best["bic"]))):
                best = rec

    if best is None:
        raise RuntimeError("Template fit failed: no valid loc produced.")

    summary = pd.DataFrame(records).sort_values(["sse", "ks", "bic"]).reset_index(drop=True)
    return summary, best, (x_ref, y_ref)

# --------------- One full pass (given a normalization mode) ---------------
def run_pass(norm_mode, color_map=None):
    """
    Run the whole batch for a given normalization mode.
    Returns a dict with:
      - mode, out_dir, per_ds (list of dicts with params, ECDFs, and v_norm)
      - xgrid ranges used (normalized/original)
      - template file paths
    """
    out_dir = os.path.join(OUT_DIR_ROOT, f"nm_{norm_mode}")
    os.makedirs(out_dir, exist_ok=True)

    # Configure fitter
    fit.PATH_FOLDER    = PATH_FOLDER
    fit.METRIC         = "flops"
    fit.NORMALIZE_MODE = norm_mode

    per_ds = []
    global_flops_min_pos = None
    global_flops_q99 = None
    global_norm_min = None
    global_norm_q995 = None

    for base in file_bases:
        # Unique outputs per file
        fit.NAME_FILE        = base
        fit.OUT_TXT          = os.path.join(out_dir, f"{base}_params.txt")
        fit.OUT_JSON         = os.path.join(out_dir, f"{base}_params.json")
        fit.OUT_FIG_MODEL    = os.path.join(out_dir, f"{base}_fit.png")
        fit.OUT_FIG_COMPARE  = os.path.join(out_dir, f"{base}_fit_and_sim.png")

        # Run the per-dataset fit (writes TXT/JSON/figures)
        fit.main()

        # Read dataset CSV once
        csv_path = os.path.join(PATH_FOLDER, base + ".csv")
        df = pd.read_csv(csv_path)

        # Robustly find FLOPs column
        flops_col = None
        for c in df.columns:
            if "flop" in c.lower():
                flops_col = c
                break
        if flops_col is None:
            flops_col = df.columns[-1]

        v = df[flops_col].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            print(f"[WARN] No positive FLOPs in {base}, skipping.")
            continue

        # Load model JSON
        with open(fit.OUT_JSON, "r") as f:
            P = json.load(f)
        norm = P["normalization"]
        bn   = P["best_norm"]

        # Original-units model params
        loc_u, w_u, mu_u, sigma_u = to_original_units_from_norm(bn, norm)

        # Empirical ECDF in original units
        x_emp_u, y_emp_u = ecdf_xy(v)

        # Global original-units range for shared x-grid
        v_min_pos = float(np.min(v))
        v_q99     = float(np.quantile(v, 0.99))
        global_flops_min_pos = v_min_pos if global_flops_min_pos is None else min(global_flops_min_pos, v_min_pos)
        global_flops_q99     = v_q99     if global_flops_q99     is None else max(global_flops_q99, v_q99)

        # Normalized data using EXACT normalization the fitter used
        v_norm = apply_same_normalization(v, norm)
        x_emp_n, y_emp_n = ecdf_xy(v_norm)

        # Track global normalized range
        vnorm_min = float(np.min(v_norm))
        vnorm_q99 = float(np.quantile(v_norm, 0.995))
        global_norm_min  = vnorm_min if global_norm_min is None else min(global_norm_min, vnorm_min)
        global_norm_q995 = vnorm_q99 if global_norm_q995 is None else max(global_norm_q995, vnorm_q99)

        # Store everything (including raw normalized vector for template pooling)
        per_ds.append(dict(
            base=base,
            norm=norm,
            v_norm=v_norm,                        # <— important for template pooling
            # normalized params to plot model in normalized space
            loc_n=float(bn["loc"]),
            w_n=np.array(bn["w"], float),
            mu_n=np.array(bn["mu"], float),
            sigma_n=np.array(bn["sigma"], float),
            # original-units params to plot model in FLOPs
            loc_u=loc_u, w_u=w_u, mu_u=mu_u, sigma_u=sigma_u,
            # empirical (original & normalized)
            x_emp_u=x_emp_u, y_emp_u=y_emp_u,
            x_emp_n=x_emp_n, y_emp_n=y_emp_n,
        ))

    if not per_ds:
        raise SystemExit(f"No datasets were processed for norm_mode={norm_mode}; nothing to plot.")

    # ---- 1) Normalized-space models (best_norm) ----
    min_loc_n = min(d["loc_n"] for d in per_ds)
    global_norm_min = min_loc_n if global_norm_min is None else min(global_norm_min, min_loc_n)
    xmin_n = min(global_norm_min, min_loc_n + 1e-6)
    xmax_n = max(1.3, global_norm_q995 * 1.1)
    xgrid_n = np.linspace(xmin_n, xmax_n, 2000)

    plt.figure(figsize=(11.5, 6))
    for i, d in enumerate(per_ds):
        color = f"C{i % 10}" if color_map is None else color_map[d["base"]]
        Fm = mixture_cdf_lognorm_shift(xgrid_n, d["w_n"], d["mu_n"], d["sigma_n"], d["loc_n"])
        plt.plot(xgrid_n, Fm, lw=2.0, color=color, label=d["base"])
    plt.ylim(0, 1)
    plt.xlim(0, xmax_n)
    plt.xlabel("Normalized units")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Shifted Lognormal Mixture CDF — Normalized (models only) [{norm_mode}]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    out_norm = os.path.join(out_dir, "combined_models_normalized.png")
    plt.tight_layout(); plt.savefig(out_norm, dpi=150); plt.close()
    print(f"[{norm_mode}] Combined normalized models saved to: {out_norm}")

    # ---- 1b) Normalized-space: models + empirical ECDF ----
    plt.figure(figsize=(11.5, 6))
    for i, d in enumerate(per_ds):
        color = f"C{i % 10}" if color_map is None else color_map[d["base"]]
        # model
        Fm = mixture_cdf_lognorm_shift(xgrid_n, d["w_n"], d["mu_n"], d["sigma_n"], d["loc_n"])
        plt.plot(xgrid_n, Fm, lw=2.0, color=color, label=f"{d['base']} (model)")
        # empirical
        plt.step(d["x_emp_n"], d["y_emp_n"], where="post", lw=1.4, color=color, ls="--", alpha=0.7,
                 label=f"{d['base']} (empirical)")
    plt.ylim(0, 1)
    plt.xlim(0, xmax_n)
    plt.xlabel("Normalized units")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Shifted Lognormal Mixture CDF — Normalized (model + empirical) [{norm_mode}]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    out_norm_emp = os.path.join(out_dir, "combined_models_normalized_with_empirical.png")
    plt.tight_layout(); plt.savefig(out_norm_emp, dpi=150); plt.close()
    print(f"[{norm_mode}] Combined normalized (with empirical) saved to: {out_norm_emp}")

    # ---- 2) Original units (FLOPs): models + empirical ECDF ----
    xmin_u = max(1.0, (global_flops_min_pos or 1.0))
    xmax_u = float(global_flops_q99 or (xmin_u * 10.0))
    if xmax_u <= xmin_u:
        xmax_u = xmin_u * 10.0
    xgrid_u = np.geomspace(xmin_u, xmax_u, 1600)

    plt.figure(figsize=(12, 7))
    for i, d in enumerate(per_ds):
        color = f"C{i % 10}" if color_map is None else color_map[d["base"]]
        # Model CDF in original units
        Fm = mixture_cdf_lognorm_shift(xgrid_u, d["w_u"], d["mu_u"], d["sigma_u"], d["loc_u"])
        plt.plot(xgrid_u, Fm, lw=2.0, color=color, label=f"{d['base']} (model)")
        # Empirical ECDF in original units
        plt.step(d["x_emp_u"], d["y_emp_u"], where="post", lw=1.4, color=color, ls="--", alpha=0.7,
                 label=f"{d['base']} (empirical)")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("FLOPs (original units)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Shifted Lognormal Mixture CDF — Original units (model + empirical) [{norm_mode}]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    out_flops = os.path.join(out_dir, "combined_models_flops_with_empirical.png")
    plt.tight_layout(); plt.savefig(out_flops, dpi=150); plt.close()
    print(f"[{norm_mode}] Combined original-units (with empirical) saved to: {out_flops}")

    # ---- 3) Write a combined TXT with parameters ----
    combined_txt = os.path.join(out_dir, "combined_models_params.txt")
    with open(combined_txt, "w") as f:
        f.write("=== Combined shifted-lognormal mixture parameters ===\n")
        f.write(f"Normalization mode used per dataset is recorded below (type + metadata).\n")
        f.write(f"Files included: {len(per_ds)}\n\n")
        for d in per_ds:
            base = d["base"]; norm = d["norm"]; t = norm.get("type", "none"); desc = norm.get("desc", "")
            f.write(f"--- {base} ---\n")
            f.write("Normalization:\n")
            if t == "affine":
                f.write(f" type=affine  desc={desc}\n a={norm['a']}\n b={norm['b']}\n")
            elif t == "scale":
                f.write(f" type=scale   desc={desc}\n s={norm['s']}\n")
            else:
                f.write(" type=none\n")
            # Normalized-space params
            w_n, mu_n, sg_n, loc_n = d["w_n"], d["mu_n"], d["sigma_n"], d["loc_n"]
            k = len(w_n)
            f.write(f"\n[Normalized space]\n k={k}, loc_norm={loc_n}\n")
            for i in range(k):
                f.write(f" comp{i+1}: w={w_n[i]:.6f}, mu_log_norm={mu_n[i]:.6f}, sigma={sg_n[i]:.6f}, exp(mu_norm)={np.exp(mu_n[i]):.6g}\n")
            # Original-units params
            w_u, mu_u, sg_u, loc_u = d["w_u"], d["mu_u"], d["sigma_u"], d["loc_u"]
            f.write(f"\n[Original units (FLOPs)]\n loc={loc_u}\n")
            for i in range(k):
                f.write(f" comp{i+1}: w={w_u[i]:.6f}, mu_log={mu_u[i]:.6f}, sigma={sg_u[i]:.6f}, exp(mu)={np.exp(mu_u[i]):.6g} FLOPs\n")
            f.write("\n")
    print(f"[{norm_mode}] Combined params TXT saved to: {combined_txt}")

    # ================== 4) Global normalized template (one shared shape) ==================
    # Pool normalized samples; optionally cap per dataset to avoid huge arrays
    v_all = []
    for d in per_ds:
        vn = d["v_norm"]
        if TEMPLATE_MAX_PER_DATASET and vn.size > TEMPLATE_MAX_PER_DATASET:
            rng = np.random.default_rng(TEMPLATE_RANDOM_STATE)
            idx = rng.choice(vn.size, size=TEMPLATE_MAX_PER_DATASET, replace=False)
            vn = vn[idx]
        v_all.append(vn)
    v_all = np.concatenate(v_all, axis=0)
    v_all = v_all[np.isfinite(v_all) & (v_all > 0)]

    tmpl_summary, tmpl_best, tmpl_ref = fit_template_normalized(v_all)

    # Save template params (JSON + TXT)
    tmpl_dir = out_dir
    tmpl_json = {
        "mode": norm_mode,
        "template_norm": {
            "loc": float(tmpl_best["loc"]),
            "w": tmpl_best["w"].tolist(),
            "mu": tmpl_best["mu"].tolist(),
            "sigma": tmpl_best["sigma"].tolist(),
            "k": int(tmpl_best["k"]),
            "sse": float(tmpl_best["sse"]),
            "ks": float(tmpl_best["ks"]),
            "bic": float(tmpl_best["bic"]),
        }
    }
    with open(os.path.join(tmpl_dir, "template_normalized_params.json"), "w") as f:
        json.dump(tmpl_json, f, indent=2)

    with open(os.path.join(tmpl_dir, "template_normalized_params.txt"), "w") as f:
        t = tmpl_json["template_norm"]
        f.write(f"Template (normalized) — mode={norm_mode}\n")
        f.write(f" k={t['k']}, loc_norm={t['loc']}\n")
        for i, (wi, mi, si) in enumerate(zip(t["w"], t["mu"], t["sigma"]), 1):
            f.write(f" comp{i}: w={wi:.6f}, mu_log_norm={mi:.6f}, sigma={si:.6f}, exp(mu_norm)={np.exp(mi):.6g}\n")
        f.write(f"SSE={t['sse']:.6f}, KS={t['ks']:.6f}, BIC={t['bic']:.2f}\n")

    # ---- Template figure: normalized (template + empirical ECDFs) ----
    x_ref, _ = tmpl_ref
    t = tmpl_json["template_norm"]
    locN = t["loc"]; wN = np.array(t["w"]); muN = np.array(t["mu"]); sgN = np.array(t["sigma"])

    xmin_n2 = min([d["x_emp_n"].min() for d in per_ds] + [x_ref.min()])
    xmax_n2 = max([d["x_emp_n"].max() for d in per_ds] + [x_ref.max()])
    xgrid_n2 = np.linspace(max(0.0, xmin_n2), xmax_n2, 2000)

    plt.figure(figsize=(11.5, 6))
    Fm = mixture_cdf_lognorm_shift(xgrid_n2, wN, muN, sgN, locN)
    plt.plot(xgrid_n2, Fm, lw=2.5, color="k", label="Template (model)")
    for i, d in enumerate(per_ds):
        color = f"C{i % 10}" if color_map is None else color_map[d["base"]]
        plt.step(d["x_emp_n"], d["y_emp_n"], where="post", lw=1.4, color=color, ls="--", alpha=0.8,
                 label=f"{d['base']} (emp.)")
    plt.ylim(0, 1); plt.xlim(0, xmax_n2)
    plt.xlabel("Normalized units"); plt.ylabel("Cumulative Probability")
    plt.title(f"Global template vs empirical — normalized [{norm_mode}]")
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right", fontsize=8, ncol=2)
    out_tmpl_norm = os.path.join(tmpl_dir, "template_normalized_model_with_empirical.png")
    plt.tight_layout(); plt.savefig(out_tmpl_norm, dpi=150); plt.close()
    print(f"[{norm_mode}] Template (normalized) figure saved to: {out_tmpl_norm}")

    # ---- Template figure: convert template to each dataset's units + empirical ----
    plt.figure(figsize=(12, 7))
    for i, d in enumerate(per_ds):
        color = f"C{i % 10}" if color_map is None else color_map[d["base"]]
        norm = d["norm"]
        # Back-transform template → original units for dataset i
        if norm.get("type") == "affine":
            a, b = float(norm["a"]), float(norm["b"])
            locU = a + b * locN
            muU  = muN + np.log(b)
        elif norm.get("type") == "scale":
            s = float(norm["s"])
            locU = s * locN
            muU  = muN + np.log(s)
        else:
            locU = locN; muU = muN
        # Evaluate on empirical x for a neat overlay
        FmU = mixture_cdf_lognorm_shift(d["x_emp_u"], wN, muU, sgN, locU)
        plt.plot(d["x_emp_u"], FmU, lw=2.0, color=color, label=f"{d['base']} (template→units)")
        plt.step(d["x_emp_u"], d["y_emp_u"], where="post", lw=1.2, color=color, ls="--", alpha=0.7,
                 label=f"{d['base']} (emp.)")

    plt.xscale("log"); plt.ylim(0, 1)
    plt.xlabel("FLOPs (original units)"); plt.ylabel("Cumulative Probability")
    plt.title(f"Global template (converted) vs empirical — original units [{norm_mode}]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    out_tmpl_units = os.path.join(tmpl_dir, "template_original_units_with_empirical.png")
    plt.tight_layout(); plt.savefig(out_tmpl_units, dpi=150); plt.close()
    print(f"[{norm_mode}] Template (original units) figure saved to: {out_tmpl_units}")

    return {
        "mode": norm_mode,
        "out_dir": out_dir,
        "per_ds": per_ds,
        "xgrid_n_range": (xmin_n, xmax_n),
        "xgrid_u_range": (xmin_u, xmax_u),
        "template_files": {
            "json": os.path.join(tmpl_dir, "template_normalized_params.json"),
            "txt":  os.path.join(tmpl_dir, "template_normalized_params.txt"),
            "fig_norm":  out_tmpl_norm,
            "fig_units": out_tmpl_units,
        }
    }

# -------------------- Run(s) --------------------
if COMPARE_NORM_MODES:
    # Stable color mapping per dataset (same color across modes)
    uniq_bases = list(file_bases)
    color_map = {b: f"C{i % 10}" for i, b in enumerate(uniq_bases)}

    results = []
    for mode in NORM_MODES_TO_COMPARE:
        results.append(run_pass(mode, color_map=color_map))

    # ---- Side-by-side comparison: normalized models only (minmax vs minmax_robust) ----
    if len(results) >= 2:
        resA, resB = results[0], results[1]
        perA, perB = resA["per_ds"], resB["per_ds"]

        dictA = {d["base"]: d for d in perA}
        dictB = {d["base"]: d for d in perB}
        shared_bases = [b for b in file_bases if b in dictA and b in dictB]

        _, xmaxA = resA["xgrid_n_range"]
        _, xmaxB = resB["xgrid_n_range"]
        xA = np.linspace(0, xmaxA, 2000)
        xB = np.linspace(0, xmaxB, 2000)

        fig = plt.figure(figsize=(13, 5), layout="constrained")
        ax1, ax2 = fig.subplots(1, 2, sharey=True)

        for b in shared_bases:
            dA = dictA[b]; dB = dictB[b]
            col = color_map[b]
            FA = mixture_cdf_lognorm_shift(xA, dA["w_n"], dA["mu_n"], dA["sigma_n"], dA["loc_n"])
            FB = mixture_cdf_lognorm_shift(xB, dB["w_n"], dB["mu_n"], dB["sigma_n"], dB["loc_n"])
            ax1.plot(xA, FA, lw=2.0, color=col, label=b)
            ax2.plot(xB, FB, lw=2.0, color=col, label=b)

        ax1.set_title(f"Normalized models — {results[0]['mode']}")
        ax2.set_title(f"Normalized models — {results[1]['mode']}")
        for ax, xmax in ((ax1, xmaxA), (ax2, xmaxB)):
            ax.set_ylim(0, 1); ax.set_xlim(0, xmax)
            ax.set_xlabel("Normalized units"); ax.grid(True, alpha=0.3)
        ax1.set_ylabel("Cumulative Probability")
        # One legend for both
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
        out_cmp = os.path.join(OUT_DIR_ROOT, "combined_models_normalized_compare_minmax_vs_minmax_robust.png")
        plt.savefig(out_cmp, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Comparison (normalized, models only) saved to: {out_cmp}")
else:
    # Single pass only (baseline)
    run_pass(PRIMARY_NORM_MODE)
