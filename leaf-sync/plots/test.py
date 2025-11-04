#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re

# ============= CONFIGURACIÓN ============
PARAMS_FILE = "params/minibatch_c_20_mb_0.6_mix_lognorm_shift_params.txt"
OUT_FIG = "mix_hist_ecdf_reconstructed.png"

X_LABEL = "GFLOPs"
Y_LABEL_PDF = "Estimated PDF"
Y_LABEL_CDF = "Cumulative Probability"

X_MIN, X_MAX = 0e9, 5e9
N_POINTS = 1500
COLORS = ["C0", "C1", "C2"]
LABELS = ["Component 1", "Component 2", "Component 3"]

# ============= FUNCIONES AUXILIARES =============

def parse_params_file(file_path):
    """Lee el archivo de parámetros y devuelve diccionario con loc, w, mu, sigma"""
    with open(file_path, "r") as f:
        text = f.read()

    loc = float(re.search(r"loc=([\-0-9\.Ee]+)", text).group(1))
    comps = re.findall(r"comp\d+: w=([0-9\.Ee+-]+), mu_log=([0-9\.Ee+-]+), sigma=([0-9\.Ee+-]+)", text)
    w, mu, sigma = zip(*[(float(a), float(b), float(c)) for a,b,c in comps])
    w = np.array(w) / np.sum(w)  # normalizar pesos
    mu = np.array(mu)
    sigma = np.array(sigma)
    return dict(loc=loc, w=w, mu=mu, sigma=sigma)


def pdf_shifted_lognorm(x, w, mu, sigma, loc):
    """PDF total (mezcla de lognormales desplazadas)"""
    x = np.asarray(x, float)
    mask = x > loc
    pdf = np.zeros_like(x)
    if np.any(mask):
        z = np.log(x[mask] - loc)
        for wi, m, s in zip(w, mu, sigma):
            pdf[mask] += wi * (1.0 / ((x[mask]-loc)*s*np.sqrt(2*np.pi))) * np.exp(-0.5*((z-m)/s)**2)
    return pdf


def pdf_components(x, w, mu, sigma, loc):
    """PDF de cada componente individual"""
    x = np.asarray(x, float)
    mask = x > loc
    pdfs = []
    if np.any(mask):
        z = np.log(x[mask] - loc)
        for wi, m, s in zip(w, mu, sigma):
            pdf = np.zeros_like(x)
            pdf[mask] = wi * (1.0 / ((x[mask]-loc)*s*np.sqrt(2*np.pi))) * np.exp(-0.5*((z-m)/s)**2)
            pdfs.append(pdf)
    return pdfs


def cdf_shifted_lognorm(x, w, mu, sigma, loc):
    """CDF total"""
    x = np.asarray(x, float)
    mask = x > loc
    cdf = np.zeros_like(x)
    if np.any(mask):
        z = np.log(x[mask] - loc)
        for wi, m, s in zip(w, mu, sigma):
            cdf[mask] += wi * stats.norm.cdf((z - m) / s)
    return np.clip(cdf, 0, 1)

# ============= CARGAR PARÁMETROS Y GENERAR FIGURA =============

p = parse_params_file(PARAMS_FILE)
print("[INFO] Parámetros cargados:")
for k,v in p.items():
    if isinstance(v, np.ndarray):
        print(f"  {k} = {v}")
    else:
        print(f"  {k} = {v:.3e}")

x = np.linspace(X_MIN, X_MAX, N_POINTS)

# PDFs individuales y total
pdfs = pdf_components(x, p["w"], p["mu"], p["sigma"], p["loc"])
pdf_total = np.sum(pdfs, axis=0)

# CDF total
cdf_total = cdf_shifted_lognorm(x, p["w"], p["mu"], p["sigma"], p["loc"])

# ============= PLOT =============
fig, ax1 = plt.subplots(figsize=(8,5), dpi=300)
width = (x[1]-x[0])

# Plot PDFs
bottom = np.zeros_like(x)
for i, pdf in enumerate(pdfs):
    ax1.bar(x, pdf, width=width, align="center", color=COLORS[i],
            alpha=0.6, label=LABELS[i], edgecolor="none", bottom=bottom)
    bottom += pdf

# Línea de PDF total
ax1.plot(x, pdf_total, ls="--", color="k", lw=1.5, label="Simulated PDF")

ax1.set_xlabel(X_LABEL)
ax1.set_ylabel(Y_LABEL_PDF)
ax1.set_xlim(X_MIN, X_MAX)
ax1.grid(alpha=0.3)

# CDF (segundo eje)
ax2 = ax1.twinx()
ax2.plot(x, cdf_total, color="k", ls="--", lw=1.8, label="Simulated CDF")
ax2.set_ylim(0, 1)
ax2.set_ylabel(Y_LABEL_CDF)

# Leyenda combinada
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="best", frameon=True)

plt.tight_layout()
plt.savefig(OUT_FIG)
plt.close()
print(f"[OK] Figura guardada en {OUT_FIG}")
