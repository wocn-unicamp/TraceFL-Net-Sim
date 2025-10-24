import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # solo para la CDF del modelo (normal CDF)
# ------------------- parámetros del modelo -------------------
loc = 714.9380166666666
w   = np.array([0.290092, 0.335257, 0.374651])
mu  = np.array([7.186880, 8.083923, 6.879985])
sig = np.array([0.053885, 0.171617, 0.346232])

# ------------------- muestreador -------------------
def sample_times(n, seed=123):
    rng = np.random.default_rng(seed)
    comp = rng.choice(3, size=n, p=w)           # 1) componente
    z    = rng.normal(mu[comp], sig[comp])      # 2) normal en log-espacio
    y    = np.exp(z)                            # 3) lognormal
    x    = loc + y                              # 4) shift
    return x

# ------------------- utilidades -------------------
def ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True); return path

def ecdf_xy(a):
    a = np.sort(np.asarray(a, float))
    n = a.size
    y = np.arange(1, n+1, dtype=float)/n
    return a, y

def mixture_cdf(x):
    """CDF de la mezcla lognormal desplazada (opcional para overlay)."""
    x = np.asarray(x, float)
    F = np.zeros_like(x, float)
    m = x > loc
    if np.any(m):
        z = np.log(np.clip(x[m] - loc, 1e-12, None))
        for wi, mi, si in zip(w, mu, sig):
            F[m] += wi * stats.norm.cdf((z - mi)/si)
    return np.clip(F, 0.0, 1.0)

# ------------------- demo: truncar al p99.9 + plots -------------------
ensure_dir("figures")

# 1) muestreo (ajusta n si quieres curvas más suaves)
x = sample_times(10000, seed=123)

# 2) umbral de truncamiento en p=99.9
p = 99
q = np.percentile(x, p)

# 3) truncado (descarta valores > q)
x_trunc = x[x <= q]

print(f"Full sample:     mean={x.mean():.1f} ms, median={np.median(x):.1f} ms, max={x.max():.1f} ms")
print(f"Truncated ≤ p{p}: mean={x_trunc.mean():.1f} ms, median={np.median(x_trunc):.1f} ms, max={x_trunc.max():.1f} ms")
print(f"Cut at p{p} = {q:.1f} ms (removed {len(x) - len(x_trunc)} samples)")

# 4) figuras: histograma + CDF (ECDF) con overlay del modelo (opcional)
fig = plt.figure(figsize=(12,5), layout="constrained")
ax_hist, ax_cdf = fig.subplots(1, 2, sharey=False)

# --- Histograma ---
ax_hist.hist(x, bins=60, density=True, alpha=0.25, label="Full sample")
ax_hist.hist(x_trunc, bins=60, density=True, alpha=0.8, label=f"Truncated ≤ p{p}")
ax_hist.axvline(q, ls=":", lw=1.5, label=f"p{p} cut")
ax_hist.set_xlabel("Time (ms)")
ax_hist.set_ylabel("Density")
ax_hist.set_title("Histogram of time (full vs truncated)")
ax_hist.grid(True, alpha=0.3)
ax_hist.legend()

# --- CDF (ECDF) ---
xe, ye = ecdf_xy(x_trunc)
ax_cdf.step(xe, ye, where="post", lw=1.8, label="ECDF (truncated)")
# Overlay CDF del modelo (opcional; comenta si no la quieres)
xfit = np.linspace(xe.min(), xe.max(), 1200)
ax_cdf.plot(xfit, mixture_cdf(xfit), "--", lw=2.0, label="Model CDF")
ax_cdf.axvline(q, ls=":", lw=1.5, label=f"p{p} cut")
ax_cdf.set_xlim(xe.min(), xe.max())
ax_cdf.set_ylim(0, 1)
ax_cdf.set_yticks(np.linspace(0,1,6))  # 0,0.2,...,1
ax_cdf.set_xlabel("Time (ms)")
ax_cdf.set_ylabel("Cumulative Probability")
ax_cdf.set_title(f"ECDF (truncated at p{p})")
ax_cdf.grid(True, alpha=0.3)
ax_cdf.legend(loc="lower right")

plt.savefig("figures/time_hist_and_cdf_truncated.png", dpi=150)
plt.show()
