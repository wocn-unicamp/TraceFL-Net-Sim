# Retry with math.erf vectorized (NumPy does not expose erf by default).
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, erf

def sample_lognormal(n, mu_log, sigma_log, loc=0.0, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    return loc + np.exp(mu_log + sigma_log * z)

def cdf_shifted_lognorm(x, mu_log, sigma_log, loc):
    t = np.zeros_like(x, dtype=float)
    mask = x > loc
    u = (np.log(x[mask] - loc) - mu_log) / (sqrt(2.0) * sigma_log)
    # vectorize math.erf for array input
    t[mask] = 0.5 * (1.0 + np.vectorize(erf)(u))
    return t

# ============= PARÁMETROS =============
# mu = 4301039271
# sigma = 1.422196253
# loc = 493848180

mu = 897494661
sigma = 1.384252684
loc = 77301307

N_SAMPLES = 1000
SEED = 42

# Interpretación robusta de mu
if mu > 1_000:  # parece ser 'scale' (SciPy)
    mu_log = np.log(mu)
    interp = "Interpreté mu como 'scale'; usé mu_log = ln(mu)."
else:
    mu_log = float(mu)
    interp = "Interpreté mu como media en log-espacio (mu_log)."

# Muestras
x = sample_lognormal(N_SAMPLES, mu_log, sigma, loc=loc, seed=SEED)
x_sorted = np.sort(x)
ecdf = np.arange(1, N_SAMPLES + 1) / N_SAMPLES

# Teórica
x_grid_min = max(loc + 1e-9, x_sorted[0])
x_grid_max = x_sorted[-1]
x_grid = np.linspace(x_grid_min, x_grid_max, 800)
F_theo = cdf_shifted_lognorm(x_grid, mu_log, sigma, loc)

# Resumen
emp_mean = float(np.mean(x))
emp_std = float(np.std(x, ddof=1))

# Plot
plt.figure(figsize=(7,5))
plt.plot(x_sorted, ecdf, label="Empirical CDF")
plt.plot(x_grid, F_theo, label="Theoretical CDF (shifted lognormal)")
plt.xlabel("x")
plt.ylabel("CDF")
plt.title("Shifted Lognormal CDF\n" + interp +
          f"\nmu_log={mu_log:.5f}, sigma_log={sigma:.5f}, loc={loc}, "
          f"mean≈{emp_mean:.2f}, std≈{emp_std:.2f}")
plt.legend()
plt.tight_layout()

# limite axis x
plt.xlim(0, 4e10)

plt.show()
# out_path = "/mnt/data/cdf_plot.png"
# plt.savefig(out_path, dpi=150)
# out_path
