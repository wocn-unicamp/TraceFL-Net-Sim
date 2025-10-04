import os, re
import numpy as np
import matplotlib.pyplot as plt

# ===================== Config =====================
PARAMS_FILE      = "params/mix_lognorm_shift_params.txt"
N_CLIENTS        = 30
UPPER_TRUNC      = 5_017_145_700.0
MB_PER_CLIENT    = 26.4
MBIT_PER_CLIENT  = MB_PER_CLIENT * 8.0     # 211.2 Mbit
N_SAMPLES_CDF    = 20000                   # para CDF suave
BASE_SEED        = 123
OUT_DIR          = "figures/ge"
FIG_SIZE         = (15, 4.5)
DPI              = 150
SORT_BARS        = True                    # <<--- ORDENAR BARRAS POR CAPACIDAD

# ===================== Utilidades =====================

# ===================== Config (añade esto) =====================
N_ROUNDS        = 100          # número de rondas/iteraciones por cliente
T_MAX_SECONDS   = 5           # ventana [0,1),...,[4,5] s
SHOW_ERRORBARS  = True        # barras de error (desv. estándar entre rondas)
INCLUDE_OVERFLOW_NOTE = True  # anotar cuántas tareas quedan > T_MAX_SECONDS
# ===================== Fin Config =====================


_float = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_params(path=PARAMS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(rf"\bloc\s*=\s*({_float})", text)
    if not m:
        raise ValueError("No se encontró 'loc=' en el archivo de parámetros.")
    loc = float(m.group(1))
    comps = re.findall(rf"w\s*=\s*({_float}).*?mu_log\s*=\s*({_float}).*?sigma\s*=\s*({_float})", text)
    if not comps:
        raise ValueError("No se encontraron componentes con w, mu_log, sigma.")
    w, mu, sig = [], [], []
    for wi, mi, si in comps:
        w.append(float(wi)); mu.append(float(mi)); sig.append(float(si))
    w = np.asarray(w, float); mu = np.asarray(mu, float); sig = np.asarray(sig, float)
    w = w / w.sum()
    return dict(loc=loc, w=w, mu=mu, sig=sig)

def sample_flops(n, params, seed, upper_trunc=UPPER_TRUNC):
    """Muestra FLOPs de la mezcla con censura superior."""
    loc = float(params["loc"])
    w   = np.asarray(params["w"],  float)
    mu  = np.asarray(params["mu"], float)
    sig = np.asarray(params["sig"], float)
    rng  = np.random.default_rng(seed)
    comp = rng.choice(len(w), size=n, p=w)
    z    = rng.normal(mu[comp], sig[comp])
    y    = np.exp(z)
    xF   = loc + y
    return np.minimum(xF, upper_trunc)

def ecdf_xy(a):
    a = np.sort(np.asarray(a, float))
    n = a.size
    y = np.arange(1, n+1, dtype=float) / n
    return a, y

# ===== Distribuciones de capacidad (en GFLOPs/s) =====
def caps_all_equal(n, rng):
    return np.full(n, 1.0, dtype=float)  # 1 GFLOPs/s

def caps_exp_inverted(n, rng, lam=3.0):
    """Exponencial invertida en [0.5, 1.5): más masa cerca de 0.5."""
    U = rng.uniform(0.0, 1.0, size=n)
    frac = (np.exp(-lam*U) - np.exp(-lam)) / (1.0 - np.exp(-lam))
    return 0.5 + frac * 1.0

def caps_gaussian(n, rng, mean=1.0, std=0.2):
    c = rng.normal(loc=mean, scale=std, size=n)
    return np.clip(c, 0.5, 1.5)

def caps_bimodal(n, rng, p=0.5, m1=0.7, s1=0.10, m2=1.3, s2=0.10):
    k = rng.random(n) < p
    c = np.where(k, rng.normal(m1, s1, size=n), rng.normal(m2, s2, size=n))
    return np.clip(c, 0.5, 1.5)

SCENARIOS = [
    ("all_equal",      "All equal (1 GFLOPs/s)",          caps_all_equal),
    ("exp_inverted",   "Inverted exponential (0.5→1.5)",  caps_exp_inverted),
    ("gaussian",       "Gaussian around 1 GFLOPs/s",      caps_gaussian),
    ("bimodal",        "Bimodal (slow 0.7 / fast 1.3)",   caps_bimodal),
]

def plot_for_scenario(params, tag, title, cap_sampler):
    """
    Genera la figura de 3 paneles para un escenario de capacidades:
      (1) Barras de capacidad por cliente (ordenadas opcionalmente),
      (2) CDF de tiempos (Monte Carlo poblacional),
      (3) Carga por segundo (Mbit/s) PROMEDIO sobre N_ROUNDS con IC95%.

    Requiere en el ámbito global:
      - ensure_dir, sample_flops, ecdf_xy
      - OUT_DIR, FIG_SIZE, DPI
      - N_CLIENTS, N_SAMPLES_CDF, MB_PER_CLIENT, MBIT_PER_CLIENT
      - SORT_BARS, BASE_SEED
      - N_ROUNDS, T_MAX_SECONDS, SHOW_ERRORBARS, INCLUDE_OVERFLOW_NOTE
    """
    ensure_dir(OUT_DIR)

    # Semillas deterministas por 'tag' (evitar hash() salado de Python)
    tag_offset = sum(tag.encode("utf-8")) % 10_000
    base = BASE_SEED + tag_offset
    rng_caps_30   = np.random.default_rng(base + 0)
    rng_caps_many = np.random.default_rng(base + 1)
    seed_flops_30 = base + 2
    seed_flops_mc = base + 3

    # Capacidades (GFLOPs/s)
    caps_30_gflops   = cap_sampler(N_CLIENTS, rng_caps_30)        # fijas por cliente a lo largo de las rondas
    caps_many_gflops = cap_sampler(N_SAMPLES_CDF, rng_caps_many)  # para CDF poblacional

    # Barras de capacidad (solo estética de orden)
    if SORT_BARS:
        caps_plot = np.sort(caps_30_gflops)
        x_clients = np.arange(1, N_CLIENTS + 1)
        x_label   = "Client rank (ascending capacity)"
    else:
        caps_plot = caps_30_gflops
        x_clients = np.arange(1, N_CLIENTS + 1)
        x_label   = "Client ID"

    # --------- CDF de tiempos (Monte Carlo poblacional) ----------
    flops_mc  = sample_flops(N_SAMPLES_CDF, params=params, seed=seed_flops_mc)
    time_s_mc = flops_mc / (caps_many_gflops * 1e9)
    x_t, y_t  = ecdf_xy(time_s_mc)

    # --------- MÚLTIPLES RONDAS por cliente ----------
    edges_sec = np.arange(0, T_MAX_SECONDS + 1, 1)   # [0,1),[1,2),...,[T_MAX_SECONDS-1,T_MAX_SECONDS]
    centers   = edges_sec[:-1] + 0.5
    R         = int(N_ROUNDS)

    counts_rounds   = np.zeros((R, edges_sec.size - 1), dtype=int)
    overflow_rounds = np.zeros(R, dtype=int)

    for r in range(R):
        # FLOPs independientes por ronda; capacidades constantes por cliente
        flops_r  = sample_flops(N_CLIENTS, params=params, seed=seed_flops_30 + 1000 * r)
        time_s_r = flops_r / (caps_30_gflops * 1e9)

        c, _ = np.histogram(time_s_r, bins=edges_sec)
        counts_rounds[r]   = c
        overflow_rounds[r] = np.count_nonzero(time_s_r >= edges_sec[-1])

    # Carga por segundo (Mbit/s)
    load_mbit_s_rounds = counts_rounds * MBIT_PER_CLIENT
    mean_load          = load_mbit_s_rounds.mean(axis=0)
    if R > 1:
        std_load  = load_mbit_s_rounds.std(axis=0, ddof=1)
        sem_load  = std_load / np.sqrt(R)
        ci95      = 1.96 * sem_load
    else:
        ci95 = np.zeros_like(mean_load)

    mean_overflow = overflow_rounds.mean()
    frac_overflow = (mean_overflow / N_CLIENTS) if N_CLIENTS else 0.0

    # --------------------- FIGURA ---------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIG_SIZE, layout="constrained")

    # (1) Capacidades por cliente
    ax1.bar(x_clients, caps_plot, width=0.8, align="center",
            alpha=0.85, edgecolor="none", label=f"N={N_CLIENTS}")
    ax1.set_title(f"Capacity per client — {title}")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Capacity (GFLOPs/s)")
    ax1.set_xticks(x_clients)
    if N_CLIENTS > 16:
        for tick in ax1.get_xticklabels():
            tick.set_rotation(90)
    y_max = max(1.6, caps_plot.max() * 1.05)
    ax1.set_ylim(0, y_max)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper left")

    # (2) CDF de tiempos
    ax2.step(x_t, y_t, where="post", lw=3.0, color="1.0")  # halo
    ax2.step(x_t, y_t, where="post", lw=1.6, ls=(0, (4, 2)), color="0.15", label="Time CDF")
    ax2.set_title("Time CDF (tasks across capacity distribution)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_xlim(0, T_MAX_SECONDS)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    # (3) Carga media por segundo + IC95%
    ax3.bar(centers, mean_load, width=0.9, align="center",
            alpha=0.85, edgecolor="none",
            label=f"Mean over {R} rounds ({MB_PER_CLIENT} MB × 8/client)")
    if SHOW_ERRORBARS and R > 1:
        ax3.errorbar(centers, mean_load, yerr=ci95,
                     fmt='none', capsize=3, alpha=0.7, lw=1.2)

    ax3.set_title("Per-second load (Mbit/s) — mean over rounds")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Load (Mbit/s)")
    ax3.set_xlim(0, T_MAX_SECONDS)
    ax3.set_xticks(np.arange(0, T_MAX_SECONDS + 1, 1))
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(loc="upper right")

    if INCLUDE_OVERFLOW_NOTE and mean_overflow > 0:
        ax3.text(0.02, 0.95,
                 f"Avg. tasks > {T_MAX_SECONDS}s per round: {mean_overflow:.2f} "
                 f"({100*frac_overflow:.1f}%)",
                 transform=ax3.transAxes, va="top", ha="left", fontsize=9)

    out_path = os.path.join(OUT_DIR, f"capacity_time_and_load_{tag}_multi.png")
    plt.savefig(out_path, dpi=DPI)
    plt.close()

    # Resumen en consola
    print(f"Figura guardada: {out_path}")
    print("  mean counts per 1s bin:", counts_rounds.mean(axis=0).round(2),
          "| mean load (Mbit/s):", mean_load.round(1))
    print(f"  mean overflow > {T_MAX_SECONDS}s: {mean_overflow:.2f} "
          f"({100*frac_overflow:.1f}% de {N_CLIENTS})")


# ===================== Run =====================
if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    params = load_params(PARAMS_FILE)
    SCENARIOS = [
        ("all_equal",      "All equal (1 GFLOPs/s)",          caps_all_equal),
        ("exp_inverted",   "Inverted exponential (0.5→1.5)",  caps_exp_inverted),
        ("gaussian",       "Gaussian around 1 GFLOPs/s",      caps_gaussian),
        ("bimodal",        "Bimodal (slow 0.7 / fast 1.3)",   caps_bimodal),
    ]
    for tag, title, sampler in SCENARIOS:
        plot_for_scenario(params, tag, title, sampler)
