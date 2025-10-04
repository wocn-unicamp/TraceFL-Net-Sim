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
    ensure_dir(OUT_DIR)

    # Semillas por escenario
    rng_caps_30   = np.random.default_rng(BASE_SEED + hash(tag) % 10_000)
    rng_caps_many = np.random.default_rng(BASE_SEED + hash(tag) % 10_000 + 1)
    seed_flops_30 = BASE_SEED + hash(tag) % 10_000 + 2
    seed_flops_mc = BASE_SEED + hash(tag) % 10_000 + 3

    # Capacidades (GFLOPs/s)
    caps_30_gflops   = cap_sampler(N_CLIENTS, rng_caps_30)
    caps_many_gflops = cap_sampler(N_SAMPLES_CDF, rng_caps_many)

    # ---- ORDENAMOS LAS BARRAS PARA MOSTRAR LA DISTRIBUCIÓN ----
    if SORT_BARS:
        caps_plot = np.sort(caps_30_gflops)
        x_clients = np.arange(1, N_CLIENTS + 1)  # “rank” del cliente (1..N)
        x_label   = "Client rank (ascending capacity)"
    else:
        caps_plot = caps_30_gflops
        x_clients = np.arange(1, N_CLIENTS + 1)
        x_label   = "Client ID"

    # Tiempos para CDF (Monte Carlo): N_SAMPLES_CDF tareas
    flops_mc  = sample_flops(N_SAMPLES_CDF, params=params, seed=seed_flops_mc)
    time_s_mc = flops_mc / (caps_many_gflops * 1e9)
    x_t, y_t  = ecdf_xy(time_s_mc)

    # Carga por segundo (30 clientes, 1 envío c/u)
    flops_30   = sample_flops(N_CLIENTS, params=params, seed=seed_flops_30)
    time_s_30  = flops_30 / (caps_30_gflops * 1e9)
    edges_sec  = np.arange(0, 5 + 1, 1)          # [0,1),...,[4,5]
    counts, _  = np.histogram(time_s_30, bins=edges_sec)
    load_mbit_s= counts * MBIT_PER_CLIENT
    centers    = edges_sec[:-1] + 0.5

    # Figura (3 subfiguras)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIG_SIZE, layout="constrained")

    # 1) Barras por cliente (ordenadas para ver la distribución)
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

    # 2) CDF del tiempo (s), eje 0–5 s
    ax2.step(x_t, y_t, where="post", lw=3.0, color="1.0")                   # halo
    ax2.step(x_t, y_t, where="post", lw=1.6, ls=(0, (4, 2)), color="0.15", label="Time CDF")
    ax2.set_title("Time CDF (tasks across capacity distribution)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    # 3) Carga por Mbit/s (0–5 s)
    ax3.bar(centers, load_mbit_s, width=0.9, align="center",
            alpha=0.85, edgecolor="none", label=f"{MB_PER_CLIENT} MB × 8 per client")
    ax3.set_title("Per-second load (Mbit/s) for 30 clients")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Load (Mbit/s)")
    ax3.set_xlim(0, 5)
    ax3.set_xticks(np.arange(0, 6, 1))
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.legend(loc="upper right")

    out_path = os.path.join(OUT_DIR, f"capacity_time_and_load_{tag}.png")
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Figura guardada: {out_path}")
    print("  counts 0-1..4-5:", counts, "| load Mbit/s:", load_mbit_s)

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
