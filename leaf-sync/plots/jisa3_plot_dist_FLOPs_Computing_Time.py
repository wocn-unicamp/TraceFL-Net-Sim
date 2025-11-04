import re
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== CONSTANTS  =====================
PARAMS_FILE        = "params/mix_lognorm_shift_params.txt"
NUM_CLIENTS        = 30
NUM_SIMULATIONS    = 1000
SEED               = 123
APPLY_TRUNCATION   = False   # True: aplica upper_trunc; False: ignora truncamiento

# ================= CAPACITY DISTRIBUTION PARAMS ==============
CAP_DIRAC_MEAN     = 1.0    # GFLOPs/s

CAP_GAUSS_MEAN     = 1.0    # GFLOPs/s
CAP_GAUSS_STD      = 0.2    # GFLOPs/s

CAP_EXP_MEAN       = 1      # GFLOPs/s

CAP_BIMODAL_MEAN1  = 0.75   # GFLOPs/s
CAP_BIMODAL_MEAN2  = 1.25   # GFLOPs/s
CAP_BIMODAL_STD    = 0.10   # GFLOPs/s
CAP_BIMODAL_WEIGHT = 0.5

CAP_LOGN_MU        = np.log(1.0)  # media del log
CAP_LOGN_SIGMA     = 0.35         # desvío del log (ajusta a gusto)

CAP_WEIBULL_SHAPE   = 2.0     # k (forma). Con k=2 y escala=1, F(2)≈0.982
CAP_WEIBULL_SCALE   = 1.0     # λ (escala) en GFLOPs/s

CAP_BETA_A          = 2.0     # parámetros de forma α
CAP_BETA_B          = 5.0     # parámetros de forma β
CAP_BETA_SCALE      = 2.0     # soporte [0, CAP_BETA_SCALE] GFLOPs/s (CDF=1 en x=2)

# Límites para capacidades (clipping)
CLIP_MIN           = 0.01
CLIP_MAX           = 10

# ===================== MONTE CARLO SEEDS =====================
SEED_FLOPS = 123
SEED_CAPS  = 456

# =================== PLOTTING CONSTANTS  =====================
OUT_DIR = "figures/monte_carlo"
DPI     = 300
BIN_W   = 0.2     # Seconds, fixed-width histogram bars
BIN_W_CAP = 0.025  # GFLOPs/s, fixed-width histogram bars for capacities

# =================== PLOTTING CONSTANTS  =====================
OUT_DIR   = "figures/monte_carlo"
DPI       = 300
BIN_W     = 0.2
BIN_W_CAP = 0.025

# Tamaño fijo de figura
FIG_WIDTH_IN  = 7
FIG_HEIGHT_IN = 5

# ---- NUEVAS: fuentes globales ----
FONT_TITLE   = 20
FONT_LABEL   = 18
FONT_TICK    = 18
FONT_LEGEND  = 18
FONT_ANNOT   = 16   # (cuadro de hiperparámetros)

# ---- NUEVAS: límites de eje X por tipo de figura ----
XLIM_TIME      = (0, 10)   # segundos
XLIM_CAPACITY  = (0,  2)   # GFLOPs/s
XLIM_LOAD      = (0, 15)   # segundos (bins)


# =================== LOAD MONTE CARLO CONSTANTS ===================
MODEL_SIZE_MBYTES = 26.0   # Tamaño del modelo (MBytes) para calcular carga
BIN_SIZE_S        = 1.0    # Segundos, tamaño del bin para carga (1 s)

def _figsize():
    return (FIG_WIDTH_IN, FIG_HEIGHT_IN)

# ========================= CORE I/O =========================
def load_mixture3_params(path: str):
    float_re = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    txt = open(path, "r", encoding="utf-8").read()

    m_loc = re.search(rf"\bloc\s*=\s*({float_re})", txt)
    if not m_loc:
        raise ValueError("Could not find 'loc = ...' in the parameter file.")
    loc = float(m_loc.group(1))

    comps = re.findall(
        rf"w\s*=\s*({float_re}).*?mu_log\s*=\s*({float_re}).*?sigma\s*=\s*({float_re})",
        txt, flags=re.S
    )
    if len(comps) < 3:
        raise ValueError("Expected at least 3 mixture components (w, mu_log, sigma).")

    w, mu_log, sigma = [], [], []
    for wi, mui, si in comps[:3]:
        w.append(float(wi)); mu_log.append(float(mui)); sigma.append(float(si))

    w = np.asarray(w, dtype=float)
    mu_log = np.asarray(mu_log, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Mixture weights must sum to a positive value.")
    w = w / w_sum

    params = dict(loc=float(loc), w=w, mu_log=mu_log, sigma=sigma)

    m_trunc = re.search(rf"\bupper_trunc\s*=\s*({float_re})", txt)
    if m_trunc:
        params["upper_trunc"] = float(m_trunc.group(1))
    return params


# ========================= SAMPLERS =========================

def trunc_rand(flops, upper, on=True, rng=None):
    if not (on and upper is not None): return flops
    rng = np.random.default_rng() if rng is None else rng
    u = np.asarray(upper); m = flops > u
    if not m.any(): return flops
    lo = (0.8*u[m] if np.ndim(u) else 0.8*u) * (1+np.finfo(float).eps)
    hi =  (   u[m] if np.ndim(u) else    u)
    flops[m] = rng.uniform(lo, hi, m.sum())
    return flops

def sample_flops_clients(params: dict, C: int, n: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w   = np.asarray(params["w"], dtype=float)
    mu  = np.asarray(params["mu_log"], dtype=float)
    sig = np.asarray(params["sigma"], dtype=float)
    loc = float(params["loc"])

    if w.shape[0] != 3 or mu.shape[0] != 3 or sig.shape[0] != 3:
        raise ValueError("This function expects exactly 3 mixture components.")

    comp  = rng.choice(3, size=(n, C), p=w)
    z     = rng.normal(loc=mu[comp], scale=sig[comp])
    flops = loc + np.exp(z)

    upper = params.get("upper_trunc", None)
    if APPLY_TRUNCATION and (upper is not None):
        trunc_rand(flops, upper, on=True, rng=rng)
    return flops


def sample_capacity_clients(dist: str, C: int, n: int, seed: int | None = None) -> np.ndarray:
    """
    Muestrea capacidades por cliente según 'dist' y elimina (rechazo + re-muestreo)
    cualquier valor > 2.0 GFLOPs/s. Mantiene la forma (n, C) y la reproducibilidad.
    """
    rng = np.random.default_rng(seed)
    MAX_CAP = 2.0  # GFLOPs/s
    SAFETY_MAX_TRIES = 50

    # --- helpers por distribución (para re-muestreo de un número arbitrario de puntos) ---
    def draw(dist_name: str, size: int) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=float)

        if dist_name == "dirac":
            return np.full(size, CAP_DIRAC_MEAN, dtype=float)

        elif dist_name == "gaussian":
            return rng.normal(loc=CAP_GAUSS_MEAN, scale=CAP_GAUSS_STD, size=size)

        elif dist_name == "exponential":
            return rng.exponential(scale=CAP_EXP_MEAN, size=size)

        elif dist_name == "bimodal":
            m = rng.random(size) < CAP_BIMODAL_WEIGHT
            out = np.empty(size, dtype=float)
            out[m]  = rng.normal(loc=CAP_BIMODAL_MEAN1, scale=CAP_BIMODAL_STD, size=m.sum())
            out[~m] = rng.normal(loc=CAP_BIMODAL_MEAN2, scale=CAP_BIMODAL_STD, size=(~m).sum())
            return out

        elif dist_name == "lognormal":
            # np.random.lognormal usa mu y sigma del LOG
            return rng.lognormal(mean=CAP_LOGN_MU, sigma=CAP_LOGN_SIGMA, size=size)

        elif dist_name == "weibull":
            # rng.weibull(a=k) tiene escala 1 → escalamos por λ
            return CAP_WEIBULL_SCALE * rng.weibull(a=CAP_WEIBULL_SHAPE, size=size)

        elif dist_name == "beta":
            # Soporte [0, CAP_BETA_SCALE] (típicamente ≤ 2 ya)
            return rng.beta(CAP_BETA_A, CAP_BETA_B, size=size) * CAP_BETA_SCALE

        else:
            raise ValueError(
                "Unknown dist. Use one of: "
                "'dirac', 'gaussian', 'exponential', 'bimodal', 'weibull', 'beta', 'lognormal'."
            )

    # --- muestreo inicial en forma (n, C) ---
    if dist == "dirac":
        caps = np.full((n, C), CAP_DIRAC_MEAN, dtype=float)
    elif dist == "gaussian":
        caps = rng.normal(loc=CAP_GAUSS_MEAN, scale=CAP_GAUSS_STD, size=(n, C))
    elif dist == "exponential":
        caps = rng.exponential(scale=CAP_EXP_MEAN, size=(n, C))
    elif dist == "bimodal":
        mask = rng.random((n, C)) < CAP_BIMODAL_WEIGHT
        caps = np.empty((n, C), dtype=float)
        caps[mask]  = rng.normal(loc=CAP_BIMODAL_MEAN1, scale=CAP_BIMODAL_STD, size=mask.sum())
        caps[~mask] = rng.normal(loc=CAP_BIMODAL_MEAN2, scale=CAP_BIMODAL_STD, size=(~mask).sum())
    elif dist == "lognormal":
        caps = rng.lognormal(mean=CAP_LOGN_MU, sigma=CAP_LOGN_SIGMA, size=(n, C))
    elif dist == "weibull":
        caps = CAP_WEIBULL_SCALE * rng.weibull(a=CAP_WEIBULL_SHAPE, size=(n, C))
    elif dist == "beta":
        caps = rng.beta(CAP_BETA_A, CAP_BETA_B, size=(n, C)) * CAP_BETA_SCALE
    else:
        raise ValueError(
            "Unknown dist. Use one of: "
            "'dirac', 'gaussian', 'exponential', 'bimodal', 'weibull', 'beta', 'lognormal'."
        )

    # --- clipping mínimo por robustez (evita 0) ---
    if CLIP_MIN is not None:
        np.maximum(caps, CLIP_MIN, out=caps)

    # --- RECHAZO + RE-MUESTREO para eliminar valores > MAX_CAP ---
    tries = 0
    over = caps > MAX_CAP
    while np.any(over) and tries < SAFETY_MAX_TRIES:
        k = over.sum()
        caps[over] = draw(dist, k)
        if CLIP_MIN is not None:
            np.maximum(caps, CLIP_MIN, out=caps)
        over = caps > MAX_CAP
        tries += 1

    # Fallback de seguridad: si aún queda algún valor marginal > MAX_CAP, ajústalo al float inmediatamente menor que MAX_CAP
    if np.any(over):
        caps[over] = np.nextafter(MAX_CAP, 0.0)

    # --- (opcional) clipping superior final para garantizar [CLIP_MIN, MAX_CAP] ---
    np.clip(caps, CLIP_MIN if CLIP_MIN is not None else -np.inf, MAX_CAP, out=caps)

    return caps




def compute_times_seconds(flops: np.ndarray, caps_gflops: np.ndarray) -> np.ndarray:
    caps_eff = np.maximum(caps_gflops, 1e-12)
    return flops / (caps_eff * 1e9)


# ===================== ECDF / PDF (CÁLCULO) =====================
def compute_ecdf(x: np.ndarray):
    xs = np.sort(np.asarray(x, float))
    n  = xs.size
    Fn = np.arange(1, n + 1, dtype=float) / n
    return xs, Fn


def compute_hist_prob(x: np.ndarray,
                      bin_width: float,
                      lo: float | None = None,
                      hi: float | None = None,
                      q_hi: float | None = 0.999,
                      max_bins: int = 5000):
    """
    Histograma robusto (prob por bin). Controla colas extremas:
    - si hi no se pasa, usa cuantil superior q_hi;
    - limita el nº de bins a max_bins.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]  # por si acaso

    # Inferir límites
    if lo is None:
        lo = np.floor(np.nanmin(x) / bin_width) * bin_width
    if hi is None:
        if q_hi is not None:
            hi = np.quantile(x, q_hi)
        else:
            hi = np.nanmax(x)
        hi = np.ceil(hi / bin_width) * bin_width

    # Limitar nº de bins
    n_bins = int(np.ceil((hi - lo) / bin_width))
    if n_bins > max_bins:
        hi = lo + max_bins * bin_width
        n_bins = max_bins

    if hi <= lo:
        hi = lo + bin_width
        n_bins = 1

    edges = np.arange(lo, hi + bin_width, bin_width)
    counts, _ = np.histogram(x, bins=edges)
    total = counts.sum()
    probs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    return edges, probs



def ks_distance_empirical(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample KS statistic (empirical, no SciPy)."""
    x = np.sort(np.asarray(x, float))
    y = np.sort(np.asarray(y, float))
    n, m = x.size, y.size
    i = j = 0
    cdf_x = cdf_y = 0.0
    d = 0.0
    while i < n and j < m:
        if x[i] <= y[j]:
            i += 1
            cdf_x = i / n
        else:
            j += 1
            cdf_y = j / m
        d = max(d, abs(cdf_x - cdf_y))
    # drain the tails
    while i < n:
        i += 1; cdf_x = i / n; d = max(d, abs(cdf_x - cdf_y))
    while j < m:
        j += 1; cdf_y = j / m; d = max(d, abs(cdf_x - cdf_y))
    return float(d)


# ========================= PLOTTING =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_time_cdf_pdf(xs_cdf: np.ndarray,
                      Fn_cdf: np.ndarray,
                      hist_edges: np.ndarray,
                      hist_probs: np.ndarray,
                      label: str,
                      out_dir: str = OUT_DIR,
                      dpi: int = DPI,
                      bin_width: float = BIN_W):
    """
    Plot Empirical CDF (left axis) and Estimated PDF (histogram, right axis) on ONE figure.
    This function ONLY plots; it does NOT compute ECDF/PDF.
    """
    ensure_dir(out_dir)
    fig, ax1 = plt.subplots(figsize=_figsize(), layout="constrained")

    ax2 = ax1.twinx()
    ax2.bar(hist_edges[:-1], hist_probs, width=bin_width, align="edge",
            alpha=0.35, label="Estimated PDF")

    ax1.step(xs_cdf, Fn_cdf, where="post", linewidth=1.8, zorder=3, label="Empirical CDF")

    ax1.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    ax1.set_ylabel("Empirical CDF", fontsize=FONT_LABEL)
    ax2.set_ylabel("Estimated PDF", fontsize=FONT_LABEL)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(*XLIM_TIME)
    ax1.grid(True, alpha=0.3)

    # fig.suptitle(f"Time — {label} Function", fontsize=FONT_TITLE)

    # leyenda + tamaños
    lines, labels_ = [], []
    l1 = ax1.get_lines()
    if l1: lines.append(l1[0]); labels_.append(l1[0].get_label())
    bars = ax2.containers
    if bars: lines.append(bars[0]); labels_.append("Estimated PDF")
    ax1.legend(lines, labels_, loc="center right", fontsize=FONT_LEGEND)

    # ticks
    ax1.tick_params(axis="both", labelsize=FONT_TICK)
    ax2.tick_params(axis="both", labelsize=FONT_TICK)

    out_path = os.path.join(out_dir, f"time_{label}.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print("Saved:", out_path)

# === NUEVO: cálculo ECDF + hist para capacidades puras ===
def compute_capacity_ecdf_hist_for_dist(dist: str,
                                        C: int,
                                        n: int,
                                        seed_caps: int,
                                        bin_width: float = BIN_W):
    """
    Muestra capacidades para 'dist', aplana a población, y devuelve
    (xs, Fn, edges, probs) para graficar la ECDF y el histograma (prob por bin).
    """
    caps = sample_capacity_clients(dist, C=C, n=n, seed=seed_caps)
    x = caps.ravel()  # GFLOPs/s
    xs, Fn = compute_ecdf(x)
    edges, probs = compute_hist_prob(x, bin_width=bin_width)
    return xs, Fn, edges, probs



# === Helper: string con hiperparámetros visibles en la figura ===
def format_capacity_hparams(dist: str) -> str:
    f3 = lambda v: f"{float(v):.3f}"
    if dist == "dirac":
        return f"mean={f3(CAP_DIRAC_MEAN)}"
    elif dist == "gaussian":
        return f"mean={f3(CAP_GAUSS_MEAN)}, std={f3(CAP_GAUSS_STD)}"
    elif dist == "exponential":
        return f"mean={f3(CAP_EXP_MEAN)}"
    elif dist == "bimodal":
        return (f"w1={f3(CAP_BIMODAL_WEIGHT)}, "
                f"mean1={f3(CAP_BIMODAL_MEAN1)}, mean2={f3(CAP_BIMODAL_MEAN2)}, "
                f"std={f3(CAP_BIMODAL_STD)}")
    elif dist == "lognormal":
        return f"mu_log={f3(CAP_LOGN_MU)}, sigma={f3(CAP_LOGN_SIGMA)}"
    elif dist == "weibull":
        return f"shape(k)={f3(CAP_WEIBULL_SHAPE)}, scale(λ)={f3(CAP_WEIBULL_SCALE)}"
    elif dist == "beta":
        return f"a={f3(CAP_BETA_A)}, b={f3(CAP_BETA_B)}, scale={f3(CAP_BETA_SCALE)}"
    else:
        return ""

# === ploteo ECDF + hist para capacidades puras ===

def plot_capacity_cdf_pdf(xs_cdf: np.ndarray,
                          Fn_cdf: np.ndarray,
                          hist_edges: np.ndarray,
                          hist_probs: np.ndarray,
                          label: str,                    # ← usamos 'label' como nombre de la dist
                          out_dir: str = OUT_DIR,
                          dpi: int = DPI,
                          bin_width: float = BIN_W):
    """
    Figura individual con ECDF (eje izq.) y histograma como Estimated PDF (eje der.)
    para la distribución de CAPACIDADES puras. Muestra hiperparámetros configurados.
    """
    ensure_dir(out_dir)
    fig, ax1 = plt.subplots(figsize=_figsize(), layout="constrained")

    ax2 = ax1.twinx()
    ax2.bar(hist_edges[:-1], hist_probs, width=bin_width, align="edge",
            alpha=0.35, label="Estimated PDF")

    ax1.step(xs_cdf, Fn_cdf, where="post", linewidth=1.8, zorder=3, label="Empirical CDF")

    ax1.set_xlabel("Capacity (GFLOPs/s)", fontsize=FONT_LABEL)
    ax1.set_ylabel("Empirical CDF", fontsize=FONT_LABEL)
    ax2.set_ylabel("Estimated PDF", fontsize=FONT_LABEL)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(*XLIM_CAPACITY)
    ax1.grid(True, alpha=0.3)

    # fig.suptitle(f"Capacity — {label} Function", fontsize=FONT_TITLE)

    # cuadro con hiperparámetros
    hp = format_capacity_hparams(label)
    if hp:
        ax1.text(0.01, 0.99, hp, transform=ax1.transAxes,
                 ha="left", va="top",
                 fontsize=FONT_ANNOT, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, lw=0.5))

    lines, labels_ = [], []
    l1 = ax1.get_lines()
    if l1: lines.append(l1[0]); labels_.append(l1[0].get_label())
    bars = ax2.containers
    if bars: lines.append(bars[0]); labels_.append("Estimated PDF")
    ax1.legend(lines, labels_, loc="center right", fontsize=FONT_LEGEND)

    ax1.tick_params(axis="both", labelsize=FONT_TICK)
    ax2.tick_params(axis="both", labelsize=FONT_TICK)

    out_path = os.path.join(out_dir, f"capacity_{label}.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print("Saved:", out_path)



# ===================== MONTE CARLO FOR LOAD (Mbps) =====================

def compute_load_mbps_matrix(T: np.ndarray,
                             model_size_mbytes: float = 26.0,
                             bin_size_s: float = 1.0,
                             t_max: float | None = None):
    """
    Construye el 'embeb' Monte Carlo de carga por segundo (Mbps) a partir de T.
    T: array 2D (n_sim, C) con tiempos de finalización en segundos para cada cliente.
    
    Para cada simulación i:
      - cuenta cuántos clientes terminan en cada intervalo [k, k+bin_size_s)
      - multiplica por (model_size_mbytes * 8) Mbits por finalización
      - divide por la duración del bin (bin_size_s), quedando en Mbit/s (Mbps)
    
    Devuelve:
      - bin_left_edges: np.ndarray con los bordes izquierdos de cada segundo (0,1,2,...)
      - load_matrix_mbps: np.ndarray de shape (n_sim, n_bins) con la carga por bin en cada simulación
    """
    T = np.asarray(T, float)
    assert T.ndim == 2, "T debe ser 2D (n_sim, C)."
    n_sim, n_clients = T.shape

    # Determinar rango temporal
    t_max_obs = np.nanmax(T)
    if t_max is None:
        t_max = t_max_obs
    n_bins = int(np.ceil(t_max / bin_size_s))
    n_bins = max(n_bins, 1)

    # Bins [0, 1, 2, ...] * bin_size_s
    edges = np.arange(0, (n_bins + 1)) * bin_size_s  # longitud n_bins+1
    bin_left_edges = edges[:-1]                       # longitud n_bins

    # Conversión: MBytes -> Mbits y por segundo (bin_size_s)
    mbit_per_completion = model_size_mbytes * 8.0
    scale = mbit_per_completion / bin_size_s

    # Construir matriz de cargas por simulación
    load_matrix = np.zeros((n_sim, n_bins), dtype=float)
    for i in range(n_sim):
        # Conteo por bin para la simulación i (sobre sus C tiempos)
        counts, _ = np.histogram(T[i, :], bins=edges)  # cuenta en [k, k+1)
        load_matrix[i, :] = counts * scale

    return bin_left_edges, load_matrix


def summarize_load_monte_carlo(load_matrix: np.ndarray,
                               ci_level: float = 0.95):
    """
    Resume el embeb Monte Carlo por bin:
      - media por bin
      - intervalo de confianza (normal approx) por bin
    
    Devuelve:
      - mean_mbps: promedio por segundo (shape n_bins,)
      - ci_low_mbps, ci_high_mbps: límites inferior y superior (shape n_bins,)
      - std_mbps: desvío estándar por bin (shape n_bins,)
    """
    load_matrix = np.asarray(load_matrix, float)
    assert load_matrix.ndim == 2, "load_matrix debe ser 2D (n_sim, n_bins)."
    n_sim = load_matrix.shape[0]

    mean = load_matrix.mean(axis=0)
    std = load_matrix.std(axis=0, ddof=1) if n_sim > 1 else np.zeros_like(mean)

    # Aproximación normal para IC
    from math import erf, sqrt
    # Convertir nivel de confianza a z (inversa de la CDF normal)
    # Para 95% ~ 1.96. Si quieres exacto: usa SciPy. Aquí dejamos 1.96 por defecto.
    if abs(ci_level - 0.95) < 1e-9:
        z = 1.96
    else:
        # fallback simple: aproxima con 1.96 para cualquier ci_level razonable
        z = 1.96

    sem = std / np.sqrt(n_sim) if n_sim > 0 else np.zeros_like(std)
    half = z * sem
    ci_low = mean - half
    ci_high = mean + half
    return mean, ci_low, ci_high, std


def plot_load_bar(bin_left_edges: np.ndarray,
                  mean_mbps: np.ndarray,
                  ci_low_mbps: np.ndarray | None = None,
                  ci_high_mbps: np.ndarray | None = None,
                  bin_size_s: float = 1.0,
                  title_prefix: str = "Mean offered load",
                  label: str | None = None,   # ← nombre de la función/dist (p.ej., "gaussian")
                  out_dir: str = OUT_DIR,
                  dpi: int = DPI,
                  filename: str = "load_monte_carlo_bars.png",
                  x_max_seconds: int | None = None,
                  y_max_mbps: float | None = None):   # ← NUEVO: fuerza ticks hasta este segundo
    """
    Grafica la carga media por segundo en un diagrama de barras.
    Si se proveen ci_low/ci_high, agrega barras de error (IC).
    El título incluye el nombre de la función/distribución si 'label' no es None.

    Parámetros adicionales:
      - x_max_seconds: si se pasa, fuerza los ticks (y el eje X) hasta ese segundo
        aunque no existan barras allí (útil para mostrar 9, 10, etc.).
    """
    ensure_dir(out_dir)

    # Centros de barra (para bins [k, k+bin_size_s) → centro en k+0.5*bin_size_s)
    x = bin_left_edges + bin_size_s * 0.5
    width = bin_size_s * 0.9

    fig, ax = plt.subplots(figsize=_figsize(), layout="constrained")
    ax.bar(x, mean_mbps, width=width, alpha=0.8, label="Mean (Mbps)")

    if ci_low_mbps is not None and ci_high_mbps is not None:
        yerr = np.vstack([mean_mbps - ci_low_mbps, ci_high_mbps - mean_mbps])
        ax.errorbar(x, mean_mbps, yerr=yerr, fmt="none", capsize=3, linewidth=1)

    ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    ax.set_ylabel("Load (Mbit/s)", fontsize=FONT_LABEL)
    title = f"{title_prefix} — {label}" if label else title_prefix
    # ax.set_title(title, fontsize=FONT_TITLE)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=FONT_LEGEND)

    # ----- Ticks 1..x_max_seconds (forzados si se pide) -----
    # Si no se especifica, usamos el último borde derecho observado (p. ej., 8 s)
    if x_max_seconds is None:
        # último borde derecho = último bin_left + bin_size
        last_right = float(bin_left_edges[-1] + bin_size_s) if len(bin_left_edges) else 0.0
        x_max_seconds = max(1, int(np.ceil(last_right)))

    if y_max_mbps is not None:
        last_upper = max(y_max_mbps, mean_mbps.max() if len(mean_mbps) else 0.0)
        y_max_mbps = last_upper * 1.05  

    tick_positions = np.arange(1, x_max_seconds + 1)  # 1,2,...,x_max_seconds
    tick_labels = [str(i) for i in tick_positions]
    try:
        ax.set_xticks(tick_positions, tick_labels)
    except TypeError:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    # Limites para ver también los ticks adicionales (alineados a centros de barra)
    ax.set_xlim(0.5, x_max_seconds + 0.5)

    ax.set_ylim(0, y_max_mbps)

    # Ticks y fuentes
    ax.tick_params(axis="both", labelsize=FONT_TICK)

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print("Saved:", out_path)




# ===================== MAIN SCRIPT =====================
if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    
    # Carga parámetros de FLOPs
    params = load_mixture3_params(PARAMS_FILE)

    # ÚNICA simulación Monte Carlo para FLOPs
    X_flops = sample_flops_clients(params, C=NUM_CLIENTS, n=NUM_SIMULATIONS, seed=SEED_FLOPS)
    print(f"Sampled FLOPs: shape={X_flops.shape}, mean={X_flops.mean():.3f}, std={X_flops.std():.3f}")

    for dist in ("dirac", "gaussian", "exponential", "bimodal", "weibull", "beta", "lognormal"):

        print(f"\nProcessing distribution: {dist}")
        
        #=== ECDF + histograma para tiempos de computación ===
        
        # ÚNICA simulación Monte Carlo para capacidades
        caps = sample_capacity_clients(dist, C=NUM_CLIENTS, n=NUM_SIMULATIONS, seed=SEED_CAPS)

        # Tiempos (segundos) de computación
        T = compute_times_seconds(X_flops, caps) 
        print(f" Computation times: shape={T.shape}, mean={T.mean():.4f} s, std={T.std():.4f} s, max={T.max():.4f} s")

        # Copia para evitar modificar T original
        T_matrix = T.copy()

        # Aplanar a población    
        T = T.ravel()

        # ECDF y histograma (probabilidad por bin fijo de 0.1 s)
        xs, Fn = compute_ecdf(T)
        edges, probs = compute_hist_prob(T, bin_width=BIN_W)

        # Plot and save
        plot_time_cdf_pdf(xs, Fn, edges, probs, label=dist, out_dir=OUT_DIR, dpi=DPI, bin_width=BIN_W)


        # === ECDF + histograma para capacidades puras ===

        xs, Fn, edges, probs = compute_capacity_ecdf_hist_for_dist(
            dist=dist,
            C=NUM_CLIENTS,
            n=NUM_SIMULATIONS,
            seed_caps=SEED_CAPS,
            bin_width=BIN_W_CAP
        )

        # Etiqueta y plot (archivo con nombre diferente al de tiempos)
        label = dist
        plot_capacity_cdf_pdf(
            xs_cdf=xs,
            Fn_cdf=Fn,
            hist_edges=edges,
            hist_probs=probs,
            label=label,
            out_dir=OUT_DIR,
            dpi=DPI,
            bin_width=BIN_W_CAP
        )


        #=== MONTE CARLO PARA CARGA (Mbps) ===

        bin_edges_s, load_matrix = compute_load_mbps_matrix(T_matrix, model_size_mbytes=MODEL_SIZE_MBYTES, bin_size_s=BIN_SIZE_S)

        
        mean_mbps, ci_lo, ci_hi, _ = summarize_load_monte_carlo(load_matrix)
        
        plot_load_bar(
        bin_left_edges=bin_edges_s,
        mean_mbps=mean_mbps,
        ci_low_mbps=ci_lo,
        ci_high_mbps=ci_hi,
        bin_size_s=BIN_SIZE_S,
        label=dist,
        filename=f"load_{dist}.png",
        x_max_seconds=10,      # ← fuerza ticks hasta 10 s
        y_max_mbps=2500
        )

