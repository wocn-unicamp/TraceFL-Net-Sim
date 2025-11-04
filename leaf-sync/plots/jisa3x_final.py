import os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ====================== CONFIGURACIÓN GENERAL ======================
PARAMS_FILE     = "params/mix_lognorm_shift_params.txt"
OUT_DIR         = "figures/monte_carlo"
NUM_CLIENTS     = 20
NUM_SIMULATIONS = 100
SEED_FLOPS      = 123
SEED_CAPS       = 456
DPI             = 300
BIN_W           = 0.5
BIN_W_CAP       = 0.025
MODEL_SIZE_MBYTES = 26.0
BIN_SIZE_S        = 1.0
XLIM_TIME         = (0, 16)
XLIM_CAPACITY     = (0, 2)
XLIM_LOAD         = (0, 16)
FONT_LABEL  = 18
FONT_TICK   = 16
FONT_LEGEND = 16

# =================== FUNCIONES AUXILIARES ===================
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def _figsize(): return (7, 5)

# =================== LECTURA DE PARÁMETROS ===================
def load_mixture3_params(path: str):
    txt = open(path, "r", encoding="utf-8").read()
    float_re = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    loc = float(re.search(rf"\bloc\s*=\s*({float_re})", txt).group(1))
    comps = re.findall(rf"w\s*=\s*({float_re}).*?mu_log\s*=\s*({float_re}).*?sigma\s*=\s*({float_re})", txt, re.S)
    w, mu, sig = map(np.array, zip(*[(float(a), float(b), float(c)) for a,b,c in comps[:3]]))
    w /= w.sum()
    upper = re.search(rf"\bupper_trunc\s*=\s*({float_re})", txt)
    return dict(loc=loc, w=w, mu_log=mu, sigma=sig,
                upper_trunc=float(upper.group(1)) if upper else None)

# =================== MUETREO TRUNCADO ===================
def trunc_rand(x, upper, rng):
    mask = x > upper
    if mask.any():
        lo = 0.8 * upper
        x[mask] = rng.uniform(lo, upper, mask.sum())
    return x

def sample_flops_clients(params, C, n, seed):
    rng = np.random.default_rng(seed)
    comp = rng.choice(3, size=(n, C), p=params["w"])
    z = rng.normal(loc=params["mu_log"][comp], scale=params["sigma"][comp])
    flops = params["loc"] + np.exp(z)
    upper = params.get("upper_trunc", None)
    if upper:
        flops = trunc_rand(flops, upper, rng)
    return flops

def sample_capacity_clients(dist: str, C: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    MAX_CAP, MIN_CAP = 1.75, 0.25

    def _draw_base(size):
        if dist == "dirac": return np.full(size, 1.0)
        if dist == "gaussian": return rng.normal(1.0, 0.2, size)
        if dist == "exponential": return rng.exponential(1.0, size)
        if dist == "bimodal":
            mask = rng.random(size) < 0.5
            out = np.empty(size)
            out[mask] = rng.normal(0.5, 0.1, mask.sum())
            out[~mask] = rng.normal(1.5, 0.1, (~mask).sum())
            return out
        if dist == "lognormal": return rng.lognormal(mean=0.0, sigma=0.35, size=size)
        if dist == "weibull":   return rng.weibull(a=2.0, size=size)
        if dist == "beta":      return rng.beta(2.0, 5.0, size=size) * 2.0
        raise ValueError(f"Unknown dist: {dist}")

    total = n * C
    vals = []
    while len(vals) < total:
        x = _draw_base(total)
        x = x[(x >= MIN_CAP) & (x <= MAX_CAP)]
        vals.extend(x)
    return np.array(vals[:total]).reshape(n, C)

# =================== CÁLCULOS ===================
def compute_times_seconds(flops, caps_gflops):
    return flops / (np.maximum(caps_gflops, 1e-12) * 1e9)

def compute_ecdf(x):
    xs = np.sort(x)
    Fn = np.arange(1, len(xs) + 1) / len(xs)
    return xs, Fn

def compute_hist_prob(x, bin_width):
    lo, hi = np.min(x), np.max(x)
    if np.isclose(lo, hi):  # Dirac-like case
        lo -= 0.05 * lo if lo != 0 else 0.05
        hi += 0.05 * hi if hi != 0 else 0.05
    edges = np.arange(lo, hi + bin_width, bin_width)
    counts, _ = np.histogram(x, bins=edges)
    probs = counts / counts.sum()
    return edges, probs

# =================== LOAD (Mbps) ===================
def compute_load_mbps_matrix(T, model_size_mbytes=26.0, bin_size_s=1.0, t_cut=None):
    """Construye matriz Monte Carlo de carga (Mbps), recortando tiempos > t_cut si se indica"""
    n_sim, n_clients = T.shape
    if t_cut is not None:
        T = np.clip(T, None, t_cut)  # truncar los tiempos por encima del percentil 99.9%

    t_max = np.nanmax(T)
    n_bins = int(np.ceil(t_max / bin_size_s))
    edges = np.arange(0, (n_bins + 1)) * bin_size_s
    bin_left_edges = edges[:-1]

    mbit_per_completion = model_size_mbytes * 8.0
    scale = mbit_per_completion / bin_size_s
    load_matrix = np.zeros((n_sim, n_bins))

    for i in range(n_sim):
        counts, _ = np.histogram(T[i, :], bins=edges)
        load_matrix[i, :] = counts * scale

    return bin_left_edges, load_matrix

def summarize_load_monte_carlo(load_matrix):
    mean = load_matrix.mean(axis=0)
    std = load_matrix.std(axis=0, ddof=1)
    n = load_matrix.shape[0]
    z = 1.96
    sem = std / np.sqrt(n)
    ci_low, ci_high = mean - z * sem, mean + z * sem
    return mean, ci_low, ci_high, std

# =================== PLOTS ===================
def _plot_dual_axis(xs, Fn, edges, probs, xlab, xlim, out_path, cutoff=None):
    fig, ax1 = plt.subplots(figsize=_figsize(), layout="constrained")
    ax2 = ax1.twinx()

    # cortar la CDF en el 99.9% pero sin alterar el xlim
    if cutoff is not None:
        mask = xs <= cutoff
        xs = xs[mask]
        Fn = Fn[mask]
        if len(Fn) > 0:
            Fn[-1] = 1.0  # forzar que la curva llegue a 100%

    epdf = probs / np.sum(probs)
    bars = ax2.bar(edges[:-1], epdf, width=np.diff(edges),
                   align="edge", alpha=0.35, color="#ff7f0e",
                   label="Estimated PDF")
    (ecdf_line,) = ax1.step(xs, Fn * 100, where="post",
                            color="#1f77b4", linewidth=2.0,
                            label="Empirical CDF")

    ax1.set_xlabel(xlab, fontsize=FONT_LABEL)
    ax1.set_ylabel("Clients (%)", fontsize=FONT_LABEL)
    ax2.set_ylabel("Clients (Histogram)", fontsize=FONT_LABEL)
    ax1.set_xlim(*xlim)
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax1.legend([ecdf_line, bars], ["Empirical CDF", "Estimated PDF"],
               loc="lower center", bbox_to_anchor=(0.5, 0.98),
               ncol=2, frameon=False, fontsize=FONT_LEGEND)
    ax1.tick_params(labelsize=FONT_TICK)
    ax2.tick_params(labelsize=FONT_TICK)
    ax1.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_time_cdf_pdf(xs, Fn, edges, probs, label, cutoff=None):
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, f"time_{label}.png")
    _plot_dual_axis(xs, Fn, edges, probs, "Time (s)", XLIM_TIME, path, cutoff=cutoff)

def plot_capacity_cdf_pdf(xs, Fn, edges, probs, label):
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, f"capacity_{label}.png")
    _plot_dual_axis(xs, Fn, edges, probs, "Capacity (GFLOPs/s)", XLIM_CAPACITY, path)

def plot_load_bar(bin_left_edges, mean_mbps, ci_low, ci_high,
                  bin_size_s=1.0, label=None, out_dir=OUT_DIR,
                  round_duration=None):
    ensure_dir(out_dir)
    x = bin_left_edges + bin_size_s * 0.5
    width = bin_size_s * 0.9
    peak_load = np.nanmax(mean_mbps)
    avg_load = np.nanmean(mean_mbps)
    rd = float(round_duration) if round_duration is not None else float(x[-1])

    fig, ax = plt.subplots(figsize=_figsize(), layout="constrained")
    ax.bar(x, mean_mbps, width=width, alpha=0.8, color="#1f77b4")
    yerr = np.vstack([mean_mbps - ci_low, ci_high - mean_mbps])
    ax.errorbar(x, mean_mbps, yerr=yerr, fmt="none", capsize=3, color="black")

    ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
    ax.set_ylabel("Load (Mbit/s)", fontsize=FONT_LABEL)
    
    ax.set_ylim(0, 1750)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_TICK)

    xticks = np.arange(2, 22, 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    text = (f"Peak load = {peak_load:.1f} Mbps\n"
            f"Average load = {avg_load:.1f} Mbps\n"
            f"Round duration = {rd:.1f} s")
    ax.text(0.98, 0.97, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=FONT_LEGEND,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    ax.set_xlim(XLIM_LOAD[0], XLIM_LOAD[1])
    plt.savefig(os.path.join(out_dir, f"load_{label}.png"), dpi=DPI)
    plt.close(fig)
    print(f"Saved: load_{label}.png | Peak={peak_load:.2f}, Avg={avg_load:.2f}, Duration={rd:.2f}s")

# =================== MAIN ===================
if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    params = load_mixture3_params(PARAMS_FILE)
    X_flops = sample_flops_clients(params, NUM_CLIENTS, NUM_SIMULATIONS, SEED_FLOPS)

    for dist in ["dirac", "gaussian", "exponential", "bimodal", "weibull", "beta", "lognormal"]:
        print(f"\nProcessing distribution: {dist}")
        caps = sample_capacity_clients(dist, NUM_CLIENTS, NUM_SIMULATIONS, SEED_CAPS)
        T = compute_times_seconds(X_flops, caps)

        xs, Fn = compute_ecdf(T.ravel())
        edges, probs = compute_hist_prob(T.ravel(), BIN_W)

        # corte al 99.9 %
        cutoff_idx = np.searchsorted(Fn, 1.0) # 0.999
        round_duration = float(xs[min(cutoff_idx, len(xs) - 1)])

        # graficar CDF truncada en 99.9 %
        plot_time_cdf_pdf(xs, Fn, edges, probs, dist, cutoff=round_duration)

        # CAPACITY
        xs, Fn = compute_ecdf(caps.ravel())
        edges, probs = compute_hist_prob(caps.ravel(), BIN_W_CAP)
        plot_capacity_cdf_pdf(xs, Fn, edges, probs, dist)

        # LOAD usando solo tiempos <= round_duration
        bin_edges_s, load_matrix = compute_load_mbps_matrix(T, MODEL_SIZE_MBYTES, BIN_SIZE_S, t_cut=round_duration)
        mean_mbps, ci_lo, ci_hi, _ = summarize_load_monte_carlo(load_matrix)
        plot_load_bar(bin_edges_s, mean_mbps, ci_lo, ci_hi,
                      BIN_SIZE_S, label=dist, out_dir=OUT_DIR,
                      round_duration=round_duration)
