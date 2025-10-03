import os, re
import numpy as np
import matplotlib.pyplot as plt

# ===================== Config =====================
PARAMS_FILE   = "params/mix_lognorm_shift_params.txt"
FLOPS_PER_SEC = 1e9                      # capacidad del cliente (FLOPs/s)
UPPER_TRUNC   = 5_017_145_700.0          # cota superior (FLOPs)
OUT_CDF_PATH  = "figures/generation/time_cdf.png"
N_SAMPLES     = 10000
SEED          = 123

# ===================== Utilidades =====================
_float = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_params(path=PARAMS_FILE):
    """
    Espera un archivo con formato:
      Best: <modelo>  loc=<valor>
      comp1: w=..., mu_log=..., sigma=..., ...
      comp2: w=..., mu_log=..., sigma=..., ...
      ...
    Devuelve: dict(loc=float, w=np.ndarray, mu=np.ndarray, sig=np.ndarray)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    m = re.search(rf"\bloc\s*=\s*({_float})", text)
    if not m:
        raise ValueError("No se encontró 'loc=' en el archivo de parámetros.")
    loc = float(m.group(1))

    comps = re.findall(rf"w\s*=\s*({_float}).*?mu_log\s*=\s*({_float}).*?sigma\s*=\s*({_float})", text)
    if not comps:
        raise ValueError("No se encontraron líneas de componentes con w, mu_log y sigma.")

    w, mu, sig = [], [], []
    for wi, mi, si in comps:
        w.append(float(wi)); mu.append(float(mi)); sig.append(float(si))
    w = np.asarray(w, dtype=float); mu = np.asarray(mu, dtype=float); sig = np.asarray(sig, dtype=float)
    w = w / w.sum()  # normaliza por robustez
    return dict(loc=loc, w=w, mu=mu, sig=sig)

def sample_flops(n, params, seed=SEED, upper_trunc=UPPER_TRUNC):
    """Muestra FLOPs de la mezcla lognormal desplazada con censura superior."""
    loc = float(params["loc"])
    w   = np.asarray(params["w"],  float)
    mu  = np.asarray(params["mu"], float)
    sig = np.asarray(params["sig"], float)

    rng  = np.random.default_rng(seed)
    comp = rng.choice(len(w), size=n, p=w)
    z    = rng.normal(mu[comp], sig[comp])
    y    = np.exp(z)
    xF   = loc + y
    xF   = np.minimum(xF, upper_trunc)  # censura superior
    return xF

def flops_to_ms(flops, flops_per_sec=FLOPS_PER_SEC):
    return (np.asarray(flops, float) / float(flops_per_sec)) * 1e3

def ecdf_xy(a):
    a = np.sort(np.asarray(a, float))
    n = a.size
    y = np.arange(1, n+1, dtype=float) / n
    return a, y

def plot_time_cdf(t_ms, out_path=OUT_CDF_PATH):
    """Grafica la CDF (ECDF) del tiempo en ms y la guarda en generation/."""
    ensure_dir(os.path.dirname(out_path) or ".")
    x, y = ecdf_xy(t_ms)
    plt.figure(figsize=(8, 4), layout="constrained")
    # ECDF con halo (opcional) + línea principal
    plt.step(x, y, where="post", lw=3.0, color="1.0")                 # halo blanco
    plt.step(x, y, where="post", lw=1.6, ls=(0, (4, 2)), color="0.15", label="Empirical CDF (time)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of simulated time")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.savefig(out_path, dpi=150)
    plt.close()

# ===================== Ejecución =====================
if __name__ == "__main__":
    params = load_params(PARAMS_FILE)
    flops = sample_flops(N_SAMPLES, params=params, seed=SEED, upper_trunc=UPPER_TRUNC)
    t_ms  = flops_to_ms(flops, FLOPS_PER_SEC)

    plot_time_cdf(t_ms, out_path=OUT_CDF_PATH)
    print(f"CDF del tiempo guardada en: {OUT_CDF_PATH}")
