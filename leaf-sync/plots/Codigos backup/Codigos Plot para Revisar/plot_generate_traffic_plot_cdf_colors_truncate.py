import numpy as np
import matplotlib.pyplot as plt

# --- Mezcla lognormal desplazada ---
loc = 714.9380166666666
w   = np.array([0.290092, 0.335257, 0.374651])
mu  = np.array([7.186880, 8.083923, 6.879985])
sig = np.array([0.053885, 0.171617, 0.346232])

def sample_times(n, seed=123):
    """Muestras de la mezcla y su componente (0,1,2)."""
    rng  = np.random.default_rng(seed)
    comp = rng.choice(3, size=n, p=w)
    z    = rng.normal(mu[comp], sig[comp])
    x    = loc + np.exp(z)
    return x, comp

def sample_times_truncated(n, lower=None, upper=None, seed=123, batch=None):
    """
    Muestras TRUNCADAS de la mezcla en [lower, upper] usando rejection sampling.
    - Si lower/upper es None, no se aplica esa cota.
    """
    if lower is None and upper is None:
        raise ValueError("Especifica lower y/o upper para truncar.")
    rng = np.random.default_rng(seed)
    batch = n if batch is None else int(batch)
    xs, comps = [], []
    while len(xs) < n:
        comp_b = rng.choice(3, size=batch, p=w)
        z_b    = rng.normal(mu[comp_b], sig[comp_b])
        x_b    = loc + np.exp(z_b)
        mask   = np.ones_like(x_b, dtype=bool)
        if lower is not None:
            mask &= (x_b >= lower)
        if upper is not None:
            mask &= (x_b <= upper)
        if mask.any():
            xs.append(x_b[mask])
            comps.append(comp_b[mask])
    x = np.concatenate(xs)[:n]
    comp = np.concatenate(comps)[:n]
    return x, comp

def ecdf_xy(a):
    a = np.sort(np.asarray(a, float))
    n = a.size
    y = np.arange(1, n+1, dtype=float) / n
    return a, y

def plot_hist_with_cdf(x, comp, bins=50, mode="stacked",
                       show_points=False, x_trunc=None, trunc_label=None):
    colors = ["C0", "C1", "C2"]
    labels = ["Log1", "Log2", "Log3"]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    edges = np.histogram_bin_edges(x, bins=bins)
    widths = np.diff(edges)
    B = len(widths)
    lefts = edges[:-1]

    # Densidades por componente
    K = int(comp.max()) + 1
    n = x.size
    dens = []
    for k in range(K):
        cnt, _ = np.histogram(x[comp == k], bins=edges)
        dens.append(cnt / (n * widths))
    dens = np.vstack(dens)

    if mode == "stacked":
        bottom = np.zeros(B)
        for k, (c, lbl) in enumerate(zip(colors, labels)):
            ax1.bar(lefts, dens[k], width=widths, align="edge",
                    bottom=bottom, color=c, alpha=0.6, edgecolor="none",
                    label=f"{lbl} (stacked)")
            bottom += dens[k]
        total = bottom
    elif mode == "grouped":
        g = len(labels)
        for k, (c, lbl) in enumerate(zip(colors, labels)):
            ax1.bar(lefts + (k / g) * widths, dens[k], width=widths / g,
                    align="edge", color=c, alpha=0.8, edgecolor="none",
                    label=f"{lbl} (grouped)")
        total = dens.sum(axis=0)
    else:
        raise ValueError("mode debe ser 'stacked' o 'grouped'.")

    # Densidad total (contorno)
    ax1.step(edges, np.r_[total, total[-1]], where="post",
             lw=1.2, ls="--", color="k", alpha=0.6, label="Total density")

    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3, zorder=0)

    # --- ECDF total (con halo) ---
    ax2 = ax1.twinx()
    xs_all, ys_all = ecdf_xy(x)
    ax2.step(xs_all, ys_all, where="post", lw=3.0, color="1.0", zorder=1)           # halo blanco
    ecdf_line = ax2.step(xs_all, ys_all, where="post", lw=1.3,
                         ls=(0, (4, 2)), color="0.15", zorder=2, label="ECDF (total)")[0]

    # --- ECDF truncada (si se pasa x_trunc) ---
    trunc_line = None
    if x_trunc is not None and len(x_trunc) > 0:
        xt, yt = ecdf_xy(x_trunc)
        ax2.step(xt, yt, where="post", lw=3.0, color="1.0", zorder=3)   # halo blanco
        trunc_line = ax2.step(xt, yt, where="post", lw=1.6, ls="-.",
                              color="C3", zorder=4,
                              label=trunc_label or "ECDF (truncated)")[0]

    # Puntos opcionales
    if show_points:
        order = np.argsort(x)
        comp_sorted = comp[order]
        idx = np.linspace(0, x.size - 1, min(400, x.size), dtype=int)
        ax2.scatter(xs_all[idx], ys_all[idx], s=14,
                    c=np.take(colors, comp_sorted[idx]), alpha=0.9,
                    edgecolors="white", linewidths=0.4, zorder=5)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Cumulative Probability")

    # Leyenda combinada
    h1, l1 = ax1.get_legend_handles_labels()
    extra = [ecdf_line] + ([trunc_line] if trunc_line is not None else [])
    ax1.legend(h1 + extra, l1 + [h.get_label() for h in extra], loc="best")

    plt.tight_layout()
    plt.show()

# --- Ejemplo de uso ---
x, comp = sample_times(3000, seed=123)

# Trunca por intervalo (ajusta a lo que necesites):
lower, upper = 1000, 5000
x_trunc = x[(x >= lower) & (x <= upper)]  # opción 1: truncar la MISMA muestra
# O bien generar nueva muestra YA truncada estadísticamente:
# x_trunc, _ = sample_times_truncated(3000, lower=lower, upper=upper, seed=123)

plot_hist_with_cdf(
    x, comp, bins=60, mode="stacked", show_points=False,
    x_trunc=x_trunc, trunc_label=f"ECDF (truncated [{lower},{upper}])"
)
