import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la mezcla lognormal desplazada
loc = 714.9380166666666
w   = np.array([0.290092, 0.335257, 0.374651])
mu  = np.array([7.186880, 8.083923, 6.879985])
sig = np.array([0.053885, 0.171617, 0.346232])
upper_trunc = 5017.145700  # cota superior

# def sample_times(n, seed=123):
#     """Devuelve muestras x y el índice de componente comp∈{0,1,2} para cada muestra."""
#     rng = np.random.default_rng(seed)
#     comp = rng.choice(3, size=n, p=w)           # componente elegido
#     z    = rng.normal(mu[comp], sig[comp])      # normal en log-espacio
#     x    = loc + np.exp(z)                      # lognormal desplazada
#     return x, comp

def sample_times(n, seed=123):
    rng  = np.random.default_rng(seed)
    comp = rng.choice(3, size=n, p=w)           # 1) componente
    z    = rng.normal(mu[comp], sig[comp])      # 2) normal en log-espacio
    y    = np.exp(z)                            # 3) lognormal
    x    = loc + y                              # 4) shift
    x    = np.minimum(x, upper_trunc)           # 5) censura superior (1 línea)
    return x, comp


def ecdf_xy(a):
    a = np.sort(np.asarray(a, float))
    n = a.size
    y = np.arange(1, n+1, dtype=float) / n
    return a, y



def plot_hist_with_cdf(x, comp, bins=50, mode="stacked", show_points=False):
    colors = ["C0", "C1", "C2"]
    labels = ["Log1", "Log2", "Log3"]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    edges = np.histogram_bin_edges(x, bins=bins)
    widths = np.diff(edges)
    B = len(widths)
    lefts = edges[:-1]

    # Densidades por componente (suman a la total)
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

    # Contorno de densidad total (longitudes correctas)
    ax1.step(edges, np.r_[total, total[-1]], where="post",
             lw=1.2, ls="--", color="k", alpha=0.6, label="Total density")

    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Density")
    ax1.grid(True, alpha=0.3, zorder=0)

    # === ECDF mejorada ===
    ax2 = ax1.twinx()
    xs_all = np.sort(x)
    ys_all = np.arange(1, x.size + 1) / x.size

    # 1) Base blanca gruesa (halo)
    ax2.step(xs_all, ys_all, where="post",
             lw=3.0, color="1.0", alpha=1.0, zorder=1)
    # 2) Línea principal gris oscuro, discontinua
    ax2.step(xs_all, ys_all, where="post",
             lw=1.3, ls=(0, (4, 2)), color="0.15", alpha=1.0, zorder=2,
             label="ECDF (total)")

    # Puntos por componente (opcionales y submuestreados)
    if show_points:
        order = np.argsort(x)
        comp_sorted = comp[order]
        idx = np.linspace(0, x.size - 1, min(400, x.size), dtype=int)
        ax2.scatter(xs_all[idx], ys_all[idx], s=14,
                    c=np.take(colors, comp_sorted[idx]), alpha=0.9,
                    edgecolors="white", linewidths=0.4, zorder=3)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Cumulative Probability")
    ax2.tick_params(axis="y", colors="0.15")

    # Leyenda combinada
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + [h2[0]], l1 + [l2[0]], loc="best")

    plt.tight_layout()
    plt.show()


# Ejemplo de uso:
x, comp = sample_times(2000, seed=123)
plot_hist_with_cdf(x, comp, bins=50, mode="stacked", show_points=False)
