#!/usr/bin/env python3
"""
Genera traces con capacidad de procesamiento heterogenea (bimodal).

Modelo:
    C_i ~ N(0.5, 0.12^2)  para el 50% de los clientes  (hardware lento)
    C_i ~ N(1.5, 0.12^2)  para el otro 50%             (hardware rapido)
    truncadas a [0.20, 1.80] GFLOP/s por muestreo con rechazo

    T = local_computations / (C_i * 1e9)

La capacidad se asigna UNA VEZ por cliente y se reutiliza en todas las rondas.
Los traces originales no se modifican.

Uso:
    python generate_bimodal_traces.py
    python generate_bimodal_traces.py --seed 7 --scope file
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLS = ["client_id", "round", "hierarchy", "num_samples", "set",
        "bytes_read", "bytes_written", "local_computations"]


# ---------------------------------------------------------------- utilidades

def dataset_de(nombre):
    """Deduce el dataset a partir del nombre del archivo."""
    n = nombre.lower()
    if "femnist" in n:
        return "femnist"
    if "shakespeare" in n:
        return "shakespeare"
    return "otro"


def muestrear_capacidades(media, std, minimo, maximo, n, rng):
    """Muestreo con rechazo: descarta lo que cae fuera de [minimo, maximo].

    No se usa np.clip porque acumularia masa exactamente en los limites.
    Con los valores por defecto se rechaza el 0.62% de las muestras.
    """
    valores = np.empty(0)
    while len(valores) < n:
        candidatos = rng.normal(media, std, size=max(2 * (n - len(valores)), 32))
        candidatos = candidatos[(candidatos >= minimo) & (candidatos <= maximo)]
        valores = np.concatenate([valores, candidatos])
    return valores[:n]


def asignar_capacidades(clientes, cfg, rng):
    """Reparte los clientes 50/50 entre los dos modos y les da una capacidad.

    El reparto es determinista (barajar y cortar por la mitad), no Bernoulli
    independiente, para que con pocos clientes no salga un 80/20 por azar.
    Con numero impar, el cliente extra va al modo 1.
    """
    orden = np.array(sorted(clientes), dtype=object)   # orden canonico
    rng.shuffle(orden)

    n1 = len(orden) - len(orden) // 2
    modo1, modo2 = orden[:n1], orden[n1:]

    cap1 = muestrear_capacidades(cfg.mode1_mean, cfg.mode1_std,
                                 cfg.min_capacity, cfg.max_capacity, len(modo1), rng)
    cap2 = muestrear_capacidades(cfg.mode2_mean, cfg.mode2_std,
                                 cfg.min_capacity, cfg.max_capacity, len(modo2), rng)

    return pd.DataFrame({
        "client_id": np.concatenate([modo1, modo2]),
        "mode": [1] * len(modo1) + [2] * len(modo2),
        "capacity_gflops": np.concatenate([cap1, cap2]),
    }).sort_values("client_id").reset_index(drop=True)


def graficar(capacidades, cfg, salida, titulo):
    """Histograma de capacidades POR CLIENTE (no por fila del trace)."""
    bins = np.linspace(cfg.min_capacity, cfg.max_capacity, 60)
    ancho = bins[1] - bins[0]
    x = np.linspace(cfg.min_capacity, cfg.max_capacity, 500)

    plt.figure(figsize=(7, 4))
    for modo, media, std, color in [(1, cfg.mode1_mean, cfg.mode1_std, "tab:blue"),
                                    (2, cfg.mode2_mean, cfg.mode2_std, "tab:orange")]:
        datos = capacidades.loc[capacidades["mode"] == modo, "capacity_gflops"]
        plt.hist(datos, bins=bins, color=color, alpha=0.65,
                 label=f"Mode {modo} (n={len(datos)}, mean={datos.mean():.3f})")
        # gaussiana teorica escalada al numero de clientes de ese modo
        pdf = np.exp(-0.5 * ((x - media) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        plt.plot(x, pdf * len(datos) * ancho, color=color, linewidth=1.2)
        plt.axvline(media, color=color, linestyle=":", linewidth=1)

    plt.axvline(cfg.min_capacity, color="gray", linestyle="--", linewidth=1)
    plt.axvline(cfg.max_capacity, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Processing capacity (GFLOP/s)")
    plt.ylabel("Number of clients")
    plt.xlim(cfg.min_capacity - 0.05, cfg.max_capacity + 0.05)
    plt.title(titulo)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(salida, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="sys")
    p.add_argument("--output-dir", default="sys_bimodal")
    p.add_argument("--pattern", default="sys_metrics_*.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode1-mean", type=float, default=0.5)
    p.add_argument("--mode2-mean", type=float, default=1.5)
    p.add_argument("--mode1-std", type=float, default=0.12)
    p.add_argument("--mode2-std", type=float, default=0.12)
    p.add_argument("--min-capacity", type=float, default=0.20)
    p.add_argument("--max-capacity", type=float, default=1.80)
    p.add_argument("--scope", choices=["dataset", "file"], default="dataset",
                   help="dataset: un mapa por dataset, el mismo cliente tiene la "
                        "misma capacidad en todos los traces. file: un mapa por archivo.")
    cfg = p.parse_args()

    entrada = Path(cfg.input_dir)
    salida = Path(cfg.output_dir)
    graficas = salida / "plots"
    salida.mkdir(parents=True, exist_ok=True)
    graficas.mkdir(exist_ok=True)

    archivos = sorted(entrada.glob(cfg.pattern))
    if not archivos:
        raise SystemExit(f"No hay archivos {cfg.pattern} en {entrada}")

    # 1) Agrupar los archivos que comparten mapa de capacidades.
    grupos = {}
    for archivo in archivos:
        clave = dataset_de(archivo.name) if cfg.scope == "dataset" else archivo.stem
        grupos.setdefault(clave, []).append(archivo)

    rng = np.random.default_rng(cfg.seed)

    for clave, del_grupo in sorted(grupos.items()):
        # 2) Union de los client_id de todos los traces del grupo.
        clientes = set()
        for archivo in del_grupo:
            clientes |= set(pd.read_csv(archivo, header=None, names=COLS,
                                        usecols=["client_id"], dtype=str)["client_id"])

        # 3) Un mapa cliente -> capacidad, fijo para todas las rondas y archivos.
        #    (Aqui es donde entraria mas adelante el ajuste de Amdahl:
        #     capacidad_efectiva = capacidad * speedup(p_dataset, num_cores))
        capacidades = asignar_capacidades(clientes, cfg, rng)
        capacidades.to_csv(salida / f"capacities_{clave}.csv", index=False,
                           float_format="%.6f")
        graficar(capacidades, cfg, graficas / f"capacity_{clave}.png",
                 f"{clave}: {len(capacidades)} clients")

        mapa = capacidades.set_index("client_id")["capacity_gflops"]
        m1 = capacidades[capacidades["mode"] == 1]["capacity_gflops"]
        m2 = capacidades[capacidades["mode"] == 2]["capacity_gflops"]

        print(f"\n=== {clave}: {len(del_grupo)} traces, {len(capacidades)} clientes unicos")
        print(f"    modo 1: n={len(m1):<4} media={m1.mean():.4f}  std={m1.std(ddof=0):.4f}")
        print(f"    modo 2: n={len(m2):<4} media={m2.mean():.4f}  std={m2.std(ddof=0):.4f}")
        print(f"    capacidad en [{mapa.min():.4f}, {mapa.max():.4f}] GFLOP/s")

        # 4) Reescribir cada trace anadiendo capacity_gflops y time.
        for archivo in del_grupo:
            df = pd.read_csv(archivo, header=None, names=COLS, dtype=str)
            capacidad = df["client_id"].map(mapa)
            assert capacidad.notna().all(), f"cliente sin capacidad en {archivo.name}"

            flops = df["local_computations"].astype(float)
            tiempo = flops / (capacidad * 1e9)

            df["capacity_gflops"] = capacidad.map("{:.6f}".format)
            df["time"] = tiempo.map("{:.9f}".format)
            df.to_csv(salida / archivo.name, header=False, index=False)

            n_modo1 = capacidades.set_index("client_id").loc[
                df["client_id"].unique(), "mode"].eq(1).sum()
            n_total = df["client_id"].nunique()
            print(f"    {archivo.name:<50} {n_total:>4} clientes "
                  f"({n_modo1} modo1 / {n_total - n_modo1} modo2)  "
                  f"T de {tiempo.min():.2f} a {tiempo.max():.2f} s")

    print(f"\nTraces en {salida}, graficas en {graficas}")
    print(f"Originales intactos en {entrada}")


if __name__ == "__main__":
    main()