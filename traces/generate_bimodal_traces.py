#!/usr/bin/env python3
"""
Genera traces con capacidad de procesamiento heterogenea (bimodal).

Modelo:
    C_i ~ N(0.5, 0.12^2)  para el 50% de los clientes  (hardware lento)
    C_i ~ N(1.5, 0.12^2)  para el otro 50%             (hardware rapido)
    truncadas a [0.20, 1.80] GFLOP/s por muestreo con rechazo

    T = local_computations / (C_i * 1e9)

Con AMDAHL = True esas capacidades se interpretan como la capacidad de UN
nucleo y se multiplican por el speedup de la ley de Amdahl, que depende del
dataset:

    speedup = 1 / ((1 - p) + p / cores)
    C_efectiva = C_base * speedup

Con AMDAHL = False el speedup es 1 y el resultado es el de siempre.

La capacidad se asigna UNA VEZ por cliente y se reutiliza en todas las rondas.
Los traces originales no se modifican.

Uso:  python generate_bimodal_traces.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13
})


# ============================ CONFIGURACION ============================

AMDAHL = True          # <-- la flag: True aplica la ley de Amdahl

CORES = 4               # nucleos por dispositivo
P_FEMNIST = 0.95        # fraccion paralelizable de femnist
P_SHAKESPEARE = 0.4     # fraccion paralelizable de shakespeare

MODE1_MEAN = 0.5        # GFLOP/s, hardware lento
MODE2_MEAN = 1.5        # GFLOP/s, hardware rapido
MODE1_STD = 0.12
MODE2_STD = 0.12
MIN_CAPACITY = 0.20     # limites de la capacidad base
MAX_CAPACITY = 1.80

SEED = 42

# dataset: un mapa por dataset, el mismo cliente tiene la misma capacidad en
#          todos los traces. Es lo que mantiene comparables las CDFs.
# file:    un mapa por archivo.
SCOPE = "dataset"

AQUI = Path(__file__).resolve().parent
ENTRADA = AQUI / "sys"
PATRON = "sys_metrics_*.csv"
# Directorios distintos para no pisar una version con la otra.
SALIDA = AQUI / ("sys_bimodal_amdahl" if AMDAHL else "sys_bimodal")

# =======================================================================


COLS = ["client_id", "round", "hierarchy", "num_samples", "set",
        "bytes_read", "bytes_written", "local_computations"]


def dataset_de(nombre):
    """Deduce el dataset a partir del nombre del archivo."""
    n = nombre.lower()
    if "femnist" in n:
        return "femnist"
    if "shakespeare" in n:
        return "shakespeare"
    return "otro"


def amdahl_speedup(cores, p):
    """Cuanto acelera repartir el trabajo entre varios nucleos.

    Con p=0.95 y 4 nucleos el speedup es 3.478; con p=0.4 y 4 nucleos, 1.429.
    """
    return 1.0 / ((1.0 - p) + p / cores)


def muestrear_capacidades(media, std, n, rng):
    """Muestreo con rechazo: descarta lo que cae fuera de los limites.

    No se usa np.clip porque acumularia masa exactamente en el limite. Con los
    valores por defecto se rechaza el 0.62% de las muestras.
    """
    valores = np.empty(0)
    while len(valores) < n:
        candidatos = rng.normal(media, std, size=max(2 * (n - len(valores)), 32))
        candidatos = candidatos[(candidatos >= MIN_CAPACITY) &
                                (candidatos <= MAX_CAPACITY)]
        valores = np.concatenate([valores, candidatos])
    return valores[:n]


def asignar_capacidades(clientes, speedup, rng):
    """Reparte los clientes 50/50 entre los dos modos y les da una capacidad.

    El reparto es determinista (barajar y cortar por la mitad), no Bernoulli
    independiente, para que con pocos clientes no salga un 80/20 por azar.
    Con numero impar, el cliente extra va al modo 1.

    La capacidad base se muestrea dentro de [MIN, MAX] y despues se multiplica
    por el speedup. Los limites se aplican a la base, que es donde tienen
    sentido fisico; escalar despues no deforma la distribucion.
    """
    orden = np.array(sorted(clientes), dtype=object)   # orden canonico
    rng.shuffle(orden)

    n1 = len(orden) - len(orden) // 2
    modo1, modo2 = orden[:n1], orden[n1:]

    cap1 = muestrear_capacidades(MODE1_MEAN, MODE1_STD, len(modo1), rng)
    cap2 = muestrear_capacidades(MODE2_MEAN, MODE2_STD, len(modo2), rng)
    base = np.concatenate([cap1, cap2])

    return pd.DataFrame({
        "client_id": np.concatenate([modo1, modo2]),
        "mode": [1] * len(modo1) + [2] * len(modo2),
        "capacity_base_gflops": base,
        "speedup": speedup,
        "capacity_gflops": base * speedup,
    }).sort_values("client_id").reset_index(drop=True)


def graficar(capacidades, speedup, salida, titulo):
    """Histograma de capacidades efectivas POR CLIENTE (no por fila)."""
    minimo = MIN_CAPACITY * speedup
    maximo = MAX_CAPACITY * speedup
    bins = np.linspace(minimo, maximo, 60)
    ancho = bins[1] - bins[0]
    x = np.linspace(minimo, maximo, 500)

    plt.figure(figsize=(7, 4.5))
    for modo, media, std, color in [
            (1, MODE1_MEAN * speedup, MODE1_STD * speedup, "tab:blue"),
            (2, MODE2_MEAN * speedup, MODE2_STD * speedup, "tab:orange")]:
        datos = capacidades.loc[capacidades["mode"] == modo, "capacity_gflops"]
        plt.hist(datos, bins=bins, color=color, alpha=0.65,
                 label=f"Mode {modo} (n={len(datos)}, mean={datos.mean():.3f})")
        # gaussiana teorica escalada al numero de clientes de ese modo
        pdf = np.exp(-0.5 * ((x - media) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        plt.plot(x, pdf * len(datos) * ancho, color=color, linewidth=1.2)
        plt.axvline(media, color=color, linestyle=":", linewidth=1)

    plt.axvline(minimo, color="gray", linestyle="--", linewidth=1)
    plt.axvline(maximo, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Processing capacity (GFLOP/s)")
    plt.ylabel("Number of clients")
    plt.xlim(minimo - 0.05 * speedup, maximo + 0.05 * speedup)
    plt.title(titulo)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(salida, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    graficas = SALIDA / "plots"
    SALIDA.mkdir(parents=True, exist_ok=True)
    graficas.mkdir(exist_ok=True)

    archivos = sorted(ENTRADA.glob(PATRON))
    if not archivos:
        raise SystemExit(f"No hay archivos {PATRON} en {ENTRADA}")

    print(f"Amdahl: {'SI' if AMDAHL else 'NO'}"
          + (f" (cores={CORES})" if AMDAHL else ""))

    # 1) Agrupar los archivos que comparten mapa de capacidades.
    grupos = {}
    for archivo in archivos:
        clave = dataset_de(archivo.name) if SCOPE == "dataset" else archivo.stem
        grupos.setdefault(clave, []).append(archivo)

    rng = np.random.default_rng(SEED)

    for clave, del_grupo in sorted(grupos.items()):
        # 2) Union de los client_id de todos los traces del grupo.
        clientes = set()
        for archivo in del_grupo:
            clientes |= set(pd.read_csv(archivo, header=None, names=COLS,
                                        usecols=["client_id"], dtype=str)["client_id"])

        # 3) Speedup de Amdahl. Depende del dataset, porque la fraccion
        #    paralelizable no es la misma en femnist que en shakespeare.
        dataset = dataset_de(del_grupo[0].name)
        if AMDAHL:
            p_dataset = P_FEMNIST if dataset == "femnist" else P_SHAKESPEARE
            speedup = amdahl_speedup(CORES, p_dataset)
        else:
            p_dataset, speedup = None, 1.0

        # 4) Un mapa cliente -> capacidad, fijo para todas las rondas y archivos.
        capacidades = asignar_capacidades(clientes, speedup, rng)
        capacidades.to_csv(SALIDA / f"capacities_{clave}.csv", index=False,
                           float_format="%.6f")

        titulo = f"{len(capacidades)} clients"
        if AMDAHL:
            titulo += f"  |  Amdahl p={p_dataset}, {CORES} cores, speedup={speedup:.3f}"
        graficar(capacidades, speedup, graficas / f"capacity_{clave}.png", titulo)

        mapa = capacidades.set_index("client_id")["capacity_gflops"]
        modos = capacidades.set_index("client_id")["mode"]
        m1 = capacidades[capacidades["mode"] == 1]["capacity_gflops"]
        m2 = capacidades[capacidades["mode"] == 2]["capacity_gflops"]

        print(f"\n=== {clave}: {len(del_grupo)} traces, {len(capacidades)} clientes unicos")
        if AMDAHL:
            print(f"    Amdahl: p={p_dataset}, speedup={speedup:.4f}, "
                  f"modos en {MODE1_MEAN * speedup:.3f} y {MODE2_MEAN * speedup:.3f} GFLOP/s")
        print(f"    modo 1: n={len(m1):<4} media={m1.mean():.4f}  std={m1.std(ddof=0):.4f}")
        print(f"    modo 2: n={len(m2):<4} media={m2.mean():.4f}  std={m2.std(ddof=0):.4f}")
        print(f"    capacidad en [{mapa.min():.4f}, {mapa.max():.4f}] GFLOP/s")

        # 5) Reescribir cada trace anadiendo capacity_gflops y time.
        for archivo in del_grupo:
            df = pd.read_csv(archivo, header=None, names=COLS, dtype=str)
            capacidad = df["client_id"].map(mapa)
            assert capacidad.notna().all(), f"cliente sin capacidad en {archivo.name}"

            flops = df["local_computations"].astype(float)
            tiempo = flops / (capacidad * 1e9)

            df["capacity_gflops"] = capacidad.map("{:.6f}".format)
            df["time"] = tiempo.map("{:.9f}".format)
            df.to_csv(SALIDA / archivo.name, header=False, index=False)

            n_total = df["client_id"].nunique()
            n_modo1 = modos.loc[df["client_id"].unique()].eq(1).sum()
            print(f"    {archivo.name:<50} {n_total:>4} clientes "
                  f"({n_modo1} modo1 / {n_total - n_modo1} modo2)  "
                  f"T de {tiempo.min():.2f} a {tiempo.max():.2f} s")

    print(f"\nTraces en {SALIDA}, graficas en {graficas}")
    print(f"Originales intactos en {ENTRADA}")


if __name__ == "__main__":
    main()
