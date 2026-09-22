#!/usr/bin/env python3
"""
Regenera traces sys_metrics de femnist fedavg desde el pool completo de
clientes.

El problema: el trace original de c=50 usa solo 50 de los 184 clientes del
pool (cada uno en las 1000 rondas), mientras que los de c=3..30 usan los 184.
Por eso su CDF de carga se despega 4.5 puntos porcentuales de las demas.

La solucion es la misma que en gen_traces_shakespeare.py: repartir las
apariciones en vez de sortearlas. La unica diferencia es el objetivo: aqui
todos los clientes pesan igual (uniforme sobre los 184), porque es a eso a
lo que convergen los traces buenos.

Uso:  python gen_traces_femnist.py
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
# Cualquier trace con el pool completo de 184 clientes sirve de referencia.
REFERENCIA = AQUI / "sys" / "sys_metrics_femnist_fedavg_c_30_e_1.csv"
SALIDA = AQUI / "sys_gen"

CLIENTES = [50]
SEED = 42

COLS = ["client_id", "round", "hierarchy", "num_samples", "set",
        "bytes_read", "bytes_written", "local_computations"]


def repartir(clientes, rounds, c, rng):
    """Devuelve una lista de rounds; cada round es una lista de c clientes.

    Todos los clientes reciben el mismo numero de apariciones (metodo del
    mayor resto para el sobrante).
    """
    n_total = rounds * c

    # 1) Cuantas veces aparece cada cliente.
    copias = {cid: n_total // len(clientes) for cid in clientes}
    sobran = n_total - sum(copias.values())
    for cid in rng.sample(sorted(clientes), sobran):
        copias[cid] += 1

    assert max(copias.values()) <= rounds, "un cliente necesita mas apariciones que rounds"

    # 2) En cada round se eligen los c clientes con mas apariciones pendientes
    #    (empates al azar). Asi nunca se repite un cliente dentro de un round.
    seleccion = []
    for _ in range(rounds):
        orden = sorted(copias, key=lambda cid: (-copias[cid], rng.random()))
        elegidos = orden[:c]
        rng.shuffle(elegidos)
        for cid in elegidos:
            copias[cid] -= 1
        seleccion.append(elegidos)

    return seleccion


def main():
    SALIDA.mkdir(exist_ok=True)

    ref = pd.read_csv(REFERENCIA, header=None, names=COLS, dtype={0: str})
    rounds = ref["round"].nunique()

    # Datos fijos de cada cliente (num_samples, flops y bytes no cambian).
    pool = ref.drop_duplicates("client_id").set_index("client_id")
    datos = pool[["num_samples", "bytes_read", "bytes_written",
                  "local_computations"]].to_dict("index")

    print(f"Referencia: {REFERENCIA.name}")
    print(f"Pool: {len(datos)} clientes, {rounds} rounds\n")

    for c in CLIENTES:
        rng = random.Random(SEED)
        seleccion = repartir(list(datos), rounds, c, rng)

        filas = []
        for numero_round, elegidos in enumerate(seleccion, start=1):
            for cid in elegidos:
                d = datos[cid]
                filas.append([
                    cid,
                    numero_round,
                    "",
                    d["num_samples"],
                    "train",
                    d["bytes_read"],
                    d["bytes_written"],
                    d["local_computations"],
                ])

        salida = SALIDA / f"sys_metrics_femnist_fedavg_c_{c}_e_1.csv"
        pd.DataFrame(filas, columns=COLS).to_csv(salida, header=False, index=False)
        print(f"c={c:<3} {len(filas):>6} filas, {len(set(x[0] for x in filas))} clientes"
              f"  ->  {salida.name}")

    print(f"\nListo. Revisa las CDFs antes de copiarlos a {AQUI / 'sys'}")


if __name__ == "__main__":
    main()