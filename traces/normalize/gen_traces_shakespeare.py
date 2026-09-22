#!/usr/bin/env python3
"""
Genera traces sys_metrics de shakespeare para varios numeros de clientes,
partiendo del trace de referencia c=20.

Idea: la CDF de la carga computacional depende solo de cuantas veces
aparece cada cliente. Si repartimos las apariciones en proporcion al peso
que cada cliente tiene en el trace de referencia, la CDF sale igual para
cualquier c (no depende del azar).

Uso:  python gen_traces_shakespeare.py
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
REFERENCIA = AQUI / "sys" / "sys_metrics_shakespeare_fedavg_c_20_e_1.csv"
SALIDA = AQUI / "sys_gen"

CLIENTES = [2, 3, 4, 5, 8, 10]
SEED = 42

COLS = ["client_id", "round", "hierarchy", "num_samples", "set",
        "bytes_read", "bytes_written", "local_computations"]


def repartir(pesos, rounds, c, rng):
    """Devuelve una lista de rounds; cada round es una lista de c clientes."""
    n_total = rounds * c

    # 1) Cuantas veces aparece cada cliente (metodo del mayor resto).
    exacto = pesos / pesos.sum() * n_total
    copias = np.floor(exacto).astype(int)
    faltan = n_total - copias.sum()
    if faltan > 0:
        extra = (exacto - copias).sort_values(ascending=False).index[:faltan]
        copias.loc[extra] += 1

    assert copias.max() <= rounds, "un cliente necesita mas apariciones que rounds"

    # 2) En cada round se eligen los c clientes con mas apariciones pendientes
    #    (empates al azar). Asi nunca se repite un cliente dentro de un round.
    pendientes = copias.to_dict()
    seleccion = []
    for _ in range(rounds):
        orden = sorted(pendientes, key=lambda cid: (-pendientes[cid], rng.random()))
        elegidos = orden[:c]
        rng.shuffle(elegidos)
        for cid in elegidos:
            pendientes[cid] -= 1
        seleccion.append(elegidos)

    return seleccion


def main():
    SALIDA.mkdir(exist_ok=True)

    ref = pd.read_csv(REFERENCIA, header=None, names=COLS)
    rounds = ref["round"].nunique()

    # Datos fijos de cada cliente (num_samples, flops, bytes son siempre iguales).
    pool = ref.drop_duplicates("client_id").set_index("client_id")
    # Peso de cada cliente en la CDF = veces que aparece en la referencia.
    pesos = ref["client_id"].value_counts()

    print(f"Referencia: {len(pool)} clientes, {rounds} rounds\n")

    for c in CLIENTES:
        rng = random.Random(SEED)
        seleccion = repartir(pesos, rounds, c, rng)

        filas = []
        for numero_round, elegidos in enumerate(seleccion, start=1):
            for cid in elegidos:
                cliente = pool.loc[cid]
                filas.append([
                    cid,
                    numero_round,
                    "",
                    cliente["num_samples"],
                    "train",
                    cliente["bytes_read"],
                    cliente["bytes_written"],
                    cliente["local_computations"],
                ])

        salida = SALIDA / f"sys_metrics_shakespeare_fedavg_c_{c}_e_1.csv"
        pd.DataFrame(filas, columns=COLS).to_csv(salida, header=False, index=False)
        print(f"c={c:<3} {len(filas):>4} filas  ->  {salida.name}")

    print(f"\nListo. Revisa las CDFs antes de copiarlos a {AQUI / 'sys'}")


if __name__ == "__main__":
    main()