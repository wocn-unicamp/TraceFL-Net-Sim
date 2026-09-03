#!/usr/bin/env python3
"""
Junta los traces de net/ que solo se diferencian por la seed.

Para cada tipo de simulacion crea un unico CSV en net_join/ donde primero
van todas las filas del round 1 (de todas las seeds), luego las del round 2,
y asi sucesivamente.

Uso:  python join_traces.py
"""

import re
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
NET = AQUI / "net"
NET_JOIN = AQUI / "net_join"


def main():
    NET_JOIN.mkdir(exist_ok=True)

    # 1) Agrupar los archivos. La clave es el nombre sin la parte "_seed_XXXX",
    #    asi que todos los que solo cambian de seed caen en el mismo grupo.
    grupos = {}
    for archivo in sorted(NET.glob("*.csv")):
        clave = re.sub(r"_seed_\d+", "", archivo.name)
        grupos.setdefault(clave, []).append(archivo)

    # 2) Juntar cada grupo y ordenar por round.
    for clave, archivos in sorted(grupos.items()):
        # dtype=str -> los numeros se copian tal cual estan en el archivo
        partes = [pd.read_csv(a, dtype=str) for a in archivos]
        df = pd.concat(partes, ignore_index=True)

        # sort estable: agrupa por round y dentro de cada round mantiene
        # el orden original (seed por seed, fila por fila)
        df["_round"] = df["round_number"].astype(int)
        df = df.sort_values("_round", kind="stable").drop(columns="_round")

        df.to_csv(NET_JOIN / clave, index=False)
        print(f"{clave}  <-  {len(archivos)} seeds, {len(df)} filas")

    print(f"\nListo: {len(grupos)} archivos escritos en {NET_JOIN}")


if __name__ == "__main__":
    main()