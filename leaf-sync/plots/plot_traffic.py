import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_bases_1 = [
    "sys_metrics_minibatch_c_20_mb_1",
    "sys_metrics_minibatch_c_20_mb_0.9",
    "sys_metrics_minibatch_c_20_mb_0.8",
    "sys_metrics_minibatch_c_20_mb_0.6",
    "sys_metrics_minibatch_c_20_mb_0.5",
    "sys_metrics_minibatch_c_20_mb_0.4",
    "sys_metrics_minibatch_c_20_mb_0.2",
]

def load_data(sys_metrics_file):
    if not os.path.isfile(sys_metrics_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {sys_metrics_file}")
    df = pd.read_csv(sys_metrics_file)
    df.columns = [
        "client_id", "round_number", "idk", "samples", "set",
        "bytes_read", "bytes_written", "FLOPs"
    ]
    df.index.name = "index"
    return df

def convert_FLOPs_to_time(df):
    capacity_fps = 1e9  # 1 GFLOPS
    df = df.copy()
    df["time"] = (df["FLOPs"] / capacity_fps) * 1e3  # ms
    return df

def ensure_dir(path="figures"):
    os.makedirs(path, exist_ok=True)
    return path

def label_from_base(name):
    # extrae el valor después de "mb_" (e.g., "0.8" de "..._mb_0.8")
    m = re.search(r"mb_([0-9.]+)$", name)
    if m:
        return f"mb={m.group(1)}", float(m.group(1))
    # fallback
    return name, np.nan

def main():
    path_folder = "../results/sys/"
    ext_file = ".csv"

    # cargar y combinar
    frames = []
    hue_vals = []
    for base in file_bases_1:
        fpath = os.path.join(path_folder, base + ext_file)
        try:
            df = load_data(fpath)
            df = convert_FLOPs_to_time(df)
            label, mb_val = label_from_base(base)
            df = df[["time"]].dropna().copy()
            df["config"] = label
            df["mb_val"] = mb_val
            frames.append(df)
            hue_vals.append((label, mb_val))
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
        except Exception as e:
            print(f"[WARN] Falha ao processar {fpath}: {e}")

    if not frames:
        print("[ERRO] Nenhum arquivo carregado.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # orden de la leyenda por mb ascendente (1.0, 0.9, 0.8, ...)
    order = sorted({t for t in hue_vals}, key=lambda z: (np.isnan(z[1]), z[1]))
    hue_order = [lbl for (lbl, _) in order]

    ensure_dir("figures")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))

    # Una sola llamada con hue para todas las curvas
    sns.ecdfplot(data=all_df, x="time", hue="config", hue_order=hue_order, lw=1.8)

    plt.xlabel("Client Computation Time (ms)")
    plt.ylabel("ECDF")
    plt.title("Client Computation Time — ECDF (minibatch c=20)")
    plt.ylim(0, 1)

    # Opcional: recortar extremos si hay colas largas (descomenta si quieres)
    # lo, hi = np.percentile(all_df["time"], [0.0, 99.5])
    # plt.xlim(lo, hi)

    plt.legend(title="Minibatch fraction", loc="lower right")
    plt.tight_layout()
    plt.savefig("figures/ecdf_time_minibatch_c20.png", dpi=150)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()
