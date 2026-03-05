import os
import pandas as pd
import numpy as np

# =========================
# Config
# =========================
SIM_TYPE = "serial_lowcap" # serial | paralelo | serial_lowcap
IN_FOLDER = f"../results/sys/fine_{SIM_TYPE}/"
OUT_FOLDER = f"../results/time/fine_{SIM_TYPE}/"
os.makedirs(OUT_FOLDER, exist_ok=True)

C = 64
EPOCHS = range(1, 6)

# Headers
WRITE_HEADER_MAIN = True   # para: IN_FOLDER *_time.csv (augmentado) y OUT_FOLDER *_hom.csv / *_het.csv
WRITE_HEADER_TIME = False  # para: OUT_FOLDER *_hom_time.csv / *_het_time.csv

# Capacidades (GFLOPs/s)
BASE_GFLOPS = 114.0
HOM_CAP_GFLOPS = 100.0
HET_SPLIT = [(0.50, 100.0), (0.30, 64.0), (0.20, 200.0)]  # 50/30/20

COLS_FULL = ["client", "round_number", "computingTime", "computingDemand", "computingCap", "computingtime"]
COLS_TIME = ["client", "round", "computingtime", "time"]


# =========================
# Helpers
# =========================
def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str, header: bool) -> None:
    df.to_csv(path, index=False, header=header)


def fixed_hetero_cap_map(n_clients: int, split: list[tuple[float, float]]) -> dict[int, float]:
    counts = [int(np.floor(p * n_clients)) for (p, _) in split]
    counts[-1] = n_clients - sum(counts[:-1])  # ajusta para sumar exacto

    caps = []
    for cnt, (_, gflops) in zip(counts, split):
        caps.extend([float(gflops)] * cnt)

    return {i + 1: caps[i] for i in range(n_clients)}  # client 1..C


def add_client_seq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["round_number"] = df["round_number"].astype(int)
    df["client"] = df.groupby("round_number").cumcount() + 1
    return df


def add_demand_flops(df: pd.DataFrame, base_gflops: float) -> pd.DataFrame:
    df = df.copy()
    df["computingTime"] = df["computingTime"].astype(float)
    df["computingDemand"] = df["computingTime"] * (base_gflops * 1e9)  # FLOPs
    return df


def build_homogeneous(df: pd.DataFrame, cap_gflops: float) -> pd.DataFrame:
    out = df.copy()
    out["computingCap"] = float(cap_gflops)  # GFLOPs/s
    out["computingtime"] = out["computingDemand"] / (out["computingCap"] * 1e9)  # s
    return out[COLS_FULL].copy()


def build_heterogeneous(df: pd.DataFrame, cap_map: dict[int, float]) -> pd.DataFrame:
    out = df.copy()
    out["computingCap"] = out["client"].map(cap_map).astype(float)  # GFLOPs/s
    out["computingtime"] = out["computingDemand"] / (out["computingCap"] * 1e9)  # s
    return out[COLS_FULL].copy()


def add_time_column(df: pd.DataFrame) -> pd.DataFrame:
    # time(round 1) = computingtime
    # time(round r) = computingtime + max(time round r-1)
    out = df.copy()
    durations = out.groupby("round_number")["computingtime"].max().sort_index()
    offsets = durations.cumsum().shift(1, fill_value=0.0)
    out["time"] = out["computingtime"] + out["round_number"].map(offsets)
    return out


def build_time_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["client", "round_number", "computingtime", "time"]].copy()
    out = out.rename(columns={"round_number": "round"})
    return out[COLS_TIME]


def augment_original_with_caps_demand_times(
    df_raw: pd.DataFrame,
    base_gflops: float,
    hom_cap_gflops: float,
    cap_map_seq: dict[int, float],
) -> pd.DataFrame:
    # agrega al final: computingDemand, computingCap_hom, computingCap_het, computingTime_hom, computingTime_het
    out = df_raw.copy()
    out["round_number"] = out["round_number"].astype(int)
    out["computingTime"] = out["computingTime"].astype(float)

    client_seq = out.groupby("round_number").cumcount() + 1  # para asignar cap_het igual que archivos individuales

    out["computingDemand"] = out["computingTime"] * (base_gflops * 1e9)  # FLOPs
    out["computingCap_hom"] = float(hom_cap_gflops)  # GFLOPs/s
    out["computingCap_het"] = client_seq.map(cap_map_seq).astype(float)  # GFLOPs/s

    out["computingTime_hom"] = out["computingTime"] * (base_gflops / hom_cap_gflops)
    out["computingTime_het"] = out["computingTime"] * (base_gflops / out["computingCap_het"])

    base_cols = list(df_raw.columns)
    extra_cols = ["computingDemand", "computingCap_hom", "computingCap_het", "computingTime_hom", "computingTime_het"]
    return out[base_cols + extra_cols]


# =========================
# Main
# =========================
def main():
    cap_map_seq = fixed_hetero_cap_map(C, HET_SPLIT)

    for epoch in EPOCHS:
        in_path = os.path.join(IN_FOLDER, f"sys_metrics_fedavg_c_{C}_e_{epoch}.csv")
        df_raw = load_df(in_path)

        # (1) IN_FOLDER: original + demand + caps + computingTime_(hom/het)
        out_infolder_time = os.path.join(IN_FOLDER, f"sys_metrics_fedavg_c_{C}_e_{epoch}_time.csv")
        df_aug = augment_original_with_caps_demand_times(df_raw, BASE_GFLOPS, HOM_CAP_GFLOPS, cap_map_seq)
        save_csv(df_aug, out_infolder_time, WRITE_HEADER_MAIN)
        print(f"OK epoch={epoch} -> {out_infolder_time}")

        # (2) OUT_FOLDER: hom/het (full)
        df = add_client_seq(df_raw)
        df = add_demand_flops(df, BASE_GFLOPS)

        df_hom = build_homogeneous(df, HOM_CAP_GFLOPS)
        df_het = build_heterogeneous(df, cap_map_seq)

        out_hom = os.path.join(OUT_FOLDER, f"sys_metrics_fedavg_c_{C}_e_{epoch}_hom.csv")
        out_het = os.path.join(OUT_FOLDER, f"sys_metrics_fedavg_c_{C}_e_{epoch}_het.csv")
        save_csv(df_hom, out_hom, WRITE_HEADER_MAIN)
        save_csv(df_het, out_het, WRITE_HEADER_MAIN)
        print(f"OK epoch={epoch} -> {out_hom}")
        print(f"OK epoch={epoch} -> {out_het}")

        # (3) OUT_FOLDER: *_time.csv (4 columnas) SIN HEADER
        out_hom_time = os.path.join(OUT_FOLDER, f"sys_metrics_fedavg_c_{C}_e_{epoch}_hom_time.csv")
        out_het_time = os.path.join(OUT_FOLDER, f"sys_metrics_fedavg_c_{C}_e_{epoch}_het_time.csv")

        hom_time = build_time_csv(add_time_column(df_hom))
        het_time = build_time_csv(add_time_column(df_het))

        save_csv(hom_time, out_hom_time, WRITE_HEADER_TIME)
        save_csv(het_time, out_het_time, WRITE_HEADER_TIME)
        print(f"OK epoch={epoch} -> {out_hom_time}")
        print(f"OK epoch={epoch} -> {out_het_time}")


if __name__ == "__main__":
    main()