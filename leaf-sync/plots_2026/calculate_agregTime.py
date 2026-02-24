import os
import glob
import math
import pandas as pd


PATTERN = "sys_metrics_fedavg_c_64_e_[1-5].csv"
Z_95 = 1.96          # Aproximação normal para IC de 95%
OUTLIER_MAX = 10.0   # Remove gaps maiores que 30s


def load_csv(path: str) -> pd.DataFrame:
    # Lê o CSV e deixa o pandas inferir o separador (tab, vírgula, etc.)
    return pd.read_csv(path, sep=None, engine="python")


def compute_gaps_df(df: pd.DataFrame) -> pd.DataFrame:
    # Agrupa por round e pega:
    # - round_end  = max(sim_time_end)   (fim do round r)
    # - round_start= min(sim_time_start) (início do round r)
    rounds = (
        df.groupby("round_number")
          .agg(round_end=("sim_time_end", "max"),
               round_start=("sim_time_start", "min"))
          .sort_index()
    )

    # Próximo round (pela ordem do índice após sort_index)
    next_round = rounds.index.to_series().shift(-1)

    # gap(r) = start_{r+1} - end_{r}
    gap = rounds["round_start"].shift(-1) - rounds["round_end"]

    gdf = pd.DataFrame({
        "round": rounds.index,
        "next_round": next_round,
        "gap": gap
    }).dropna()

    return gdf


def filter_outliers(gdf: pd.DataFrame, max_gap: float = OUTLIER_MAX) -> pd.DataFrame:
    # Remove outliers: gaps maiores que max_gap
    return gdf[gdf["gap"] <= max_gap].copy()


def mean_std_ci95(values: pd.Series) -> dict:
    # Calcula média, desvio padrão e IC 95% para a média (mean ± 1.96 * SE)
    n = len(values)
    mean = float(values.mean()) if n > 0 else float("nan")
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_lo": mean - Z_95 * se,
        "ci95_hi": mean + Z_95 * se,
    }


def summarize_file(file_path: str) -> dict:
    # Resume estatísticas de um arquivo (após remover outliers)
    df = load_csv(file_path)
    gdf = filter_outliers(compute_gaps_df(df), OUTLIER_MAX)

    stats = mean_std_ci95(gdf["gap"])

    # Min/Max depois do filtro
    min_row = gdf.loc[gdf["gap"].idxmin()]
    max_row = gdf.loc[gdf["gap"].idxmax()]

    return {
        "file": os.path.basename(file_path),
        "n_gaps": stats["n"],
        "mean_gap": stats["mean"],
        "std_gap": stats["std"],
        "ci95_low": stats["ci95_lo"],
        "ci95_high": stats["ci95_hi"],
        "min_gap": float(min_row["gap"]),
        "min_between": f"{int(min_row['round'])}->{int(min_row['next_round'])}",
        "max_gap": float(max_row["gap"]),
        "max_between": f"{int(max_row['round'])}->{int(max_row['next_round'])}",
    }


def print_gaps_for_file(file_path: str) -> None:
    # Imprime a lista de gaps (já filtrados) para um arquivo
    df = load_csv(file_path)
    gdf = filter_outliers(compute_gaps_df(df), OUTLIER_MAX)

    print("\n=== Gaps (<= {:.1f}s) for: {} ===".format(OUTLIER_MAX, os.path.basename(file_path)))
    for _, row in gdf.iterrows():
        r = int(row["round"])
        nr = int(row["next_round"])
        gap = float(row["gap"])
        print(f"{r}->{nr}: {gap:.6f}")


def summarize_folder(input_dir: str) -> pd.DataFrame:
    # Processa todos os arquivos que casam com o PATTERN e retorna um DataFrame de resumo
    paths = sorted(glob.glob(os.path.join(input_dir, PATTERN)))
    rows = [summarize_file(p) for p in paths]
    return pd.DataFrame(rows).sort_values("file")


if __name__ == "__main__":
    INPUT_DIR = "/home/oscar/workspace/TraceFL-Net-Sim/leaf-sync/results/sys/fine_paralelo"
    paths = sorted(glob.glob(os.path.join(INPUT_DIR, PATTERN)))

    # 1) Imprime a lista de gaps filtrados para cada arquivo
    for p in paths:
        print_gaps_for_file(p)

    # 2) Imprime o resumo por arquivo (após remover outliers)
    summary = summarize_folder(INPUT_DIR)
    print("\n\n=== Summary per file (outliers removed: gap > {:.1f}s) ===".format(OUTLIER_MAX))
    print(summary.to_string(index=False))

    # 3) Estatísticas globais (todos os gaps filtrados concatenados)
    all_gdfs = []
    for p in paths:
        gdf = filter_outliers(compute_gaps_df(load_csv(p)), OUTLIER_MAX)
        gdf["file"] = os.path.basename(p)
        all_gdfs.append(gdf)

    all_gaps_df = pd.concat(all_gdfs, ignore_index=True)
    overall = mean_std_ci95(all_gaps_df["gap"])

    global_min = all_gaps_df.loc[all_gaps_df["gap"].idxmin()]
    global_max = all_gaps_df.loc[all_gaps_df["gap"].idxmax()]

    print("\n=== Overall (all values, outliers removed) ===")
    print("Overall mean gap:", overall["mean"])
    print("Overall std  gap:", overall["std"])
    print("Overall 95% CI  :", "[{:.6f}, {:.6f}]".format(overall["ci95_lo"], overall["ci95_hi"]))

    print("\nGlobal MIN gap:",
          float(global_min["gap"]),
          "| file:", global_min["file"],
          "| between rounds:", f"{int(global_min['round'])}->{int(global_min['next_round'])}")

    print("Global MAX gap:",
          float(global_max["gap"]),
          "| file:", global_max["file"],
          "| between rounds:", f"{int(global_max['round'])}->{int(global_max['next_round'])}")
