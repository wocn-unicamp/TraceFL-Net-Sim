import argparse
import fnmatch
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def find_leaf_stats_files(directory: Path, pattern: str) -> list[Path]:
    """Find files matching the pattern in the given directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Input directory '{directory}' does not exist.")

    return [
        f for f in directory.iterdir() if f.is_file() and fnmatch.fnmatch(f.name, pattern)
    ]


def generate_bimodal_flops(
    count: int,
    mean1: float,
    std1: float,
    mean2: float,
    std2: float,
    prob1: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generates FLOPS values sampled from a bimodal Gaussian distribution."""
    # Randomly assign each client to Mode 1 (with prob1) or Mode 2
    use_mode1 = rng.random(count) < prob1

    flops_mode1 = rng.normal(mean1, std1, count)
    flops_mode2 = rng.normal(mean2, std2, count)

    flops = np.where(use_mode1, flops_mode1, flops_mode2)

    # Ensure no zero or negative FLOPS values (minimum 1 MHz/1e6 FLOPS)
    return np.clip(flops, a_min=1e6, a_max=None)


def main():
    parser = argparse.ArgumentParser(description="Process federated system metrics CSVs.")

    # Directory and file pattern arguments
    parser.add_argument("--sample-dir", type=str, default="traces/sys/", help="Input directory")
    parser.add_argument("--output-dir", type=str, default="output_traces/", help="Output directory")
    parser.add_argument("--search-pattern", type=str, default="sys_metrics_*", help="File match pattern")

    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # FLOPS Configuration
    parser.add_argument(
        "--flops-mode",
        type=str,
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="FLOPS distribution mode across clients",
    )
    # Center / Baseline FLOPS
    parser.add_argument(
        "--clients-flops",
        type=float,
        default=8 * 10**9,
        help="Center value for FLOPS distribution (or fixed value if homogeneous)",
    )

    # Heterogeneous Bimodal Options
    parser.add_argument(
        "--bimodal-offset",
        type=float,
        default=6 * 10**9,
        help="Symmetric offset around center_flops (e.g., center=8e9 and offset=6e9 yields mean1=2e9, mean2=14e9)",
    )
    parser.add_argument(
        "--bimodal-mean1",
        type=float,
        default=None,
        help="Explicit mean for mode 1 (overrides center - offset)",
    )
    parser.add_argument(
        "--bimodal-mean2",
        type=float,
        default=None,
        help="Explicit mean for mode 2 (overrides center + offset)",
    )
    parser.add_argument("--bimodal-std1", type=float, default=0.5 * 10**9, help="Std Dev for distribution 1")
    parser.add_argument("--bimodal-std2", type=float, default=2 * 10**9, help="Std Dev for distribution 2")
    parser.add_argument("--bimodal-prob1", type=float, default=0.5, help="Probability weight of choosing distribution 1")

    args = parser.parse_args()

    # Initialize random generator
    rng = np.random.default_rng(args.seed)

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)

    # Ensure output directory exists every run
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting system metrics from the given directory...")
    try:
        sys_stats_files = find_leaf_stats_files(sample_dir, args.search_pattern)
    except Exception as error:
        print(f"Error finding files: {error}", file=sys.stderr)
        return

    if not sys_stats_files:
        print(f"No files matching pattern '{args.search_pattern}' found in '{sample_dir}'.")
        return

    # Calculate mean_1 and mean_2 centered around args.clients_flops
    center_flops = args.clients_flops
    mean1 = args.bimodal_mean1 if args.bimodal_mean1 is not None else (center_flops - args.bimodal_offset)
    mean2 = args.bimodal_mean2 if args.bimodal_mean2 is not None else (center_flops + args.bimodal_offset)

    if args.flops_mode == "heterogeneous":
        print(f"Bimodal Config -> Center: {center_flops:.2e} | Mean 1: {mean1:.2e} | Mean 2: {mean2:.2e}")

    print("Starting data processing...")

    for file_path in sys_stats_files:
        df = pd.read_csv(
            file_path,
            names=[
                "client_id",
                "round_number",
                "hierarchy",
                "num_samples",
                "set",
                "bytes_written",
                "bytes_sended",
                "local_computations",
            ],
        )

        # Drop unused columns
        df = df.drop(columns=["hierarchy", "bytes_written"])

        # Map client_id to unique integers per round
        df["client_id"] = df.groupby(["round_number"])["client_id"].transform(
            lambda x: pd.factorize(x)[0] + 1
        )

        # Assign FLOPS based on scenario
        if args.flops_mode == "homogeneous":
            flops = center_flops
        else:
            # Map unique FLOPS per unique client so hardware capacity stays constant across rounds
            unique_clients = df["client_id"].unique()
            generated_flops = generate_bimodal_flops(
                count=len(unique_clients),
                mean1=mean1,
                std1=args.bimodal_std1,
                mean2=mean2,
                std2=args.bimodal_std2,
                prob1=args.bimodal_prob1,
                rng=rng,
            )
            client_flops_map = dict(zip(unique_clients, generated_flops))
            flops = df["client_id"].map(client_flops_map)

        # Calculate time in seconds
        df["time"] = df["local_computations"] / flops

        # Save to output directory
        output_file_path = output_dir / file_path.name
        df.to_csv(output_file_path, index=False)

        print(f"Processed: '{file_path.name}' -> '{output_file_path}'")


if __name__ == "__main__":
    main()