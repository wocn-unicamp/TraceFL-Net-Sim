import argparse
import fnmatch
from pathlib import Path
import sys
import pandas as pd


def find_leaf_stats_files(directory: Path, pattern: str) -> list[Path]:
    """Find files matching the pattern in the given directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Input directory '{directory}' does not exist.")

    return [
        f for f in directory.iterdir() if f.is_file() and fnmatch.fnmatch(f.name, pattern)
    ]


def main():
    parser = argparse.ArgumentParser(description="Process federated system metrics CSVs.")

    # Directory and file pattern arguments
    parser.add_argument("--sample-dir", type=str, default="traces/sys/", help="Input directory")
    parser.add_argument("--output-dir", type=str, default="output_traces/", help="Output directory")
    parser.add_argument("--search-pattern", type=str, default="sys_metrics_*", help="File match pattern")

    # FLOPS Configuration
    parser.add_argument(
        "--flops-mode",
        type=str,
        choices=["homogeneous", "heterogeneous"],
        default="homogeneous",
        help="FLOPS distribution mode across clients",
    )

    parser.add_argument(
        "--clients-flops",
        type=float,
        default=8 * 10**9,
        help="Center value for FLOPS distribution (or fixed value if homogeneous)",
    )


    args = parser.parse_args()

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

    print("Starting data processing...")

    for file_path in sys_stats_files:
        df = pd.read_csv(
            file_path,
            header=None,
            nrows=1,
        )

        if df.empty:
            print(f"Warning: '{file_path.name}' is empty. Skipping.")
            continue

        unused_columns = []

        match args.flops_mode:
            case "homogeneous":
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

                unused_columns.extend(("hierarchy", "bytes_written"))

                df["time"] = df["local_computations"] / args.clients_flops

            case "heterogeneous":
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
                        "capacity_gflops",
                        "time",
                    ],
                )

                unused_columns.extend(("hierarchy", "bytes_written", "capacity_gflops"))

        df = df.drop(columns=unused_columns)

        # Map client_id to unique integers per round
        df["client_id"] = df.groupby(["round_number"])["client_id"].transform(
            lambda x: pd.factorize(x)[0] + 1
        )

        # Save to output directory
        output_file_path = output_dir / file_path.name
        df.to_csv(output_file_path, index=False)

        print(f"Processed: '{file_path.name}' -> '{output_file_path}'")


if __name__ == "__main__":
    main()