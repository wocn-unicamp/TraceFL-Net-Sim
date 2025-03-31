from os import listdir, path, makedirs
import re
import pandas as pd
import argparse

def find_leaf_stats_files(directory:str, pattern:str) -> list[str]:
    regex = re.compile(pattern)

    stats_files:set[str] = set()

    for filename in listdir(directory):
        if filename.endswith('.csv') and regex.match(filename):
            stats_files.add(filename)
    
    return list(stats_files)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--search-pattern", type=str, required=True)
    parser.add_argument("--clients-flops", type=int, default=8 * 10**9)

    args = parser.parse_args()

    print("Colleting system metrics from the given directory...")

    try:
        sys_stats_filenames = find_leaf_stats_files(args.sample_dir, args.search_pattern)
    except Exception as error:
        print(f"Unable to find expected files: {error}")

    print("Starting data processing")

    for sys_stats in sys_stats_filenames:
        df = pd.read_csv(
            args.sample_dir + sys_stats,
            names=[
                "client_id",
                "round_number",
                "hierarchy",
                "num_samples",
                "set",
                "bytes_written",
                "bytes_sended",
                "local_computations"
            ]
        )

        # A primeira coluna é sempre vazia e a segunda é igual a bytes_sended
        df = df.drop(["hierarchy", "bytes_written"], axis=1)

        # Map client_id to unique integers based on round_number
        df['client_id'] = df.groupby(['round_number'])['client_id'].transform(
            lambda x: pd.factorize(x)[0] + 1
        )

        splits = sys_stats.split("_")

        # response time in miliseconds
        df['time'] = (df['local_computations'] / args.clients_flops)
        
        # Prepare output file path
        output_file_path = path.join(args.output_dir, path.basename(sys_stats))

        # Ensure the output directory exists
        makedirs(args.output_dir, exist_ok=True)

        # Save the modified dataframe to CSV in the output directory
        df.to_csv(output_file_path, index=False)

        print(f"Data from {sys_stats} was processed")

if __name__ == "__main__":
    main()