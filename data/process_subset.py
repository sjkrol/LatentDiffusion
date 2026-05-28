"""
Script loads metadata from the LAION-400M dataset, filters it based on certain criteria, and saves the filtered subset to a new CSV file. The filtering criteria include:
- The image must have a width and height of at least 256 pixels.
- The NSFW content must be labeled as "unlikely", "0", or "false" (case-insensitive).

"""

import argparse
import pandas as pd

from pathlib import Path
from tqdm import tqdm

def filter_laion_subset(input_parquet_dir: str, 
                        output_parquet_dir: str,
                        target_total: int,
                        rows_per_parquet: int) -> None:
    """
    Function filters the metadata of the LAION-400M dataset based on image dimensions and saves a subset to a new parquet file.
    @author: Stephen Krol

    :param input_parquet_dir: Path to the input parquet directory containing the LAION-400M metadata.
    :type input_parquet_dir: str
    :param output_parquet_dir: Path to save the filtered subset parquet file.
    :type output_parquet_dir: str
    :param target_total: Target total number of samples in the filtered subset.
    :type target_total: int
    :param rows_per_parquet: Number of rows to include per output parquet file (for memory efficiency).
    :type rows_per_parquet: int

    :return: None
    """

    src = Path(input_parquet_dir)
    dst = Path(output_parquet_dir)
    dst.mkdir(exist_ok=True)

    rows_written = 0

    for paraquet in tqdm(src.glob("*.parquet"), desc="Processing parquet files"):
        
        # Load the metadata parquet file
        df = pd.read_parquet(paraquet)

        # filter nsfw content
        df = df[df["NSFW"].astype(str).str.lower().isin(["unlikely", "0", "false"])]

        # Filter based on image dimensions (width and height >= 256)
        df = df[(df['WIDTH'] >= 256) & (df['HEIGHT'] >= 256)]

        # Shuffle so early shards are not biased by source ordering.
        df = df.sample(frac=1, random_state=42)

        # Save the filtered subset to a new parquet file
        while len(df) and rows_written < target_total:

            take = min(rows_per_parquet, target_total - rows_written, len(df))
            out = df.iloc[:take]
            out.to_parquet(dst / f"subset_{rows_written:09d}.parquet", index=False)
            df = df.iloc[take:]
            rows_written += take
        
        if rows_written >= target_total:
            break
    
    print(f"wrote {rows_written:,} rows")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a subset of the LAION-400M dataset.")
    parser.add_argument("--input-parquet-dir", type=str, default="data/laion400m/laion-400m-metadata.parquet", help="Path to the input parquet directory containing the LAION-400M metadata.")
    parser.add_argument("--output-parquet-dir", type=str, default="data/laion400m/laion-400m-subset.parquet", help="Path to save the filtered subset parquet files.")
    parser.add_argument("--target-total", type=int, default=1000000, help="Target total number of samples in the filtered subset.")
    parser.add_argument("--rows-per-parquet", type=int, default=10000, help="Number of rows to include per output parquet file (for memory efficiency).")

    args = parser.parse_args()

    filter_laion_subset(args.input_parquet_dir, args.output_parquet_dir, args.target_total, args.rows_per_parquet)

