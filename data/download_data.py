"""
Script includes functions used for downloading and processing images from the LAION-400M dataset. 
"""

import timeit

from pathlib import Path
from img2dataset import download


def download_round(
    round_id: int,
    metadata_dir: str,
    cache_root: str = "laion-cache",
    image_size: int = 256) -> Path:

    """
    Function to download a round of images from the LAION-400M dataset using the img2dataset library.
    @author: Stephen Krol

    :param round_id: ID of the download round (used for naming the output directory).
    :type round_id: int
    :param metadata_dir: Path to the input parquet file containing the filtered LAION-400 metadata (output from process_subset.py).
    :type metadata_dir: str
    :param cache_root: Root directory for caching downloaded images (default: "laion-cache").
    :type cache_root: str
    :param image_size: Size to which downloaded images will be resized (default: 256 for 256x256 images).
    :type image_size: int

    :return: Path to the directory containing the downloaded images in webdataset format.
    :rtype: Path
    """


    round_name = f"round-{round_id:04d}"

    output_dir = Path(cache_root) / "incoming" / round_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # track time taken for download
    start_time = timeit.default_timer()

    download(
        url_list=metadata_dir,
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",

        output_folder=str(output_dir),
        output_format="webdataset",

        image_size=image_size,
        resize_mode="center_crop",
        encode_format="jpg",
        encode_quality=95,

        number_sample_per_shard=10_000,
        min_image_size=256,
        max_aspect_ratio=3,

        processes_count=8,
        thread_count=64,

        incremental_mode="incremental",
    )

    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    print(f"Download completed in {elapsed_time:.2f} seconds.")

    return output_dir


if __name__ == "__main__":

    download_round(
        round_id=1,
        metadata_dir="laion400m_filtered/subset_000000000.parquet",
        cache_root="laion-cache",
    )