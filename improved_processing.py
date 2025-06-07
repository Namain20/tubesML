"""Efficient Netflix dataset processing utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd

DATASETS = [
    "shivamb/netflix-shows",
    "ruchi798/movies-on-netflix-prime-video-hulu-and-disney",
    "victorsoeiro/netflix-tv-shows-and-movies",
    "rahulvyasm/netflix-movies-and-tv-shows",
    "asaniczka/amazon-india-products-2023-1-5m-products",
    "asaniczka/amazon-brazil-products-2023-1-3m-products",
    "jillanisofttech/amazon-product-reviews",
    "asaniczka/amazon-canada-products-2023-2-1m-products",
]


def download_kaggle_datasets() -> Dict[str, Path]:
    """Download Kaggle datasets defined in :data:`DATASETS`."""
    import kagglehub

    paths: Dict[str, Path] = {}
    for dataset_id in DATASETS:
        paths[dataset_id] = kagglehub.dataset_download(dataset_id)
    return paths


def preprocess_netflix(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Netflix dataset using vectorized operations."""
    df = df.copy()

    df["rating"] = df["rating"].fillna(df["rating"].mode().iloc[0])
    df["country"] = df["country"].fillna("Unknown")
    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")

    df["date_added"] = pd.to_datetime(
        df["date_added"].fillna(df["date_added"].mode().iloc[0]),
        format="%B %d, %Y",
    )
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    duration = df["duration"].fillna("")
    df["duration_minutes"] = (
        duration.str.extract(r"(\d+)\s*min", expand=False).astype(float).fillna(0)
    )
    df["num_seasons"] = (
        duration.str.extract(r"(\d+)\s*Season", expand=False).astype(float).fillna(0)
    )

    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Netflix dataset")
    parser.add_argument("csv", type=Path, help="Path to netflix_titles.csv")
    parser.add_argument("--out", type=Path, help="Path to save processed CSV")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.csv)
    df_processed = preprocess_netflix(df_raw)

    if args.out:
        df_processed.to_csv(args.out, index=False)
    else:
        print(df_processed.head())
