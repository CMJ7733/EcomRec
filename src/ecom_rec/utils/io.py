from pathlib import Path
import polars as pl


def read_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
