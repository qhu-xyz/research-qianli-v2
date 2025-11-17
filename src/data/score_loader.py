"""Load score features from MISO density data.

This module provides functionality to load score.parquet files from the MISO
density data directory structure.
"""

import glob
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


class ScoreLoader:
    """Load and process score.parquet files from MISO density data.

    The density data is organized as:
    density/
    ├── auction_month=YYYY-MM/
    │   ├── market_month=YYYY-MM/
    │   │   ├── market_round={1,2,3,...}/
    │   │   │   ├── outage_date=YYYY-MM-DD/
    │   │   │   │   ├── score.parquet
    │   │   │   │   ├── density.parquet
    │   │   │   │   └── limit.parquet

    Each outage_date represents a 3-day period starting from that date.
    """

    def __init__(self, base_path: str | Path):
        """Initialize score loader.

        Parameters
        ----------
        base_path : str | Path
            Base path to density data directory
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")

    def load_score_file(self, score_path: Path) -> pd.DataFrame:
        """Load a single score.parquet file.

        Parameters
        ----------
        score_path : Path
            Path to score.parquet file

        Returns
        -------
        pd.DataFrame
            Score data with partition columns added
        """
        # Extract partition information from path
        parts = score_path.parts
        partition_info = {}

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                partition_info[key] = value

        # Load parquet file
        df = pd.read_parquet(score_path)

        # Add partition columns
        for key, value in partition_info.items():
            df[key] = value

        return df

    def load_auction_month(
        self,
        auction_month: str,
        market_months: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Load all score files for a given auction month.

        Parameters
        ----------
        auction_month : str
            Auction month in format 'YYYY-MM'
        market_months : list[str], optional
            Filter to specific market months. If None, load all.

        Returns
        -------
        pd.DataFrame
            Combined score data for the auction month

        Examples
        --------
        >>> loader = ScoreLoader('/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density')
        >>> df = loader.load_auction_month('2024-01')
        """
        auction_path = self.base_path / f"auction_month={auction_month}"

        if not auction_path.exists():
            raise ValueError(f"Auction month {auction_month} not found")

        # Find all score.parquet files for this auction month
        pattern = str(auction_path / "**" / "score.parquet")
        score_files = glob.glob(pattern, recursive=True)

        if not score_files:
            raise ValueError(f"No score files found for {auction_month}")

        # Load all files
        dfs = []
        for file_path in tqdm(score_files, desc=f"Loading {auction_month}"):
            df = self.load_score_file(Path(file_path))

            # Filter by market_month if specified
            if market_months is not None:
                if df["market_month"].iloc[0] not in market_months:
                    continue

            dfs.append(df)

        if not dfs:
            raise ValueError(
                f"No matching score files found for {auction_month} "
                f"with market_months={market_months}"
            )

        # Concatenate all dataframes
        result = pd.concat(dfs, ignore_index=True)

        return result

    def load_date_range(
        self,
        start_date: str,
        end_date: str,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Load all score files within a date range.

        Parameters
        ----------
        start_date : str
            Start date in format 'YYYY-MM-DD'
        end_date : str
            End date in format 'YYYY-MM-DD'
        show_progress : bool, default True
            Show progress bar

        Returns
        -------
        pd.DataFrame
            Combined score data for the date range

        Examples
        --------
        >>> loader = ScoreLoader('/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density')
        >>> df = loader.load_date_range('2024-01-01', '2024-03-31')
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate list of auction months to load
        auction_months = pd.date_range(
            start_dt.replace(day=1), end_dt, freq="MS"
        ).strftime("%Y-%m").tolist()

        # Load each auction month
        dfs = []
        for auction_month in tqdm(
            auction_months, desc="Loading auction months", disable=not show_progress
        ):
            try:
                df = self.load_auction_month(auction_month)
                dfs.append(df)
            except ValueError as e:
                print(f"Warning: {e}")
                continue

        if not dfs:
            raise ValueError(f"No data found for date range {start_date} to {end_date}")

        # Concatenate all dataframes
        result = pd.concat(dfs, ignore_index=True)

        # Filter by outage_date to match exact date range
        result["outage_date_dt"] = pd.to_datetime(result["outage_date"])
        result = result[
            (result["outage_date_dt"] >= start_dt)
            & (result["outage_date_dt"] <= end_dt)
        ].copy()

        if len(result) == 0:
            raise ValueError(
                f"No data found after filtering by date range {start_date} to {end_date}"
            )

        return result

    def get_available_constraints(self, df: pd.DataFrame) -> list[str]:
        """Get list of unique constraint IDs in the data.

        Parameters
        ----------
        df : pd.DataFrame
            Score dataframe

        Returns
        -------
        list[str]
            Sorted list of unique constraint IDs
        """
        if "constraint_id" not in df.columns:
            raise ValueError("DataFrame does not have 'constraint_id' column")

        return sorted(df["constraint_id"].unique().tolist())

    def get_coverage_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get data coverage statistics by constraint.

        Parameters
        ----------
        df : pd.DataFrame
            Score dataframe

        Returns
        -------
        pd.DataFrame
            Coverage statistics with columns:
            - constraint_id
            - n_records
            - n_auction_months
            - n_market_months
            - first_date
            - last_date
        """
        if "constraint_id" not in df.columns:
            raise ValueError("DataFrame does not have 'constraint_id' column")

        stats = (
            df.groupby("constraint_id")
            .agg(
                n_records=("constraint_id", "count"),
                n_auction_months=("auction_month", "nunique"),
                n_market_months=("market_month", "nunique"),
                first_date=("outage_date", "min"),
                last_date=("outage_date", "max"),
            )
            .reset_index()
        )

        return stats.sort_values("n_records", ascending=False)


def load_scores(
    base_path: str | Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Convenience function to load scores for a date range.

    Parameters
    ----------
    base_path : str | Path
        Base path to density data
    start_date : str
        Start date 'YYYY-MM-DD'
    end_date : str
        End date 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        Score data for the date range

    Examples
    --------
    >>> df = load_scores(
    ...     '/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density',
    ...     '2024-01-01',
    ...     '2024-03-31'
    ... )
    """
    loader = ScoreLoader(base_path)
    return loader.load_date_range(start_date, end_date)
