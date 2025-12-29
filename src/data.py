from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

WHOLESALE_CSV_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"


@dataclass(frozen=True)
class Paths:
    raw: Path = Path("data/raw")

    def ensure(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)


def load_wholesale(paths: Paths) -> pd.DataFrame:
    """
    Download once (if needed) then load the Wholesale customers dataset.
    """
    paths.ensure()
    local = paths.raw / "wholesale_customers.csv"
    if not local.exists():
        print(f"[download] {WHOLESALE_CSV_URL} -> {local}")
        df = pd.read_csv(WHOLESALE_CSV_URL)
        df.to_csv(local, index=False)
    else:
        df = pd.read_csv(local)
    return df
