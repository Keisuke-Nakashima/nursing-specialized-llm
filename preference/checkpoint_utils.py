import os
from typing import List, Dict, Tuple, Set, Any

import pandas as pd


def canonical_no(value: Any) -> str:
    """Normalize identifier columns so they are comparable across dtypes."""
    return str(value)


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    missing_cols = [col for col in columns if col not in df.columns]
    for col in missing_cols:
        df[col] = None
    return df[columns]


def load_checkpoint_data(path: str, columns: List[str]) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load checkpoint CSV if present so the evaluation can resume."""
    if not os.path.exists(path):
        return [], set()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return [], set()
    if df.empty:
        return [], set()
    df = _ensure_columns(df, columns)
    processed = set(df["no"].astype(str).tolist())
    print(f"Loaded {len(df)} records from the existing checkpoint (to be skipped)")
    return df.to_dict("records"), processed


def save_checkpoint_data(data: List[Dict[str, Any]], path: str, columns: List[str]) -> None:
    """Persist progress so long jobs can be resumed without rerunning prior rows."""
    if not data:
        return
    df = pd.DataFrame(data)
    df = _ensure_columns(df, columns)
    df.to_csv(path, index=False)
    print(f"Checkpoint saved: {path} ({len(df)} records in total)")
