"""Dataset loading and representative-sample selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing import normalize_complaint


REQUIRED_RAW_COLUMN = "Consumer complaint narrative"


def load_preprocessed_dataset(
    path: Path,
    text_column: str = "clean_text",
    raw_text_column: str = REQUIRED_RAW_COLUMN,
    minimum_words: int = 10,
) -> pd.DataFrame:
    """Load the preprocessed CFPB dataset and validate the narrative field.

    The function accepts CSV or compressed CSV files. When ``clean_text`` is absent but the
    original CFPB narrative is present, it recreates the normalized text as a safe fallback.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {path}. "
            "Run the Milestone 3 preprocessing notebook or place the prepared file at this path."
        )

    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"The dataset is empty: {path}")

    if text_column not in frame.columns:
        if raw_text_column not in frame.columns:
            raise ValueError(
                f"Expected either '{text_column}' or '{raw_text_column}'. "
                f"Available columns: {sorted(frame.columns)}"
            )
        frame[text_column] = frame[raw_text_column].map(normalize_complaint)

    frame[text_column] = frame[text_column].fillna("").astype(str).str.strip()
    frame = frame[frame[text_column].str.split().str.len().ge(minimum_words)].copy()
    if frame.empty:
        raise ValueError("No usable narratives remain after validation.")

    return frame.reset_index(drop=True)


def select_representative_samples(
    frame: pd.DataFrame,
    number_of_samples: int,
    seed: int,
    group_column: str = "Sub-product",
    id_column: str = "Complaint ID",
) -> pd.DataFrame:
    """Select deterministic samples across common complaint groups.

    One record is selected from each of the largest groups first, then remaining slots are
    filled from the unselected corpus. This is more representative than a single random draw.
    """
    if number_of_samples < 1:
        raise ValueError("number_of_samples must be at least 1")

    n = min(number_of_samples, len(frame))
    chosen_indices: list[int] = []

    if group_column in frame.columns:
        valid_group = frame[group_column].fillna("Unknown").astype(str)
        group_order = valid_group.value_counts().index.tolist()
        for offset, group_name in enumerate(group_order):
            group_rows = frame.index[valid_group.eq(group_name)]
            if len(group_rows) == 0:
                continue
            selected = frame.loc[group_rows].sample(1, random_state=seed + offset).index[0]
            chosen_indices.append(int(selected))
            if len(chosen_indices) >= n:
                break

    if len(chosen_indices) < n:
        remaining = frame.drop(index=chosen_indices, errors="ignore")
        fill = remaining.sample(n=n - len(chosen_indices), random_state=seed).index.tolist()
        chosen_indices.extend(int(index) for index in fill)

    sample = frame.loc[chosen_indices].copy().reset_index(drop=True)
    if id_column in sample.columns:
        sample[id_column] = sample[id_column].astype(str)
    return sample
