"""
Plotting utilities for EPF predictions.

These helpers load saved prediction CSVs, normalize prediction dictionaries for
multi-day horizons, and produce comparison plots that mirror the visuals used
in the 04 notebooks.
"""

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def build_prediction_path(
    predictions_dir: Union[str, Path],
    market: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Path:
    """
    Construct the expected prediction file path for a market and horizon.
    """
    start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
    if end_date is None:
        filename = f"predictions_{market}_{start_str}.csv"
    else:
        end_str = pd.to_datetime(end_date).strftime("%Y%m%d")
        filename = f"predictions_{market}_{start_str}_to_{end_str}.csv"
    return Path(predictions_dir) / filename


def load_prediction_csv(
    predictions_dir: Union[str, Path],
    market: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Load a saved prediction CSV for a given market and target horizon.
    """
    csv_path = build_prediction_path(predictions_dir, market, start_date, end_date)
    df = pd.read_csv(csv_path)
    if "Hour" in df.columns:
        df = df.sort_values("Hour").reset_index(drop=True)
    return df


def dataframe_to_predictions(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convert a prediction dataframe into a dictionary keyed by model name.
    The Hour column (if present) is dropped.
    """
    drop_cols = [col for col in ["Hour"] if col in df.columns]
    core = df.drop(columns=drop_cols)
    return {col: core[col].to_numpy() for col in core.columns}


def prepare_predictions_for_plotting(
    predictions: Mapping[str, Sequence[Union[float, int]]],
) -> Dict[str, np.ndarray]:
    """
    Normalize predictions for plotting.

    - Flattens all arrays
    - Concatenates multi-day keys like 'YYYY-MM-DD_MODEL' into a single series
    - Aligns lengths so every series matches the target horizon
    """
    plot_date_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)$")

    grouped = {}
    passthrough = {}

    for key, values in predictions.items():
        if key.lower() == "hour":
            continue
        arr = np.asarray(values).flatten()
        match = plot_date_pattern.match(key)
        if match:
            date_str, base_name = match.groups()
            ts = pd.to_datetime(date_str)
            grouped.setdefault(base_name, []).append((ts, arr))
        else:
            passthrough[key] = arr

    processed = dict(passthrough)
    for base_name, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        concatenated = np.concatenate([a for _, a in items_sorted])
        processed[base_name] = concatenated

    if "actual" in processed:
        target_len = len(processed["actual"])
    elif processed:
        target_len = len(next(iter(processed.values())))
    else:
        target_len = 0

    aligned = {}
    for name, arr in processed.items():
        if target_len and len(arr) != target_len:
            continue
        aligned[name] = arr

    return aligned


def _get_lear_window(model_name: str) -> Optional[int]:
    """Extract calibration window number from a LEAR model name."""
    match = re.search(r"LEAR[^0-9]*(\d+)", model_name)
    if match:
        return int(match.group(1))
    return None


def _get_dnn_index(model_name: str) -> Optional[int]:
    """Extract DNN index from a model name (supports variants like 'DNN 4years')."""
    match = re.search(r"DNN[^0-9]*(\d+)", model_name)
    if match:
        return int(match.group(1))
    return None


def _get_model_color(model_name: str, prediction_horizon_days: Optional[int] = None) -> str:
    """
    Consistent color mapping for LEAR and DNN models.

    - LEAR colors vary by both calibration window and prediction horizon.
    - DNN indices map to fixed palette; non-numeric DNN names fall back but stay distinct.
    
    Parameters
    ----------
    model_name : str
        Model name (e.g., 'LEAR_1456', 'DNN_4years')
    prediction_horizon_days : int, optional
        Prediction horizon in days (1, 2, etc.). If None, uses base colors.
    """
    # Base colors for LEAR calibration windows (1-day prediction)
    lear_base_colors = {
        56: "#1f77b4",    # blue
        84: "#ff7f0e",    # orange
        1092: "#2ca02c",  # green
        1456: "#d62728",  # red
    }
    
    # Color variations for different prediction horizons
    # These are lighter/darker shades of the base colors
    horizon_variations = {
        1: 1.0,      # Full intensity (base color)
        2: 0.75,     # Slightly lighter
        3: 0.6,      # Lighter
        4: 0.5,      # Even lighter
    }
    
    dnn_colors = {
        1: "#9467bd",  # purple
        2: "#8c564b",  # brown
        3: "#e377c2",  # pink
        4: "#7f7f7f",  # gray
    }

    lear_window = _get_lear_window(model_name)
    if lear_window and lear_window in lear_base_colors:
        base_color = lear_base_colors[lear_window]
        
        # Apply prediction horizon variation if provided
        if prediction_horizon_days is not None and prediction_horizon_days > 1:
            # Get variation factor (default to 0.7 for horizons > 4 days)
            variation = horizon_variations.get(prediction_horizon_days, 0.7)
            
            # Convert hex to RGB, apply variation, convert back
            hex_color = base_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            # Blend with white to lighten
            rgb_varied = tuple(int(255 - (255 - c) * variation) for c in rgb)
            return f"#{rgb_varied[0]:02x}{rgb_varied[1]:02x}{rgb_varied[2]:02x}"
        
        return base_color

    if "DNN" in model_name:
        dnn_idx = _get_dnn_index(model_name)
        if dnn_idx and dnn_idx in dnn_colors:
            return dnn_colors[dnn_idx]
        # Rotate through the palette to keep DNNs visually distinct even if unnamed
        fallback_keys = sorted(dnn_colors.keys())
        return dnn_colors[fallback_keys[hash(model_name) % len(fallback_keys)]]

    return "#17becf"  # default cyan for other models


def plot_market_predictions(
    predictions: Mapping[str, Sequence[Union[float, int]]],
    market: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    n_days: Optional[int] = None,
    published_benchmarks: Optional[Mapping[str, Sequence[Union[float, int]]]] = None,
    price_label: str = "Price ($/MWh)",
    figsize: tuple = (20, 8),
):
    """
    Plot predictions vs actuals for a market, matching the style from the 04 notebooks.

    Either `end_date` or `n_days` can be provided to set the horizon in the title.
    """
    processed_predictions = prepare_predictions_for_plotting(predictions)
    if not processed_predictions:
        raise ValueError("No predictions available for plotting.")

    df_plot = pd.DataFrame(processed_predictions)

    y_min = df_plot.min().min() if len(df_plot) > 0 else 0
    y_max = df_plot.max().max() if len(df_plot) > 0 else 50
    y_padding = (y_max - y_min) * 0.1
    y_range = [y_min - y_padding, y_max + y_padding]

    # Calculate prediction horizon in days
    if n_days is not None:
        prediction_horizon_days = n_days
    else:
        # Infer from data length (round up if not exact multiple of 24)
        prediction_horizon_days = (len(df_plot) + 23) // 24

    if end_date is None and n_days is not None:
        end_date = pd.to_datetime(start_date) + pd.Timedelta(days=n_days - 1)
    title_suffix = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        if end_date.date() != pd.to_datetime(start_date).date():
            total_days = (end_date - pd.to_datetime(start_date)).days + 1
            title_suffix = f"{title_suffix} to {end_date.strftime('%Y-%m-%d')} ({total_days} days)"

    benchmark_plot = prepare_predictions_for_plotting(published_benchmarks) if published_benchmarks else None

    if benchmark_plot is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        if "actual" in df_plot.columns:
            ax1.plot(df_plot["actual"], label="Actual", linestyle="-", linewidth=2.5, color="black")

        for column in df_plot.columns:
            if column == "actual":
                continue
            color = _get_model_color(column, prediction_horizon_days)
            ax1.plot(df_plot[column], label=column.replace("_", " "), linestyle="--", alpha=0.7, linewidth=1.5, color=color)

        ax1.set_xlabel("Hour", fontsize=12)
        ax1.set_ylabel(price_label, fontsize=12)
        ax1.set_title(f"Current Estimates - {market} - {title_suffix}", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(y_range)

        h1, l1 = ax1.get_legend_handles_labels()
        lear_handles, lear_labels, dnn_handles, dnn_labels, other_handles, other_labels = [], [], [], [], [], []
        for h, lbl in zip(h1, l1):
            if lbl.startswith("LEAR"):
                lear_handles.append(h); lear_labels.append(lbl)
            elif lbl.startswith("DNN"):
                dnn_handles.append(h); dnn_labels.append(lbl)
            elif lbl == "Actual":
                other_handles.append(h); other_labels.append(lbl)
            else:
                other_handles.append(h); other_labels.append(lbl)
        leg1 = ax1.legend(other_handles + lear_handles, other_labels + lear_labels,
                          loc="upper left", fontsize=9, ncol=1, frameon=True)
        ax1.add_artist(leg1)
        if dnn_handles:
            ax1.legend(dnn_handles, dnn_labels, loc="upper right", fontsize=9, ncol=1, frameon=True)

        if "actual" in df_plot.columns:
            ax2.plot(df_plot["actual"], label="Actual", linestyle="-", linewidth=2.5, color="black")

        for model_name, values in benchmark_plot.items():
            if len(values) == len(df_plot):
                color = _get_model_color(model_name, prediction_horizon_days)
                ax2.plot(values, label=model_name, linestyle="--", alpha=0.7, linewidth=1.5, color=color)

        ax2.set_xlabel("Hour", fontsize=12)
        ax2.set_title(f"Published Benchmarks - {market} - {title_suffix}", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(y_range)

        h2, l2 = ax2.get_legend_handles_labels()
        lear_handles2, lear_labels2, dnn_handles2, dnn_labels2, other_handles2, other_labels2 = [], [], [], [], [], []
        for h, lbl in zip(h2, l2):
            if lbl.startswith("LEAR"):
                lear_handles2.append(h); lear_labels2.append(lbl)
            elif lbl.startswith("DNN"):
                dnn_handles2.append(h); dnn_labels2.append(lbl)
            elif lbl == "Actual":
                other_handles2.append(h); other_labels2.append(lbl)
            else:
                other_handles2.append(h); other_labels2.append(lbl)
        leg2 = ax2.legend(other_handles2 + lear_handles2, other_labels2 + lear_labels2,
                          loc="upper left", fontsize=9, ncol=1, frameon=True)
        ax2.add_artist(leg2)
        if dnn_handles2:
            ax2.legend(dnn_handles2, dnn_labels2, loc="upper right", fontsize=9, ncol=1, frameon=True)
    else:
        fig, ax = plt.subplots(figsize=(14, 8))

        if "actual" in df_plot.columns:
            ax.plot(df_plot["actual"], label="Actual", linestyle="-", linewidth=2.5, color="black")

        for column in df_plot.columns:
            if column == "actual":
                continue
            color = _get_model_color(column, prediction_horizon_days)
            ax.plot(df_plot[column], label=column.replace("_", " "), linestyle="--", alpha=0.7, linewidth=1.5, color=color)

        ax.set_xlabel("Hour", fontsize=12)
        ax.set_ylabel(price_label, fontsize=12)
        ax.set_title(f"Forecast Comparison for {market} - {title_suffix}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_range)

        handles, labels = ax.get_legend_handles_labels()
        lear_h, lear_l, dnn_h, dnn_l, other_h, other_l = [], [], [], [], [], []
        for h, lbl in zip(handles, labels):
            if lbl.startswith("LEAR"):
                lear_h.append(h); lear_l.append(lbl)
            elif lbl.startswith("DNN"):
                dnn_h.append(h); dnn_l.append(lbl)
            elif lbl == "Actual":
                other_h.append(h); other_l.append(lbl)
            else:
                other_h.append(h); other_l.append(lbl)
        legA = ax.legend(other_h + lear_h, other_l + lear_l, loc="upper left", fontsize=10, ncol=1, frameon=True)
        ax.add_artist(legA)
        if dnn_h:
            ax.legend(dnn_h, dnn_l, loc="upper right", fontsize=10, ncol=1, frameon=True)

    plt.tight_layout()
    plt.show()
    return fig


def load_published_benchmarks_for_horizon(
    csv_path: Union[str, Path],
    start_date: Union[str, pd.Timestamp],
    n_days: int,
    model_columns: Optional[Iterable[str]] = None,
) -> Dict[str, Sequence[float]]:
    """
    Load and concatenate published benchmark forecasts over a multi-day horizon.
    """
    from forecast_pipeline import load_published_benchmarks  # lazy import to avoid heavy deps at module import

    target_start = pd.to_datetime(start_date)
    benchmarks: Dict[str, list] = {}

    for day_idx in range(n_days):
        current_date = target_start + pd.Timedelta(days=day_idx)
        day_benchmarks = load_published_benchmarks(
            csv_path=csv_path,
            target_date=current_date,
            model_columns=model_columns,
        )
        for model_name, values in day_benchmarks.items():
            benchmarks.setdefault(model_name, []).extend(values)

    return benchmarks
