"""Plot diagnostics for the SLM VTEC product."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = "datasets/processed/rayvtec_china_20240505_slm.parquet"
DEFAULT_FIGURE_DIR = "results/figures"
DEFAULT_TABLE_DIR = "results/tables"
ELEVATION_BINS = [10, 15, 20, 30, 45, 60, 90]
ELEVATION_LABELS = ["10-15", "15-20", "20-30", "30-45", "45-60", "60-90"]


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a parquet or CSV table."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_json(data: dict[str, Any], path: Path) -> Path:
    """Write a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def configure_matplotlib():
    """Configure matplotlib for compact paper-style diagnostic figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def validate_input(
    df: pd.DataFrame,
    *,
    vtec_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """Validate required columns and return rows usable for plotting."""

    required = [
        timestamp_column,
        "ipp_lon",
        "ipp_lat",
        "station",
        "elevation",
        vtec_column,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input table is missing required columns: {missing}")

    out = df.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column])
    numeric_columns = ["ipp_lon", "ipp_lat", "elevation", vtec_column]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    valid = out[required].notna().all(axis=1)
    finite = np.isfinite(out[numeric_columns].to_numpy(float)).all(axis=1)
    out = out.loc[valid & finite].copy()
    if out.empty:
        raise ValueError("No valid rows remain for VTEC diagnostics.")
    return out


def date_label_from_timestamps(ts: pd.Series) -> str:
    """Return a compact date label for output filenames."""

    days = pd.to_datetime(ts).dt.strftime("%Y%m%d").dropna().unique()
    if len(days) == 1:
        return str(days[0])
    return f"{days.min()}_{days.max()}"


def choose_map_timestamp(
    df: pd.DataFrame,
    *,
    timestamp_column: str,
    vtec_column: str,
    min_points: int,
) -> pd.Timestamp:
    """Choose a data-rich timestamp with peak median VTEC for the map."""

    grouped = df.groupby(timestamp_column)[vtec_column].agg(["count", "median"])
    eligible = grouped.loc[grouped["count"] >= min_points]
    if eligible.empty:
        eligible = grouped
    return pd.Timestamp(eligible["median"].idxmax())


def save_figure(fig, path_base: Path, *, cartopy: bool = False) -> list[str]:
    """Save a figure as PNG and SVG and return paths."""

    paths = []
    for suffix in [".png", ".svg"]:
        path = path_base.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if cartopy:
            fig.savefig(path)
        else:
            fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    return paths


def add_cartopy_background(ax, extent: list[float]) -> None:
    """Add Cartopy land, coastline, and border layers."""

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#eeeeee", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f8fbff", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, zorder=1)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linestyle=":",
        linewidth=0.4,
        zorder=1,
    )


def plot_plain_map(
    plt,
    map_df: pd.DataFrame,
    *,
    output_base: Path,
    extent: list[float],
    vtec_column: str,
    selected_timestamp: pd.Timestamp,
    vmin: float,
    vmax: float,
) -> list[str]:
    """Plot a fallback longitude-latitude scatter map without Cartopy."""

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, linewidth=0.35, color="#d0d0d0", linestyle="--", zorder=0)
    sc = ax.scatter(
        map_df["ipp_lon"],
        map_df["ipp_lat"],
        c=map_df[vtec_column],
        cmap="turbo",
        s=9,
        alpha=0.9,
        linewidths=0,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("SLM VTEC (TECU)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"SLM VTEC at {selected_timestamp:%Y-%m-%d %H:%M UTC}")
    paths = save_figure(fig, output_base, cartopy=False)
    plt.close(fig)
    return paths


def plot_vtec_map(
    plt,
    df: pd.DataFrame,
    *,
    output_dir: Path,
    date_label: str,
    timestamp_column: str,
    vtec_column: str,
    map_time: str | None,
    min_map_points: int,
    extent: list[float],
) -> tuple[list[str], pd.Timestamp, int, str]:
    """Plot an IPP map colored by SLM VTEC."""

    if map_time:
        selected_timestamp = pd.Timestamp(map_time)
    else:
        selected_timestamp = choose_map_timestamp(
            df,
            timestamp_column=timestamp_column,
            vtec_column=vtec_column,
            min_points=min_map_points,
        )

    map_df = df.loc[df[timestamp_column] == selected_timestamp].copy()
    if map_df.empty:
        raise ValueError(f"No rows found for map timestamp: {selected_timestamp}")

    values = map_df[vtec_column].to_numpy(float)
    vmin, vmax = np.nanpercentile(values, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    time_label = selected_timestamp.strftime("%Y%m%d_%H%M")
    output_base = output_dir / f"vtec_slm_map_{time_label}"

    try:
        import cartopy.crs as ccrs

        fig = plt.figure(figsize=(6.2, 4.8), dpi=150)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        add_cartopy_background(ax, extent)
        sc = ax.scatter(
            map_df["ipp_lon"],
            map_df["ipp_lat"],
            c=map_df[vtec_column],
            cmap="turbo",
            s=9,
            alpha=0.9,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02, shrink=0.88)
        cb.set_label("SLM VTEC (TECU)")
        ax.set_title(f"SLM VTEC at {selected_timestamp:%Y-%m-%d %H:%M UTC}")
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="gray",
            alpha=0.55,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        paths = save_figure(fig, output_base, cartopy=True)
        plt.close(fig)
        return paths, selected_timestamp, int(len(map_df)), "cartopy"
    except Exception:
        plt.close("all")
        paths = plot_plain_map(
            plt,
            map_df,
            output_base=output_base,
            extent=extent,
            vtec_column=vtec_column,
            selected_timestamp=selected_timestamp,
            vmin=vmin,
            vmax=vmax,
        )
        return paths, selected_timestamp, int(len(map_df)), "plain_lon_lat"


def add_elevation_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add a categorical elevation-bin column."""

    out = df.copy()
    out["elevation_bin"] = pd.cut(
        out["elevation"],
        bins=ELEVATION_BINS,
        labels=ELEVATION_LABELS,
        include_lowest=True,
        right=False,
    )
    return out


def plot_elevation_distribution(
    plt,
    df: pd.DataFrame,
    *,
    output_dir: Path,
    date_label: str,
    vtec_column: str,
) -> tuple[list[str], pd.DataFrame]:
    """Plot VTEC distribution by elevation bin."""

    binned = add_elevation_bins(df)
    groups = [
        binned.loc[binned["elevation_bin"] == label, vtec_column].to_numpy(float)
        for label in ELEVATION_LABELS
    ]
    summary = (
        binned.groupby("elevation_bin", observed=True)[vtec_column]
        .agg(
            n="count",
            mean="mean",
            median="median",
            p05=lambda s: s.quantile(0.05),
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p95=lambda s: s.quantile(0.95),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    bp = ax.boxplot(
        groups,
        tick_labels=ELEVATION_LABELS,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "#111111", "linewidth": 1.1},
        boxprops={"linewidth": 0.8, "color": "#333333"},
        whiskerprops={"linewidth": 0.8, "color": "#555555"},
        capprops={"linewidth": 0.8, "color": "#555555"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.8)
    ax.set_xlabel("Elevation angle bin (deg)")
    ax.set_ylabel("SLM VTEC (TECU)")
    ax.set_title(f"SLM VTEC by elevation bin ({date_label})")
    ax.grid(axis="y", linewidth=0.35, color="#d0d0d0", linestyle="--")
    ax.set_ylim(bottom=0)
    paths = save_figure(fig, output_dir / f"vtec_slm_elevation_distribution_{date_label}")
    plt.close(fig)
    return paths, summary


def plot_station_distribution(
    plt,
    df: pd.DataFrame,
    *,
    output_dir: Path,
    date_label: str,
    vtec_column: str,
    top_stations: int,
) -> tuple[list[str], pd.DataFrame]:
    """Plot station-wise VTEC distribution summaries."""

    summary = (
        df.groupby("station")[vtec_column]
        .agg(
            n="count",
            mean="mean",
            median="median",
            p05=lambda s: s.quantile(0.05),
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p95=lambda s: s.quantile(0.95),
        )
        .sort_values(["n", "median"], ascending=[False, False])
    )
    plot_df = summary.head(top_stations).sort_values("median")

    y = np.arange(len(plot_df))
    fig_height = max(4.0, 0.18 * len(plot_df) + 1.0)
    fig, ax = plt.subplots(figsize=(6.4, fig_height), constrained_layout=True)
    ax.hlines(y, plot_df["p05"], plot_df["p95"], color="#bdbdbd", linewidth=1.4)
    ax.hlines(y, plot_df["p25"], plot_df["p75"], color="#3182bd", linewidth=3.2)
    ax.scatter(plot_df["median"], y, s=16, color="#08519c", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df.index)
    ax.set_xlabel("SLM VTEC (TECU)")
    ax.set_ylabel("Station")
    ax.set_title(f"Station-wise SLM VTEC distribution ({date_label})")
    ax.grid(axis="x", linewidth=0.35, color="#d0d0d0", linestyle="--")
    ax.set_xlim(left=0)
    paths = save_figure(fig, output_dir / f"vtec_slm_station_distribution_{date_label}")
    plt.close(fig)
    return paths, summary.reset_index()


def plot_time_evolution(
    plt,
    df: pd.DataFrame,
    *,
    output_dir: Path,
    date_label: str,
    timestamp_column: str,
    vtec_column: str,
) -> tuple[list[str], pd.DataFrame]:
    """Plot VTEC time evolution with percentile bands."""

    summary = (
        df.groupby(timestamp_column)[vtec_column]
        .agg(
            n="count",
            mean="mean",
            median="median",
            p05=lambda s: s.quantile(0.05),
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            p95=lambda s: s.quantile(0.95),
        )
        .sort_index()
        .reset_index()
    )

    import matplotlib.dates as mdates

    x = pd.to_datetime(summary[timestamp_column])
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.fill_between(x, summary["p05"], summary["p95"], color="#c6dbef", alpha=0.45)
    ax.fill_between(x, summary["p25"], summary["p75"], color="#6baed6", alpha=0.5)
    ax.plot(x, summary["median"], color="#08306b", linewidth=1.4, label="Median")
    ax.plot(x, summary["mean"], color="#de2d26", linewidth=1.0, label="Mean")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("SLM VTEC (TECU)")
    ax.set_title(f"SLM VTEC time evolution ({date_label})")
    ax.grid(axis="both", linewidth=0.35, color="#d0d0d0", linestyle="--")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.set_xlim(x.min(), x.max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    paths = save_figure(fig, output_dir / f"vtec_slm_time_evolution_{date_label}")
    plt.close(fig)
    return paths, summary


def build_vtec_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    """Create VTEC diagnostic figures and tables."""

    plt = configure_matplotlib()
    input_path = Path(args.input)
    output_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    df_raw = read_table(input_path)
    df = validate_input(
        df_raw,
        vtec_column=args.vtec_column,
        timestamp_column=args.timestamp_column,
    )
    date_label = args.date_label or date_label_from_timestamps(df[args.timestamp_column])

    map_paths, selected_time, map_rows, map_backend = plot_vtec_map(
        plt,
        df,
        output_dir=output_dir,
        date_label=date_label,
        timestamp_column=args.timestamp_column,
        vtec_column=args.vtec_column,
        map_time=args.map_time,
        min_map_points=args.min_map_points,
        extent=args.extent,
    )
    elevation_paths, elevation_summary = plot_elevation_distribution(
        plt,
        df,
        output_dir=output_dir,
        date_label=date_label,
        vtec_column=args.vtec_column,
    )
    station_paths, station_summary = plot_station_distribution(
        plt,
        df,
        output_dir=output_dir,
        date_label=date_label,
        vtec_column=args.vtec_column,
        top_stations=args.top_stations,
    )
    time_paths, time_summary = plot_time_evolution(
        plt,
        df,
        output_dir=output_dir,
        date_label=date_label,
        timestamp_column=args.timestamp_column,
        vtec_column=args.vtec_column,
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    elevation_table = table_dir / f"vtec_slm_elevation_diagnostic_{date_label}.csv"
    station_table = table_dir / f"vtec_slm_station_diagnostic_{date_label}.csv"
    time_table = table_dir / f"vtec_slm_time_diagnostic_{date_label}.csv"
    elevation_summary.to_csv(elevation_table, index=False)
    station_summary.to_csv(station_table, index=False)
    time_summary.to_csv(time_table, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "date_label": date_label,
        "row_counts": {
            "input": int(len(df_raw)),
            "valid_for_plotting": int(len(df)),
        },
        "columns": {
            "timestamp": args.timestamp_column,
            "vtec": args.vtec_column,
        },
        "map": {
            "selected_timestamp": selected_time.isoformat(),
            "rows": map_rows,
            "backend": map_backend,
            "extent": args.extent,
            "outputs": map_paths,
        },
        "outputs": {
            "map": map_paths,
            "elevation_distribution": elevation_paths,
            "station_distribution": station_paths,
            "time_evolution": time_paths,
            "elevation_table": str(elevation_table),
            "station_table": str(station_table),
            "time_table": str(time_table),
        },
        "settings": {
            "top_stations": args.top_stations,
            "min_map_points": args.min_map_points,
        },
    }
    metadata_path = Path(args.metadata_output) if args.metadata_output else (
        table_dir / f"vtec_slm_diagnostics_{date_label}.metadata.json"
    )
    metadata["outputs"]["metadata"] = str(metadata_path)
    save_json(metadata, metadata_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input parquet/CSV.")
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--metadata-output")
    parser.add_argument("--vtec-column", default="vtec_slm")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--date-label")
    parser.add_argument(
        "--map-time",
        help=(
            "Optional map timestamp. If omitted, uses the timestamp with peak "
            "median VTEC among data-rich epochs."
        ),
    )
    parser.add_argument("--min-map-points", type=int, default=100)
    parser.add_argument("--top-stations", type=int, default=35)
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        default=[60.0, 160.0, -20.0, 80.0],
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = build_vtec_diagnostics(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
