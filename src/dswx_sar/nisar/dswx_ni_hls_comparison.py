#!/usr/bin/env python3
"""Run DSWx-NI and compare its large-area WTR product with DSWx-HLS.

This module is intended for the OPERA DSWX-SAR ``dswx-ni-calval-scale``
branch.  It can be used in two ways:

1. ``run``: execute the normal DSWx-NI workflow and then compare the
   large-area NI WTR raster with every DSWx-HLS acquisition epoch inside a
   configurable time window.
2. ``aggregate``: combine the immutable per-pair JSON records written by many
   independent jobs.  No worker appends to a shared CSV, so concurrent AWS
   Batch jobs cannot corrupt the time-series summary.

The DSWx-NI intermediate raster is read from ``scratch_path``.  All persistent
comparison products are written below ``sas_output_path/hls_comparison`` unless
``--comparison-output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.warp import transform_bounds


LOGGER = logging.getLogger("dswx_sar.hls_comparison")

HLS_SHORT_NAME = "OPERA_L3_DSWX-HLS_V1"
HLS_LAYER_SUFFIX = "_B01_WTR.tif"
TIME_RE = re.compile(r"(\d{8}T\d{6}Z?)")
HLS_TILE_RE = re.compile(r"_T(\d{2}[C-HJ-NP-X][A-HJ-NP-Z]{2})_")
HLS_SENSOR_RE = re.compile(r"_(L8|L9|S2A|S2B|S2C)_")

CONFUSION_NODATA = 255
CONFUSION_TN = 1
CONFUSION_FP = 2
CONFUSION_FN = 3
CONFUSION_TP = 4
SCHEMA_VERSION = "3.0"


@dataclass(frozen=True)
class ComparisonOptions:
    """Options controlling discovery, comparison, and output."""

    mode: str = "online"
    hls_input_dir: str | None = None
    search_margin_days: float = 6.0
    search_box_size_km: float = 50.0
    epoch_gap_minutes: float = 30.0
    min_comparison_pixels: int = 10_000
    min_usable_pixel_fraction: float = 0.05
    precheck_max_dimension: int = 2_048
    png_max_dimension: int = 1_800
    ni_water_values: tuple[int, ...] = (1, 8)
    ni_valid_values: tuple[int, ...] = (0, 1, 8)
    hls_water_values: tuple[int, ...] = (1, 2)
    hls_valid_values: tuple[int, ...] = (0, 1, 2)
    ni_class_values: tuple[int, ...] = (0, 1, 3, 8)
    hls_class_values: tuple[int, ...] = (0, 1, 2)
    boundary_comparison: bool = True
    boundary_width_pixels: int = 10
    boundary_min_comparison_pixels: int = 1_000
    save_hls_assets: bool = False
    retry_failed_downloads: bool = False
    overwrite: bool = False


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _format_time(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        stream.write("\n")
    os.replace(temporary, path)


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _format_time(value: datetime) -> str:
    value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact_time(value: datetime) -> str:
    value = value.astimezone(timezone.utc)
    return value.strftime("%Y%m%dT%H%M%SZ")


def _parse_compact_time(value: str) -> datetime:
    value = value.rstrip("Z")
    return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(
        tzinfo=timezone.utc
    )


def _basename(path_or_url: str) -> str:
    parsed = urlparse(str(path_or_url))
    return Path(parsed.path).name


def _extract_ni_time_range(input_paths: Sequence[str]) -> tuple[datetime, datetime]:
    starts: list[datetime] = []
    stops: list[datetime] = []

    for input_path in input_paths:
        timestamps = TIME_RE.findall(_basename(input_path))
        if len(timestamps) < 2:
            raise ValueError(
                "Cannot extract NISAR start/stop time from input: "
                f"{input_path}"
            )
        starts.append(_parse_compact_time(timestamps[0]))
        stops.append(_parse_compact_time(timestamps[1]))

    if not starts:
        raise ValueError("The runconfig contains no NISAR input files")

    return min(starts), max(stops)


def _make_ni_run_id(
    input_paths: Sequence[str],
    start_time: datetime,
    stop_time: datetime,
) -> str:
    input_names = sorted(_basename(item) for item in input_paths)
    digest = hashlib.sha256("\n".join(input_names).encode("utf-8")).hexdigest()[:10]
    return f"NI_{_compact_time(start_time)}_{_compact_time(stop_time)}_{digest}"


def _parse_hls_filename(filename: str) -> dict[str, Any]:
    tile_match = HLS_TILE_RE.search(filename)
    sensor_match = HLS_SENSOR_RE.search(filename)
    timestamps = TIME_RE.findall(filename)

    if tile_match is None or sensor_match is None or len(timestamps) < 1:
        raise ValueError(f"Unrecognized DSWx-HLS filename: {filename}")

    generation_time = (
        _parse_compact_time(timestamps[1]) if len(timestamps) >= 2 else None
    )
    return {
        "filename": filename,
        "tile": tile_match.group(1),
        "sensor": sensor_match.group(1),
        "acquisition_time": _parse_compact_time(timestamps[0]),
        "generation_time": generation_time,
    }


def _create_valid_search_boxes(
    raster_path: Path,
    box_size_km: float,
    padding_pixels: int = 2,
) -> list[dict[str, Any]]:
    """Cover actual valid NI pixels with small WGS84 search rectangles."""

    boxes: list[dict[str, Any]] = []
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS: {raster_path}")

        box_width = max(1, math.ceil(box_size_km * 1000 / abs(src.transform.a)))
        box_height = max(1, math.ceil(box_size_km * 1000 / abs(src.transform.e)))

        for row_off in range(0, src.height, box_height):
            for col_off in range(0, src.width, box_width):
                width = min(box_width, src.width - col_off)
                height = min(box_height, src.height - row_off)
                window = Window(col_off, row_off, width, height)
                valid = src.read_masks(1, window=window) > 0
                if not np.any(valid):
                    continue

                local_rows, local_cols = np.nonzero(valid)
                min_row = max(0, row_off + int(local_rows.min()) - padding_pixels)
                max_row = min(
                    src.height - 1,
                    row_off + int(local_rows.max()) + padding_pixels,
                )
                min_col = max(0, col_off + int(local_cols.min()) - padding_pixels)
                max_col = min(
                    src.width - 1,
                    col_off + int(local_cols.max()) + padding_pixels,
                )
                valid_window = Window(
                    min_col,
                    min_row,
                    max_col - min_col + 1,
                    max_row - min_row + 1,
                )
                native_bounds = window_bounds(valid_window, src.transform)
                bbox = transform_bounds(
                    src.crs,
                    "EPSG:4326",
                    *native_bounds,
                    densify_pts=21,
                )
                boxes.append(
                    {
                        "box_id": len(boxes),
                        "bbox": tuple(float(item) for item in bbox),
                        "valid_pixels": int(np.count_nonzero(valid)),
                    }
                )

    if not boxes:
        raise ValueError(f"No valid pixels found in {raster_path}")
    return boxes


def _bounds_intersect(
    first: Sequence[float], second: Sequence[float]
) -> bool:
    west_1, south_1, east_1, north_1 = first
    west_2, south_2, east_2, north_2 = second
    return not (
        east_1 <= west_2
        or east_2 <= west_1
        or north_1 <= south_2
        or north_2 <= south_1
    )


def _raster_bounds_wgs84(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        return transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21,
        )


import random
import time

import earthaccess
import requests


def _login_earthdata(
    max_attempts=5,
    initial_wait_seconds=5,
):
    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            auth = earthaccess.login(
                strategy="environment",
            )

            if not auth.authenticated:
                raise RuntimeError(
                    "Earthdata authentication failed"
                )

            return auth

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as error:
            if attempt == max_attempts:
                raise RuntimeError(
                    "Earthdata login failed after "
                    f"{max_attempts} attempts"
                ) from error

            wait_seconds = min(
                60,
                initial_wait_seconds
                * (2 ** (attempt - 1)),
            )

            # Prevent simultaneous Batch jobs from
            # retrying at exactly the same time.
            wait_seconds += random.uniform(
                0,
                3,
            )

            LOGGER.warning(
                "Earthdata login attempt %d/%d "
                "failed: %s. Retrying in %.1f sec.",
                attempt,
                max_attempts,
                error,
                wait_seconds,
            )

            time.sleep(wait_seconds)

def _discover_online_hls_assets(
    boxes: Sequence[dict[str, Any]],
    start_time: datetime,
    stop_time: datetime,
) -> list[dict[str, Any]]:
    try:
        import earthaccess
    except ImportError as error:
        raise RuntimeError(
            "Online mode requires earthaccess. Install it in the container "
            "image, not at Batch runtime."
        ) from error

    # Batch is non-interactive.  earthaccess accepts EARTHDATA_TOKEN or the
    # EARTHDATA_USERNAME/EARTHDATA_PASSWORD pair with this strategy.
    _login_earthdata()

    assets: dict[str, dict[str, Any]] = {}
    temporal = (_format_time(start_time), _format_time(stop_time))

    for index, box in enumerate(boxes, start=1):
        LOGGER.info("Searching HLS box %d/%d", index, len(boxes))
        results = earthaccess.search_data(
            short_name=HLS_SHORT_NAME,
            temporal=temporal,
            bounding_box=box["bbox"],
        )
        for granule in results:
            for url in granule.data_links():
                filename = _basename(url)
                if not filename.endswith(HLS_LAYER_SUFFIX):
                    continue
                try:
                    parsed = _parse_hls_filename(filename)
                except ValueError:
                    LOGGER.warning("Skipping unrecognized HLS asset: %s", filename)
                    continue
                record = assets.setdefault(
                    filename,
                    {
                        **parsed,
                        "url": url,
                        "query_box_ids": set(),
                    },
                )
                record["query_box_ids"].add(box["box_id"])

    output: list[dict[str, Any]] = []
    for record in assets.values():
        record["query_box_ids"] = sorted(record["query_box_ids"])
        output.append(record)
    return sorted(output, key=lambda item: (item["acquisition_time"], item["tile"]))


def _discover_offline_hls_assets(
    input_directory: Path,
    boxes: Sequence[dict[str, Any]],
    start_time: datetime,
    stop_time: datetime,
) -> list[dict[str, Any]]:
    if not input_directory.is_dir():
        raise NotADirectoryError(input_directory)

    assets: list[dict[str, Any]] = []
    for path in sorted(input_directory.rglob(f"*{HLS_LAYER_SUFFIX}")):
        try:
            parsed = _parse_hls_filename(path.name)
        except ValueError:
            continue
        acquisition_time = parsed["acquisition_time"]
        if acquisition_time < start_time or acquisition_time > stop_time:
            continue
        reference_bounds = _raster_bounds_wgs84(path)
        hit_box_ids = [
            box["box_id"]
            for box in boxes
            if _bounds_intersect(reference_bounds, box["bbox"])
        ]
        if not hit_box_ids:
            continue
        assets.append(
            {
                **parsed,
                "path": str(path),
                "query_box_ids": hit_box_ids,
            }
        )
    return assets


def _download_hls_assets(
    assets: list[dict[str, Any]],
    cache_directory: Path,
    retry_failed_downloads: bool = False,
) -> list[dict[str, Any]]:
    try:
        import earthaccess
    except ImportError as error:
        raise RuntimeError("Online mode requires earthaccess") from error

    cache_directory.mkdir(parents=True, exist_ok=True)
    failure_cache_path = cache_directory / "download_failures.json"
    failure_cache: dict[str, dict[str, Any]] = {}
    if failure_cache_path.is_file():
        try:
            with failure_cache_path.open("r", encoding="utf-8") as stream:
                failure_cache = json.load(stream)
        except (json.JSONDecodeError, OSError):
            LOGGER.warning(
                "Ignoring invalid download failure cache: %s", failure_cache_path
            )

    missing = [
        asset
        for asset in assets
        if not (cache_directory / asset["filename"]).is_file()
    ]
    if missing:
        LOGGER.info("Downloading %d new HLS WTR assets", len(missing))

    for index, asset in enumerate(missing, start=1):
        filename = asset["filename"]
        cached_failure = failure_cache.get(filename, {})
        if (
            cached_failure.get("permanent") is True
            and not retry_failed_downloads
        ):
            LOGGER.warning(
                "Skipping cached permanent HLS download failure %d/%d: %s",
                index,
                len(missing),
                filename,
            )
            continue

        LOGGER.info("Downloading HLS asset %d/%d: %s", index, len(missing), filename)
        try:
            earthaccess.download(
                [asset["url"]],
                local_path=str(cache_directory),
                provider="POCLOUD",
            )
        except Exception as error:  # earthaccess exception types vary by version
            error_text = str(error)
            permanent = bool(
                re.search(
                    r"(?:status code|HTTP)\s*[:=]?\s*404\b",
                    error_text,
                    re.I,
                )
            )
            failure_cache[filename] = {
                "url": asset["url"],
                "error": error_text,
                "permanent": permanent,
                "updated_utc": _format_time(datetime.now(timezone.utc)),
            }
            _write_json_atomic(failure_cache_path, failure_cache)
            LOGGER.warning("Skipping failed HLS download %s: %s", filename, error)
            continue

        path = cache_directory / filename
        if not path.is_file():
            failure_cache[filename] = {
                "url": asset["url"],
                "error": "earthaccess.download returned without creating the file",
                "permanent": False,
                "updated_utc": _format_time(datetime.now(timezone.utc)),
            }
            _write_json_atomic(failure_cache_path, failure_cache)
            LOGGER.warning("HLS download did not create %s", path)
            continue

        if filename in failure_cache:
            del failure_cache[filename]
            _write_json_atomic(failure_cache_path, failure_cache)

    output: list[dict[str, Any]] = []
    for asset in assets:
        path = cache_directory / asset["filename"]
        if not path.is_file():
            LOGGER.error("HLS download did not create %s", path)
            continue
        output.append({**asset, "path": str(path)})
    return output


def _group_hls_epochs(
    assets: Sequence[dict[str, Any]], gap_minutes: float
) -> list[dict[str, Any]]:
    """Group neighboring MGRS assets from the same sensor acquisition."""

    epochs: list[dict[str, Any]] = []
    for sensor in sorted({asset["sensor"] for asset in assets}):
        sensor_assets = sorted(
            (asset for asset in assets if asset["sensor"] == sensor),
            key=lambda item: item["acquisition_time"],
        )
        current: list[dict[str, Any]] = []
        previous_time: datetime | None = None
        for asset in sensor_assets:
            acquisition_time = asset["acquisition_time"]
            if (
                current
                and previous_time is not None
                and acquisition_time - previous_time
                > timedelta(minutes=gap_minutes)
            ):
                epochs.append(_make_epoch_record(sensor, current))
                current = []
            current.append(asset)
            previous_time = acquisition_time
        if current:
            epochs.append(_make_epoch_record(sensor, current))

    return sorted(epochs, key=lambda item: item["midpoint"])


def _make_epoch_record(
    sensor: str, assets: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    start_time = min(asset["acquisition_time"] for asset in assets)
    stop_time = max(asset["acquisition_time"] for asset in assets)
    midpoint = start_time + (stop_time - start_time) / 2
    epoch_id = f"{sensor}_{_compact_time(start_time)}"
    return {
        "epoch_id": epoch_id,
        "sensor": sensor,
        "start_time": start_time,
        "stop_time": stop_time,
        "midpoint": midpoint,
        "tiles": sorted({asset["tile"] for asset in assets}),
        "assets": list(assets),
    }


def _priority(array: np.ndarray, valid_values: Sequence[int]) -> np.ndarray:
    priority = np.zeros(array.shape, dtype=np.uint8)
    priority[array != 255] = 1
    priority[np.isin(array, valid_values)] = 3
    return priority


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    total = tp + tn + fp + fn
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "comparison_pixels": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_ratio(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "iou_water": _safe_ratio(tp, tp + fp + fn),
    }


def _precheck_epoch_coverage(
    ni_path: Path,
    reference_paths: Sequence[Path],
    options: ComparisonOptions,
) -> dict[str, Any]:
    """Estimate usable overlap on a small overview before full-resolution I/O."""

    with rasterio.open(ni_path) as ni_ds, ExitStack() as stack:
        scale = max(
            1.0,
            max(ni_ds.width, ni_ds.height) / options.precheck_max_dimension,
        )
        width = max(1, int(math.ceil(ni_ds.width / scale)))
        height = max(1, int(math.ceil(ni_ds.height / scale)))
        shape = (height, width)

        ni = ni_ds.read(1, out_shape=shape, resampling=Resampling.nearest)
        sources = [
            stack.enter_context(rasterio.open(path)) for path in reference_paths
        ]
        vrts = [
            stack.enter_context(
                WarpedVRT(
                    source,
                    crs=ni_ds.crs,
                    transform=ni_ds.transform,
                    width=ni_ds.width,
                    height=ni_ds.height,
                    resampling=Resampling.nearest,
                    src_nodata=(
                        source.nodata if source.nodata is not None else 255
                    ),
                    nodata=255,
                )
            )
            for source in sources
        ]

        hls = np.full(shape, 255, dtype=np.uint8)
        for vrt in vrts:
            candidate = vrt.read(1, out_shape=shape, resampling=Resampling.nearest)
            update = _priority(
                candidate, options.hls_valid_values
            ) > _priority(hls, options.hls_valid_values)
            hls[update] = candidate[update]

        ni_evaluable = np.isin(ni, options.ni_class_values)
        hls_evaluable = np.isin(hls, options.hls_class_values)
        common = ni_evaluable & hls_evaluable
        overview_ni_pixels = int(np.count_nonzero(ni_evaluable))
        overview_common_pixels = int(np.count_nonzero(common))
        full_scale_factor = (ni_ds.width * ni_ds.height) / (width * height)

    return {
        "overview_width": width,
        "overview_height": height,
        "overview_ni_evaluable_pixels": overview_ni_pixels,
        "overview_comparison_pixels": overview_common_pixels,
        "estimated_comparison_pixels": int(
            round(overview_common_pixels * full_scale_factor)
        ),
        "estimated_usable_pixel_fraction": (
            overview_common_pixels / overview_ni_pixels
            if overview_ni_pixels
            else 0.0
        ),
    }


def _write_class_confusion_outputs(
    pair_directory: Path,
    matrix: np.ndarray,
    ni_values: Sequence[int],
    hls_values: Sequence[int],
) -> tuple[Path, Path]:
    csv_path = pair_directory / "class_confusion_matrix.csv"
    json_path = pair_directory / "class_confusion_matrix.json"
    fieldnames = ["ni_class", *[f"hls_{value}" for value in hls_values], "row_total"]
    rows = []
    for row_index, ni_value in enumerate(ni_values):
        row = {"ni_class": int(ni_value)}
        for column_index, hls_value in enumerate(hls_values):
            row[f"hls_{hls_value}"] = int(matrix[row_index, column_index])
        row["row_total"] = int(matrix[row_index].sum())
        rows.append(row)
    _write_csv_atomic(csv_path, fieldnames, rows)
    _write_json_atomic(
        json_path,
        {
            "ni_classes": list(ni_values),
            "hls_classes": list(hls_values),
            "matrix": matrix.astype(np.uint64).tolist(),
            "total": int(matrix.sum()),
            "raster_codes": [
                {
                    "value": ni_index * len(hls_values) + hls_index + 1,
                    "ni_class": int(ni_value),
                    "hls_class": int(hls_value),
                    "count": int(matrix[ni_index, hls_index]),
                }
                for ni_index, ni_value in enumerate(ni_values)
                for hls_index, hls_value in enumerate(hls_values)
            ],
            "raster_nodata": CONFUSION_NODATA,
        },
    )
    return csv_path, json_path


def _write_comparison_png(
    ni_path: Path,
    mosaic_path: Path,
    confusion_path: Path,
    output_path: Path,
    options: ComparisonOptions,
    title: str,
    confusion_panel_title: str = "Binary confusion",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    with rasterio.open(ni_path) as ni_ds:
        scale = max(
            1.0,
            max(ni_ds.width, ni_ds.height) / options.png_max_dimension,
        )
        shape = (
            max(1, int(math.ceil(ni_ds.height / scale))),
            max(1, int(math.ceil(ni_ds.width / scale))),
        )
        ni = ni_ds.read(1, out_shape=shape, resampling=Resampling.nearest)
    with rasterio.open(mosaic_path) as hls_ds:
        hls = hls_ds.read(1, out_shape=shape, resampling=Resampling.nearest)
    with rasterio.open(confusion_path) as confusion_ds:
        confusion = confusion_ds.read(
            1, out_shape=shape, resampling=Resampling.nearest
        )

    ni_binary = np.ma.masked_all(shape, dtype=np.uint8)
    ni_valid = np.isin(ni, options.ni_valid_values)
    ni_binary[ni_valid] = np.isin(ni[ni_valid], options.ni_water_values)
    hls_binary = np.ma.masked_all(shape, dtype=np.uint8)
    hls_valid = np.isin(hls, options.hls_valid_values)
    hls_binary[hls_valid] = np.isin(hls[hls_valid], options.hls_water_values)

    binary_cmap = ListedColormap(["#8c8c8c", "#2455d6"])
    confusion_cmap = ListedColormap(
        ["#ffffff", "#8c8c8c", "#e53935", "#90ee90", "#2455d6"]
    )
    confusion_display = np.where(confusion == CONFUSION_NODATA, 0, confusion)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    axes[0].imshow(ni_binary, cmap=binary_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("DSWx-NI binary water")
    axes[1].imshow(
        hls_binary,
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title("DSWx-HLS binary water")
    axes[2].imshow(
        confusion_display,
        cmap=confusion_cmap,
        vmin=0,
        vmax=4,
        interpolation="nearest",
    )
    axes[2].set_title(confusion_panel_title)
    for axis in axes:
        axis.axis("off")
    axes[2].legend(
        handles=[
            Patch(color="#2455d6", label="TP"),
            Patch(color="#8c8c8c", label="TN"),
            Patch(color="#e53935", label="FP"),
            Patch(color="#90ee90", label="FN"),
        ],
        loc="lower right",
        framealpha=0.9,
    )
    fig.suptitle(title)
    temporary = output_path.with_name(f".{output_path.name}.tmp.png")
    fig.savefig(temporary, dpi=150, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output_path)


def _dilate_boolean(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dilate a Boolean mask with an 8-connected, one-pixel neighborhood."""

    result = mask.astype(bool, copy=True)
    for _ in range(iterations):
        source = result
        expanded = source.copy()
        expanded[1:, :] |= source[:-1, :]
        expanded[:-1, :] |= source[1:, :]
        expanded[:, 1:] |= source[:, :-1]
        expanded[:, :-1] |= source[:, 1:]
        expanded[1:, 1:] |= source[:-1, :-1]
        expanded[1:, :-1] |= source[:-1, 1:]
        expanded[:-1, 1:] |= source[1:, :-1]
        expanded[:-1, :-1] |= source[1:, 1:]
        result = expanded
    return result


def _expanded_window(
    window: Window,
    halo: int,
    raster_width: int,
    raster_height: int,
) -> tuple[Window, tuple[slice, slice]]:
    """Return a clipped halo window and slices locating the original window."""

    col_start = max(0, int(window.col_off) - halo)
    row_start = max(0, int(window.row_off) - halo)
    col_stop = min(
        raster_width,
        int(window.col_off + window.width) + halo,
    )
    row_stop = min(
        raster_height,
        int(window.row_off + window.height) + halo,
    )
    expanded = Window(
        col_start,
        row_start,
        col_stop - col_start,
        row_stop - row_start,
    )
    row_offset = int(window.row_off) - row_start
    col_offset = int(window.col_off) - col_start
    core = (
        slice(row_offset, row_offset + int(window.height)),
        slice(col_offset, col_offset + int(window.width)),
    )
    return expanded, core


def _write_boundary_comparison_outputs(
    ni_path: Path,
    mosaic_path: Path,
    pair_directory: Path,
    options: ComparisonOptions,
    epoch_id: str,
    full_class_comparison_pixels: int,
) -> dict[str, Any]:
    """Evaluate only a band on both sides of valid HLS water boundaries.

    The band is derived only from valid HLS water and valid HLS non-water
    classes. HLS NoData/cloud pixels are never treated as shoreline seeds.
    Water bodies narrower than twice the requested band width naturally retain
    all of their water pixels, while deep interiors of large water bodies are
    excluded.
    """

    boundary_directory = pair_directory / "boundary"
    boundary_directory.mkdir(parents=True, exist_ok=True)
    mask_path = boundary_directory / "evaluation_mask.tif"
    confusion_path = boundary_directory / "confusion.tif"
    class_confusion_path = boundary_directory / "class_confusion.tif"
    png_path = boundary_directory / "comparison.png"
    metrics_path = boundary_directory / "metrics.json"
    class_csv_path = boundary_directory / "class_confusion_matrix.csv"
    class_json_path = boundary_directory / "class_confusion_matrix.json"

    temporary_mask = mask_path.with_name(f".{mask_path.name}.tmp")
    temporary_confusion = confusion_path.with_name(
        f".{confusion_path.name}.tmp"
    )
    temporary_class_confusion = class_confusion_path.with_name(
        f".{class_confusion_path.name}.tmp"
    )

    tp = tn = fp = fn = 0
    boundary_zone_pixels = 0
    boundary_class_pixels = 0
    class_matrix = np.zeros(
        (len(options.ni_class_values), len(options.hls_class_values)),
        dtype=np.uint64,
    )
    number_of_hls_classes = len(options.hls_class_values)

    try:
        with rasterio.open(ni_path) as ni_ds, rasterio.open(mosaic_path) as hls_ds:
            profile = ni_ds.profile.copy()
            profile.update(
                driver="GTiff",
                dtype="uint8",
                count=1,
                nodata=CONFUSION_NODATA,
                compress="DEFLATE",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )
            with rasterio.open(
                temporary_mask, "w", **profile
            ) as mask_ds, rasterio.open(
                temporary_confusion, "w", **profile
            ) as confusion_ds, rasterio.open(
                temporary_class_confusion, "w", **profile
            ) as class_ds:
                mask_ds.set_band_description(1, "HLS shoreline evaluation mask")
                confusion_ds.set_band_description(1, "HLS shoreline confusion")
                class_ds.set_band_description(
                    1, "HLS shoreline DSWx-NI x DSWx-HLS class combination"
                )
                for _, window in ni_ds.block_windows(1):
                    expanded, core = _expanded_window(
                        window,
                        options.boundary_width_pixels,
                        ni_ds.width,
                        ni_ds.height,
                    )
                    hls_expanded = hls_ds.read(1, window=expanded)
                    hls_valid_expanded = np.isin(
                        hls_expanded, options.hls_valid_values
                    )
                    hls_water_expanded = (
                        hls_valid_expanded
                        & np.isin(hls_expanded, options.hls_water_values)
                    )
                    hls_land_expanded = hls_valid_expanded & ~hls_water_expanded
                    near_water = _dilate_boolean(
                        hls_water_expanded, options.boundary_width_pixels
                    )
                    near_land = _dilate_boolean(
                        hls_land_expanded, options.boundary_width_pixels
                    )
                    boundary_expanded = (
                        (hls_water_expanded & near_land)
                        | (hls_land_expanded & near_water)
                    )
                    boundary = boundary_expanded[core]
                    hls = hls_expanded[core]
                    ni = ni_ds.read(1, window=window)

                    ni_binary_valid = np.isin(ni, options.ni_valid_values)
                    hls_binary_valid = np.isin(hls, options.hls_valid_values)
                    binary_common = boundary & ni_binary_valid & hls_binary_valid
                    ni_water = np.isin(ni, options.ni_water_values)
                    hls_water = np.isin(hls, options.hls_water_values)

                    block_tn = binary_common & ~ni_water & ~hls_water
                    block_fp = binary_common & ni_water & ~hls_water
                    block_fn = binary_common & ~ni_water & hls_water
                    block_tp = binary_common & ni_water & hls_water
                    confusion = np.full(
                        ni.shape, CONFUSION_NODATA, dtype=np.uint8
                    )
                    confusion[block_tn] = CONFUSION_TN
                    confusion[block_fp] = CONFUSION_FP
                    confusion[block_fn] = CONFUSION_FN
                    confusion[block_tp] = CONFUSION_TP
                    confusion_ds.write(confusion, 1, window=window)

                    tn += int(np.count_nonzero(block_tn))
                    fp += int(np.count_nonzero(block_fp))
                    fn += int(np.count_nonzero(block_fn))
                    tp += int(np.count_nonzero(block_tp))
                    boundary_zone_pixels += int(np.count_nonzero(boundary))

                    class_common = (
                        boundary
                        & np.isin(ni, options.ni_class_values)
                        & np.isin(hls, options.hls_class_values)
                    )
                    evaluation_mask = np.full(
                        ni.shape, CONFUSION_NODATA, dtype=np.uint8
                    )
                    evaluation_mask[class_common] = 1
                    mask_ds.write(evaluation_mask, 1, window=window)

                    class_confusion = np.full(
                        ni.shape, CONFUSION_NODATA, dtype=np.uint8
                    )
                    for ni_index, ni_value in enumerate(options.ni_class_values):
                        for hls_index, hls_value in enumerate(
                            options.hls_class_values
                        ):
                            class_pair = (
                                class_common
                                & (ni == ni_value)
                                & (hls == hls_value)
                            )
                            count = int(np.count_nonzero(class_pair))
                            class_matrix[ni_index, hls_index] += count
                            class_confusion[class_pair] = (
                                ni_index * number_of_hls_classes
                                + hls_index
                                + 1
                            )
                    class_ds.write(class_confusion, 1, window=window)

        boundary_class_pixels = int(class_matrix.sum())
        metrics = _calculate_metrics(tp, tn, fp, fn)
        status = (
            "completed"
            if metrics["comparison_pixels"]
            >= options.boundary_min_comparison_pixels
            else "insufficient_boundary_pixels"
        )
        os.replace(temporary_mask, mask_path)
        os.replace(temporary_confusion, confusion_path)
        os.replace(temporary_class_confusion, class_confusion_path)
        _write_class_confusion_outputs(
            boundary_directory,
            class_matrix,
            options.ni_class_values,
            options.hls_class_values,
        )
        accuracy_text = (
            f"{metrics['accuracy']:.3f}"
            if metrics["accuracy"] is not None
            else "NA"
        )
        _write_comparison_png(
            ni_path,
            mosaic_path,
            confusion_path,
            png_path,
            options,
            (
                f"{epoch_id} | HLS boundary +/-"
                f"{options.boundary_width_pixels}px | OA={accuracy_text}"
            ),
            confusion_panel_title="Boundary-only confusion",
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "boundary_width_pixels": options.boundary_width_pixels,
            "minimum_boundary_comparison_pixels": (
                options.boundary_min_comparison_pixels
            ),
            "boundary_zone_pixels": boundary_zone_pixels,
            "boundary_class_comparison_pixels": boundary_class_pixels,
            "boundary_fraction_of_full_class_comparison": _safe_ratio(
                boundary_class_pixels, full_class_comparison_pixels
            ),
            **metrics,
            "ni_water_values": list(options.ni_water_values),
            "ni_valid_values": list(options.ni_valid_values),
            "hls_water_values": list(options.hls_water_values),
            "hls_valid_values": list(options.hls_valid_values),
            "ni_class_values": list(options.ni_class_values),
            "hls_class_values": list(options.hls_class_values),
            "evaluation_mask": mask_path.name,
            "confusion_raster": confusion_path.name,
            "class_confusion_raster": class_confusion_path.name,
            "comparison_png": png_path.name,
            "class_confusion_csv": class_csv_path.name,
            "class_confusion_json": class_json_path.name,
            "class_confusion_matrix": class_matrix.tolist(),
        }
        _write_json_atomic(metrics_path, result)
        return result
    finally:
        temporary_mask.unlink(missing_ok=True)
        temporary_confusion.unlink(missing_ok=True)
        temporary_class_confusion.unlink(missing_ok=True)


def _compare_epoch(
    ni_path: Path,
    epoch: dict[str, Any],
    pair_directory: Path,
    options: ComparisonOptions,
    ni_midpoint: datetime,
) -> dict[str, Any]:
    pair_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = pair_directory / "pair_metrics.json"
    mosaic_path = pair_directory / "hls_wtr_mosaic.tif"
    confusion_path = pair_directory / "confusion.tif"
    class_confusion_path = pair_directory / "class_confusion.tif"
    png_path = pair_directory / "comparison.png"
    class_csv_path = pair_directory / "class_confusion_matrix.csv"
    class_json_path = pair_directory / "class_confusion_matrix.json"
    boundary_directory = pair_directory / "boundary"
    boundary_expected_paths = (
        boundary_directory / "evaluation_mask.tif",
        boundary_directory / "confusion.tif",
        boundary_directory / "class_confusion.tif",
        boundary_directory / "comparison.png",
        boundary_directory / "metrics.json",
        boundary_directory / "class_confusion_matrix.csv",
        boundary_directory / "class_confusion_matrix.json",
    )

    if options.overwrite:
        for path in (
            metrics_path,
            mosaic_path,
            confusion_path,
            class_confusion_path,
            png_path,
            class_csv_path,
            class_json_path,
            *boundary_expected_paths,
        ):
            path.unlink(missing_ok=True)

    if metrics_path.is_file() and not options.overwrite:
        with metrics_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        expected_outputs_exist = existing.get("schema_version") == SCHEMA_VERSION and (
            str(existing.get("status", "")).startswith("skipped_low_coverage")
            or all(
                path.is_file()
                for path in (
                    mosaic_path,
                    confusion_path,
                    class_confusion_path,
                    png_path,
                    class_csv_path,
                    class_json_path,
                )
            )
        )
        if (
            expected_outputs_exist
            and options.boundary_comparison
            and existing.get("status") == "completed"
        ):
            expected_outputs_exist = all(
                path.is_file() for path in boundary_expected_paths
            )
        if expected_outputs_exist:
            LOGGER.info("Reusing completed comparison %s", epoch["epoch_id"])
            return existing

    reference_paths = sorted(Path(asset["path"]) for asset in epoch["assets"])
    delta_hours = (epoch["midpoint"] - ni_midpoint).total_seconds() / 3600
    precheck = _precheck_epoch_coverage(ni_path, reference_paths, options)
    precheck_passed = (
        precheck["estimated_comparison_pixels"] >= options.min_comparison_pixels
        and precheck["estimated_usable_pixel_fraction"]
        >= options.min_usable_pixel_fraction
    )
    if not precheck_passed:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped_low_coverage_precheck",
            "pair_id": epoch["epoch_id"],
            "ni_path": str(ni_path),
            "ni_datetime": _format_time(ni_midpoint),
            "hls_epoch_id": epoch["epoch_id"],
            "hls_sensor": epoch["sensor"],
            "hls_start_datetime": _format_time(epoch["start_time"]),
            "hls_stop_datetime": _format_time(epoch["stop_time"]),
            "hls_midpoint_datetime": _format_time(epoch["midpoint"]),
            "delta_hours": delta_hours,
            "abs_delta_days": abs(delta_hours) / 24,
            "hls_tiles": epoch["tiles"],
            "hls_filenames": [path.name for path in reference_paths],
            **precheck,
            "minimum_comparison_pixels": options.min_comparison_pixels,
            "minimum_usable_pixel_fraction": options.min_usable_pixel_fraction,
            "ni_water_values": list(options.ni_water_values),
            "ni_valid_values": list(options.ni_valid_values),
            "hls_water_values": list(options.hls_water_values),
            "hls_valid_values": list(options.hls_valid_values),
            "ni_class_values": list(options.ni_class_values),
            "hls_class_values": list(options.hls_class_values),
            "hls_mosaic": None,
            "confusion_raster": None,
            "class_confusion_raster": None,
            "comparison_png": None,
            "class_confusion_csv": None,
            "class_confusion_json": None,
            "class_confusion_matrix": None,
            "boundary_comparison": None,
        }
        _write_json_atomic(metrics_path, result)
        LOGGER.info(
            "Skipping %s after overview precheck: estimated pixels=%d, fraction=%.4f",
            epoch["epoch_id"],
            precheck["estimated_comparison_pixels"],
            precheck["estimated_usable_pixel_fraction"],
        )
        return result

    temporary_mosaic = mosaic_path.with_name(f".{mosaic_path.name}.tmp")
    temporary_confusion = confusion_path.with_name(f".{confusion_path.name}.tmp")
    temporary_class_confusion = class_confusion_path.with_name(
        f".{class_confusion_path.name}.tmp"
    )

    tp = tn = fp = fn = 0
    ni_evaluable_pixels = 0
    ni_usable_pixels = 0
    ni_footprint_pixels = 0
    reference_footprint_pixels = 0
    common_footprint_pixels = 0
    class_matrix = np.zeros(
        (len(options.ni_class_values), len(options.hls_class_values)),
        dtype=np.uint64,
    )
    number_of_hls_classes = len(options.hls_class_values)

    try:
        with rasterio.open(ni_path) as ni_ds, ExitStack() as stack:
            source_datasets = [
                stack.enter_context(rasterio.open(path)) for path in reference_paths
            ]
            vrts = [
                stack.enter_context(
                    WarpedVRT(
                        source,
                        crs=ni_ds.crs,
                        transform=ni_ds.transform,
                        width=ni_ds.width,
                        height=ni_ds.height,
                        resampling=Resampling.nearest,
                        src_nodata=source.nodata if source.nodata is not None else 255,
                        nodata=255,
                    )
                )
                for source in source_datasets
            ]

            profile = ni_ds.profile.copy()
            profile.update(
                driver="GTiff",
                dtype="uint8",
                count=1,
                nodata=255,
                compress="DEFLATE",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )

            with rasterio.open(
                temporary_mosaic, "w", **profile
            ) as mosaic_ds, rasterio.open(
                temporary_confusion, "w", **profile
            ) as confusion_ds, rasterio.open(
                temporary_class_confusion, "w", **profile
            ) as class_confusion_ds:
                class_confusion_ds.set_band_description(
                    1, "DSWx-NI x DSWx-HLS class combination"
                )
                class_confusion_ds.update_tags(
                    **{
                        (
                            "CLASS_CODE_"
                            f"{ni_index * number_of_hls_classes + hls_index + 1}"
                        ): f"NI={ni_value},HLS={hls_value}"
                        for ni_index, ni_value in enumerate(options.ni_class_values)
                        for hls_index, hls_value in enumerate(
                            options.hls_class_values
                        )
                    }
                )
                for _, window in ni_ds.block_windows(1):
                    ni = ni_ds.read(1, window=window)
                    ni_footprint = ni_ds.read_masks(1, window=window) > 0
                    hls = np.full(ni.shape, 255, dtype=np.uint8)
                    hls_footprint = np.zeros(ni.shape, dtype=bool)

                    for vrt in vrts:
                        candidate = vrt.read(1, window=window)
                        candidate_mask = vrt.read_masks(1, window=window) > 0
                        hls_footprint |= candidate_mask
                        update = _priority(
                            candidate, options.hls_valid_values
                        ) > _priority(hls, options.hls_valid_values)
                        hls[update] = candidate[update]

                    mosaic_ds.write(hls, 1, window=window)

                    ni_evaluable = np.isin(ni, options.ni_valid_values)
                    hls_evaluable = np.isin(hls, options.hls_valid_values)
                    common = ni_evaluable & hls_evaluable
                    ni_usable = np.isin(ni, options.ni_class_values)
                    ni_water = np.isin(ni, options.ni_water_values)
                    hls_water = np.isin(hls, options.hls_water_values)

                    confusion = np.full(ni.shape, CONFUSION_NODATA, dtype=np.uint8)
                    block_tn = common & ~ni_water & ~hls_water
                    block_fp = common & ni_water & ~hls_water
                    block_fn = common & ~ni_water & hls_water
                    block_tp = common & ni_water & hls_water
                    confusion[block_tn] = CONFUSION_TN
                    confusion[block_fp] = CONFUSION_FP
                    confusion[block_fn] = CONFUSION_FN
                    confusion[block_tp] = CONFUSION_TP
                    confusion_ds.write(confusion, 1, window=window)

                    tn += int(np.count_nonzero(block_tn))
                    fp += int(np.count_nonzero(block_fp))
                    fn += int(np.count_nonzero(block_fn))
                    tp += int(np.count_nonzero(block_tp))
                    ni_evaluable_pixels += int(np.count_nonzero(ni_evaluable))
                    ni_usable_pixels += int(np.count_nonzero(ni_usable))
                    class_confusion = np.full(
                        ni.shape, CONFUSION_NODATA, dtype=np.uint8
                    )
                    for ni_index, ni_value in enumerate(options.ni_class_values):
                        ni_class_mask = ni == ni_value
                        if not np.any(ni_class_mask):
                            continue
                        for hls_index, hls_value in enumerate(
                            options.hls_class_values
                        ):
                            class_pair_mask = ni_class_mask & (hls == hls_value)
                            class_matrix[ni_index, hls_index] += int(
                                np.count_nonzero(class_pair_mask)
                            )
                            class_confusion[class_pair_mask] = (
                                ni_index * number_of_hls_classes
                                + hls_index
                                + 1
                            )
                    class_confusion_ds.write(
                        class_confusion, 1, window=window
                    )
                    ni_footprint_pixels += int(np.count_nonzero(ni_footprint))
                    reference_footprint_pixels += int(
                        np.count_nonzero(hls_footprint)
                    )
                    common_footprint_pixels += int(
                        np.count_nonzero(ni_footprint & hls_footprint)
                    )

        metrics = _calculate_metrics(tp, tn, fp, fn)
        usable_comparison_pixels = int(class_matrix.sum())
        usable_pixel_fraction = (
            usable_comparison_pixels / ni_usable_pixels
            if ni_usable_pixels
            else 0.0
        )
        status = (
            "completed"
            if (
                usable_comparison_pixels >= options.min_comparison_pixels
                and usable_pixel_fraction >= options.min_usable_pixel_fraction
            )
            else "skipped_low_coverage"
        )
        if status == "completed":
            os.replace(temporary_mosaic, mosaic_path)
            os.replace(temporary_confusion, confusion_path)
            os.replace(temporary_class_confusion, class_confusion_path)
            _write_class_confusion_outputs(
                pair_directory,
                class_matrix,
                options.ni_class_values,
                options.hls_class_values,
            )
            accuracy_text = (
                f"{metrics['accuracy']:.3f}"
                if metrics["accuracy"] is not None
                else "NA"
            )
            _write_comparison_png(
                ni_path,
                mosaic_path,
                confusion_path,
                png_path,
                options,
                (
                    f"{epoch['epoch_id']} | "
                    f"usable={usable_pixel_fraction:.1%} | OA={accuracy_text}"
                ),
            )
            boundary_result = None
            if options.boundary_comparison:
                boundary_result = _write_boundary_comparison_outputs(
                    ni_path=ni_path,
                    mosaic_path=mosaic_path,
                    pair_directory=pair_directory,
                    options=options,
                    epoch_id=epoch["epoch_id"],
                    full_class_comparison_pixels=usable_comparison_pixels,
                )
        else:
            boundary_result = None
            temporary_mosaic.unlink(missing_ok=True)
            temporary_confusion.unlink(missing_ok=True)
            temporary_class_confusion.unlink(missing_ok=True)

        result = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "pair_id": epoch["epoch_id"],
            "ni_path": str(ni_path),
            "ni_datetime": _format_time(ni_midpoint),
            "hls_epoch_id": epoch["epoch_id"],
            "hls_sensor": epoch["sensor"],
            "hls_start_datetime": _format_time(epoch["start_time"]),
            "hls_stop_datetime": _format_time(epoch["stop_time"]),
            "hls_midpoint_datetime": _format_time(epoch["midpoint"]),
            "delta_hours": delta_hours,
            "abs_delta_days": abs(delta_hours) / 24,
            "hls_tiles": epoch["tiles"],
            "hls_filenames": [path.name for path in reference_paths],
            "ni_footprint_pixels": ni_footprint_pixels,
            "hls_footprint_pixels_on_ni_grid": reference_footprint_pixels,
            "common_footprint_pixels": common_footprint_pixels,
            "ni_footprint_coverage": _safe_ratio(
                common_footprint_pixels, ni_footprint_pixels
            ),
            "ni_evaluable_pixels": ni_evaluable_pixels,
            "ni_usable_pixels": ni_usable_pixels,
            "usable_comparison_pixels": usable_comparison_pixels,
            "usable_pixel_fraction": usable_pixel_fraction,
            **precheck,
            "minimum_comparison_pixels": options.min_comparison_pixels,
            "minimum_usable_pixel_fraction": options.min_usable_pixel_fraction,
            **metrics,
            "ni_water_values": list(options.ni_water_values),
            "ni_valid_values": list(options.ni_valid_values),
            "hls_water_values": list(options.hls_water_values),
            "hls_valid_values": list(options.hls_valid_values),
            "ni_class_values": list(options.ni_class_values),
            "hls_class_values": list(options.hls_class_values),
            "hls_mosaic": mosaic_path.name if status == "completed" else None,
            "confusion_raster": (
                confusion_path.name if status == "completed" else None
            ),
            "class_confusion_raster": (
                class_confusion_path.name if status == "completed" else None
            ),
            "comparison_png": png_path.name if status == "completed" else None,
            "class_confusion_csv": (
                class_csv_path.name if status == "completed" else None
            ),
            "class_confusion_json": (
                class_json_path.name if status == "completed" else None
            ),
            "class_confusion_matrix": (
                class_matrix.tolist() if status == "completed" else None
            ),
            "boundary_comparison": boundary_result,
        }
        _write_json_atomic(metrics_path, result)
        return result
    finally:
        temporary_mosaic.unlink(missing_ok=True)
        temporary_confusion.unlink(missing_ok=True)
        temporary_class_confusion.unlink(missing_ok=True)


def _write_run_comparison_index(
    run_directory: Path,
    pair_results: Sequence[dict[str, Any]],
) -> tuple[Path, Path | None, Path | None]:
    """Write a compact pair index and copy the best quicklook to run root."""

    completed = sorted(
        (result for result in pair_results if result.get("status") == "completed"),
        key=lambda result: (
            -float(result.get("usable_pixel_fraction", 0.0)),
            float(result.get("abs_delta_days", math.inf)),
        ),
    )
    rank_by_id = {
        result["hls_epoch_id"]: rank
        for rank, result in enumerate(completed, start=1)
    }
    skipped = sorted(
        (result for result in pair_results if result.get("status") != "completed"),
        key=lambda result: float(result.get("abs_delta_days", math.inf)),
    )
    ordered_results = [*completed, *skipped]
    quicklook_directory = run_directory / "quicklooks"
    quicklook_directory.mkdir(parents=True, exist_ok=True)
    for stale_path in quicklook_directory.glob("*.png"):
        stale_path.unlink()

    quicklook_by_id: dict[str, Path] = {}
    boundary_quicklook_by_id: dict[str, Path] = {}
    for rank, result in enumerate(completed, start=1):
        epoch_id = result["hls_epoch_id"]
        source = run_directory / "pairs" / epoch_id / "comparison.png"
        destination = quicklook_directory / f"{rank:03d}_{epoch_id}.png"
        shutil.copy2(source, destination)
        quicklook_by_id[epoch_id] = destination.relative_to(run_directory)
        boundary_source = (
            run_directory / "pairs" / epoch_id / "boundary" / "comparison.png"
        )
        if boundary_source.is_file():
            boundary_destination = (
                quicklook_directory / f"{rank:03d}_{epoch_id}_boundary.png"
            )
            shutil.copy2(boundary_source, boundary_destination)
            boundary_quicklook_by_id[epoch_id] = (
                boundary_destination.relative_to(run_directory)
            )

    rows = []
    for result in ordered_results:
        epoch_id = result["hls_epoch_id"]
        pair_rel = Path("pairs") / epoch_id
        boundary = result.get("boundary_comparison") or {}
        rows.append(
            {
                "rank": rank_by_id.get(epoch_id, ""),
                "hls_epoch_id": epoch_id,
                "status": result.get("status"),
                "abs_delta_days": result.get("abs_delta_days"),
                "comparison_pixels": result.get(
                    "usable_comparison_pixels",
                    result.get("estimated_comparison_pixels"),
                ),
                "usable_pixel_fraction": result.get(
                    "usable_pixel_fraction",
                    result.get("estimated_usable_pixel_fraction"),
                ),
                "accuracy": result.get("accuracy"),
                "boundary_status": boundary.get("status"),
                "boundary_comparison_pixels": boundary.get("comparison_pixels"),
                "boundary_accuracy": boundary.get("accuracy"),
                "comparison_png": (
                    str(quicklook_by_id[epoch_id])
                    if result.get("status") == "completed"
                    else ""
                ),
                "boundary_comparison_png": (
                    str(boundary_quicklook_by_id[epoch_id])
                    if epoch_id in boundary_quicklook_by_id
                    else ""
                ),
                "metrics": str(pair_rel / "pair_metrics.json"),
            }
        )

    index_path = run_directory / "comparison_index.csv"
    _write_csv_atomic(
        index_path,
        (
            "rank",
            "hls_epoch_id",
            "status",
            "abs_delta_days",
            "comparison_pixels",
            "usable_pixel_fraction",
            "accuracy",
            "boundary_status",
            "boundary_comparison_pixels",
            "boundary_accuracy",
            "comparison_png",
            "boundary_comparison_png",
            "metrics",
        ),
        rows,
    )

    best_path = run_directory / "best_comparison.png"
    if not completed:
        best_path.unlink(missing_ok=True)
        boundary_best_path = run_directory / "best_boundary_comparison.png"
        boundary_best_path.unlink(missing_ok=True)
        return index_path, None, None

    source = (
        run_directory
        / "pairs"
        / completed[0]["hls_epoch_id"]
        / "comparison.png"
    )
    shutil.copy2(source, best_path)
    boundary_best_path = run_directory / "best_boundary_comparison.png"
    boundary_completed = [
        result
        for result in completed
        if (result.get("boundary_comparison") or {}).get("status") == "completed"
    ]
    if boundary_completed:
        boundary_source = (
            run_directory
            / "pairs"
            / boundary_completed[0]["hls_epoch_id"]
            / "boundary"
            / "comparison.png"
        )
        shutil.copy2(boundary_source, boundary_best_path)
    else:
        boundary_best_path.unlink(missing_ok=True)
        boundary_best_path = None
    return index_path, best_path, boundary_best_path


def run_hls_comparison(
    cfg: Any,
    options: ComparisonOptions,
    comparison_output_dir: str | None = None,
) -> dict[str, Any]:
    """Compare one completed DSWx-NI job with all candidate HLS epochs."""

    if not 0.0 <= options.min_usable_pixel_fraction <= 1.0:
        raise ValueError("min_usable_pixel_fraction must be between 0 and 1")
    if options.min_comparison_pixels < 0:
        raise ValueError("min_comparison_pixels cannot be negative")
    if options.boundary_width_pixels < 1:
        raise ValueError("boundary_width_pixels must be at least 1")
    if options.boundary_min_comparison_pixels < 0:
        raise ValueError("boundary_min_comparison_pixels cannot be negative")
    if options.precheck_max_dimension < 1 or options.png_max_dimension < 1:
        raise ValueError("overview and PNG dimensions must be positive")

    product_paths = cfg.groups.product_path_group
    scratch_directory = Path(product_paths.scratch_path)
    sas_output_directory = Path(product_paths.sas_output_path)
    ni_path = scratch_directory / "full_water_binary_WTR_set.tif"
    if not ni_path.is_file():
        raise FileNotFoundError(
            "Expected DSWx-NI large-area WTR product was not found: " f"{ni_path}"
        )

    input_paths = [str(item) for item in cfg.groups.input_file_group.input_file_path]
    ni_start, ni_stop = _extract_ni_time_range(input_paths)
    ni_midpoint = ni_start + (ni_stop - ni_start) / 2
    run_id = _make_ni_run_id(input_paths, ni_start, ni_stop)

    comparison_root = (
        Path(comparison_output_dir)
        if comparison_output_dir
        else sas_output_directory / "hls_comparison"
    )
    run_directory = comparison_root / "runs" / run_id
    pair_root = run_directory / "pairs"
    # By default, downloaded HLS source assets remain in scratch and are not
    # included in the persistent output upload. Each run has an isolated cache
    # to avoid collisions between concurrent Batch jobs.
    if options.save_hls_assets:
        cache_directory = run_directory / "hls_assets"
    else:
        cache_directory = (
            scratch_directory / "hls_comparison_cache" / run_id / "hls_assets"
        )
    run_directory.mkdir(parents=True, exist_ok=True)

    boxes = _create_valid_search_boxes(
        ni_path, box_size_km=options.search_box_size_km
    )
    search_start = ni_midpoint - timedelta(days=options.search_margin_days)
    search_stop = ni_midpoint + timedelta(days=options.search_margin_days)

    if options.mode == "online":
        assets = _discover_online_hls_assets(
            boxes, start_time=search_start, stop_time=search_stop
        )
        assets = _download_hls_assets(
            assets,
            cache_directory,
            retry_failed_downloads=options.retry_failed_downloads,
        )
    elif options.mode == "offline":
        if not options.hls_input_dir:
            raise ValueError("Offline mode requires --hls-input-dir")
        assets = _discover_offline_hls_assets(
            Path(options.hls_input_dir),
            boxes,
            start_time=search_start,
            stop_time=search_stop,
        )
    else:
        raise ValueError(f"Unsupported HLS mode: {options.mode}")

    epochs = _group_hls_epochs(assets, gap_minutes=options.epoch_gap_minutes)
    pair_results: list[dict[str, Any]] = []
    for index, epoch in enumerate(epochs, start=1):
        LOGGER.info(
            "Comparing HLS epoch %d/%d: %s",
            index,
            len(epochs),
            epoch["epoch_id"],
        )
        pair_results.append(
            _compare_epoch(
                ni_path=ni_path,
                epoch=epoch,
                pair_directory=pair_root / epoch["epoch_id"],
                options=options,
                ni_midpoint=ni_midpoint,
            )
        )

    (
        comparison_index_path,
        best_comparison_path,
        best_boundary_comparison_path,
    ) = _write_run_comparison_index(run_directory, pair_results)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "completed",
        "ni_start_datetime": _format_time(ni_start),
        "ni_stop_datetime": _format_time(ni_stop),
        "ni_midpoint_datetime": _format_time(ni_midpoint),
        "ni_source_inputs": input_paths,
        "ni_comparison_source": str(ni_path),
        "search_start_datetime": _format_time(search_start),
        "search_stop_datetime": _format_time(search_stop),
        "search_box_count": len(boxes),
        "hls_asset_count": len(assets),
        "hls_epoch_count": len(epochs),
        "completed_pair_count": sum(
            result["status"] == "completed" for result in pair_results
        ),
        "skipped_pair_count": sum(
            result["status"] != "completed" for result in pair_results
        ),
        "comparison_index": comparison_index_path.name,
        "best_comparison_png": (
            best_comparison_path.name if best_comparison_path else None
        ),
        "best_boundary_comparison_png": (
            best_boundary_comparison_path.name
            if best_boundary_comparison_path
            else None
        ),
        "hls_assets_saved_in_output": options.save_hls_assets,
        "options": asdict(options),
        "pairs": [
            {
                "hls_epoch_id": result["hls_epoch_id"],
                "status": result["status"],
                "usable_pixel_fraction": result.get(
                    "usable_pixel_fraction",
                    result.get("estimated_usable_pixel_fraction"),
                ),
                "metrics": str(
                    Path("pairs") / result["hls_epoch_id"] / "pair_metrics.json"
                ),
            }
            for result in pair_results
        ],
    }
    _write_json_atomic(run_directory / "run_manifest.json", manifest)
    LOGGER.info("HLS comparison manifest: %s", run_directory / "run_manifest.json")
    return manifest


def aggregate_pair_metrics(input_root: Path, output_directory: Path) -> None:
    """Combine per-pair records created by independent Batch jobs."""

    metric_paths = sorted(input_root.rglob("pair_metrics.json"))
    if not metric_paths:
        raise FileNotFoundError(f"No pair_metrics.json files found under {input_root}")

    raw_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for path in metric_paths:
        with path.open("r", encoding="utf-8") as stream:
            raw_record = json.load(stream)
        raw_record["metrics_path"] = str(path)
        raw_records.append(raw_record)
        record = raw_record.copy()
        record["metrics_path"] = str(path)
        for key in (
            "hls_tiles",
            "hls_filenames",
            "ni_water_values",
            "ni_valid_values",
            "hls_water_values",
            "hls_valid_values",
            "ni_class_values",
            "hls_class_values",
            "class_confusion_matrix",
            "boundary_comparison",
        ):
            if key in record:
                record[key] = json.dumps(record[key], separators=(",", ":"))
        records.append(record)

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "comparison_metrics_all.csv"
    fieldnames = sorted({key for record in records for key in record})
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary_csv, csv_path)

    completed = [
        record for record in raw_records if record.get("status") == "completed"
    ]
    tp = sum(int(record.get("tp", 0)) for record in completed)
    tn = sum(int(record.get("tn", 0)) for record in completed)
    fp = sum(int(record.get("fp", 0)) for record in completed)
    fn = sum(int(record.get("fn", 0)) for record in completed)
    completed_boundary = [
        record["boundary_comparison"]
        for record in completed
        if (record.get("boundary_comparison") or {}).get("status") == "completed"
    ]
    boundary_tp = sum(int(record.get("tp", 0)) for record in completed_boundary)
    boundary_tn = sum(int(record.get("tn", 0)) for record in completed_boundary)
    boundary_fp = sum(int(record.get("fp", 0)) for record in completed_boundary)
    boundary_fn = sum(int(record.get("fn", 0)) for record in completed_boundary)
    class_matrix = None
    class_schema = None
    incompatible_class_matrices = 0
    for record in completed:
        matrix_values = record.get("class_confusion_matrix")
        ni_values = record.get("ni_class_values")
        hls_values = record.get("hls_class_values")
        if matrix_values is None or ni_values is None or hls_values is None:
            continue
        schema = (tuple(ni_values), tuple(hls_values))
        matrix = np.asarray(matrix_values, dtype=np.uint64)
        if class_schema is None:
            class_schema = schema
            class_matrix = np.zeros(matrix.shape, dtype=np.uint64)
        if schema != class_schema or matrix.shape != class_matrix.shape:
            incompatible_class_matrices += 1
            continue
        class_matrix += matrix

    aggregate_class_csv = None
    aggregate_class_json = None
    if class_matrix is not None and class_schema is not None:
        aggregate_class_csv, aggregate_class_json = _write_class_confusion_outputs(
            output_directory,
            class_matrix,
            class_schema[0],
            class_schema[1],
        )

    boundary_class_matrix = None
    boundary_class_schema = None
    incompatible_boundary_class_matrices = 0
    for record in completed_boundary:
        matrix_values = record.get("class_confusion_matrix")
        ni_values = record.get("ni_class_values")
        hls_values = record.get("hls_class_values")
        if matrix_values is None or ni_values is None or hls_values is None:
            continue
        schema = (tuple(ni_values), tuple(hls_values))
        matrix = np.asarray(matrix_values, dtype=np.uint64)
        if boundary_class_schema is None:
            boundary_class_schema = schema
            boundary_class_matrix = np.zeros(matrix.shape, dtype=np.uint64)
        if (
            schema != boundary_class_schema
            or matrix.shape != boundary_class_matrix.shape
        ):
            incompatible_boundary_class_matrices += 1
            continue
        boundary_class_matrix += matrix

    aggregate_boundary_class_csv = None
    aggregate_boundary_class_json = None
    if boundary_class_matrix is not None and boundary_class_schema is not None:
        boundary_output_directory = output_directory / "boundary"
        aggregate_boundary_class_csv, aggregate_boundary_class_json = (
            _write_class_confusion_outputs(
                boundary_output_directory,
                boundary_class_matrix,
                boundary_class_schema[0],
                boundary_class_schema[1],
            )
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "pair_record_count": len(records),
        "completed_pair_count": len(completed),
        "skipped_pair_count": len(records) - len(completed),
        "micro_aggregate": _calculate_metrics(tp, tn, fp, fn),
        "boundary_completed_pair_count": len(completed_boundary),
        "boundary_micro_aggregate": _calculate_metrics(
            boundary_tp, boundary_tn, boundary_fp, boundary_fn
        ),
        "aggregate_class_confusion_csv": (
            aggregate_class_csv.name if aggregate_class_csv else None
        ),
        "aggregate_class_confusion_json": (
            aggregate_class_json.name if aggregate_class_json else None
        ),
        "incompatible_class_matrix_count": incompatible_class_matrices,
        "aggregate_boundary_class_confusion_csv": (
            str(Path("boundary") / aggregate_boundary_class_csv.name)
            if aggregate_boundary_class_csv
            else None
        ),
        "aggregate_boundary_class_confusion_json": (
            str(Path("boundary") / aggregate_boundary_class_json.name)
            if aggregate_boundary_class_json
            else None
        ),
        "incompatible_boundary_class_matrix_count": (
            incompatible_boundary_class_matrices
        ),
    }
    _write_json_atomic(output_directory / "accuracy_summary.json", summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run DSWx-NI followed by DSWx-HLS comparison"
    )
    run_parser.add_argument("input_yaml", help="DSWx-NI runconfig YAML")
    run_parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Reuse an existing full_water_binary_WTR_set.tif",
    )
    run_parser.add_argument("--debug-mode", action="store_true")
    run_parser.add_argument("--log-file")
    run_parser.add_argument(
        "--hls-mode", choices=("online", "offline"), default="online"
    )
    run_parser.add_argument("--hls-input-dir")
    run_parser.add_argument("--comparison-output-dir")
    run_parser.add_argument("--search-margin-days", type=float, default=6.0)
    run_parser.add_argument("--search-box-size-km", type=float, default=50.0)
    run_parser.add_argument("--epoch-gap-minutes", type=float, default=30.0)
    run_parser.add_argument("--min-comparison-pixels", type=int, default=10_000)
    run_parser.add_argument("--min-usable-pixel-fraction", type=float, default=0.05)
    run_parser.add_argument("--precheck-max-dimension", type=int, default=2_048)
    run_parser.add_argument("--png-max-dimension", type=int, default=1_800)
    run_parser.add_argument("--ni-water-values", type=int, nargs="+", default=[1, 8])
    run_parser.add_argument("--ni-valid-values", type=int, nargs="+", default=[0, 1, 8])
    run_parser.add_argument("--hls-water-values", type=int, nargs="+", default=[1, 2])
    run_parser.add_argument(
        "--hls-valid-values", type=int, nargs="+", default=[0, 1, 2]
    )
    run_parser.add_argument(
        "--ni-class-values", type=int, nargs="+", default=[0, 1, 3, 8]
    )
    run_parser.add_argument(
        "--hls-class-values", type=int, nargs="+", default=[0, 1, 2]
    )
    run_parser.add_argument(
        "--boundary-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also evaluate a shoreline band on both sides of valid HLS water "
            "boundaries (enabled by default)."
        ),
    )
    run_parser.add_argument(
        "--boundary-width-pixels",
        type=int,
        default=10,
        help=(
            "Width on each side of an HLS water boundary. At 30 m, the "
            "default 10 pixels is approximately 300 m."
        ),
    )
    run_parser.add_argument(
        "--boundary-min-comparison-pixels",
        type=int,
        default=1_000,
        help="Minimum binary pixels for boundary status=completed.",
    )
    run_parser.add_argument(
        "--save-hls-assets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save downloaded HLS source GeoTIFFs under sas_output_path. "
            "By default they remain under scratch_path and are not uploaded."
        ),
    )
    run_parser.add_argument("--retry-failed-downloads", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Aggregate per-pair metrics from many jobs"
    )
    aggregate_parser.add_argument("input_root")
    aggregate_parser.add_argument("output_directory")
    return parser


def _run_command(args: argparse.Namespace) -> None:
    from dswx_sar.common import _generate_log
    from dswx_sar.dswx_ni import dswx_ni_workflow
    from dswx_sar.nisar.dswx_ni_runconfig import RunConfig

    runconfig_args = argparse.Namespace(
        input_yaml=[args.input_yaml],
        debug_mode=args.debug_mode,
        log_file=args.log_file,
    )
    cfg = RunConfig.load_from_yaml(args.input_yaml, "dswx_ni", runconfig_args)
    _generate_log.configure_log_file(cfg.groups.log_file)

    if not args.comparison_only:
        dswx_ni_workflow(cfg)

    options = ComparisonOptions(
        mode=args.hls_mode,
        hls_input_dir=args.hls_input_dir,
        search_margin_days=args.search_margin_days,
        search_box_size_km=args.search_box_size_km,
        epoch_gap_minutes=args.epoch_gap_minutes,
        min_comparison_pixels=args.min_comparison_pixels,
        min_usable_pixel_fraction=args.min_usable_pixel_fraction,
        precheck_max_dimension=args.precheck_max_dimension,
        png_max_dimension=args.png_max_dimension,
        ni_water_values=tuple(args.ni_water_values),
        ni_valid_values=tuple(args.ni_valid_values),
        hls_water_values=tuple(args.hls_water_values),
        hls_valid_values=tuple(args.hls_valid_values),
        ni_class_values=tuple(args.ni_class_values),
        hls_class_values=tuple(args.hls_class_values),
        boundary_comparison=args.boundary_comparison,
        boundary_width_pixels=args.boundary_width_pixels,
        boundary_min_comparison_pixels=args.boundary_min_comparison_pixels,
        save_hls_assets=args.save_hls_assets,
        retry_failed_downloads=args.retry_failed_downloads,
        overwrite=args.overwrite,
    )
    # run_hls_comparison(cfg, options, args.comparison_output_dir)

    try:
        manifest = run_hls_comparison(
            cfg, options, args.comparison_output_dir
        )
    except Exception:
        LOGGER.warning(
            "DSWx-HLS comparison failed or was unavailable. "
            "Continuing with the DSWx-NI outputs.",
            exc_info=True,
        )
    else:
        if manifest.get("hls_asset_count", 0) == 0:
            LOGGER.warning(
                "No DSWx-HLS scenes were available. "
                "Continuing with the DSWx-NI outputs."
            )
def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        _run_command(args)
    else:
        aggregate_pair_metrics(Path(args.input_root), Path(args.output_directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())