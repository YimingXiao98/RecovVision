import os
from typing import Optional

import pandas as pd
from math import radians, degrees, atan2, sin, cos


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return initial bearing in degrees [0, 360)."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    y = sin(d_lon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(d_lon)
    bearing = degrees(atan2(y, x))
    return (bearing + 360) % 360


def find_points_before_after_frame(
    gps_data: pd.DataFrame, target_frame_number: int, window_frames: int = 15
):
    """Return averaged numeric rows before and after the target frame index.

    Returns (before_point_series, after_point_series) or (None, None) if not found.
    """
    max_frame = len(gps_data) - 1
    before_start = max(0, target_frame_number - window_frames)
    before_end = target_frame_number
    after_start = target_frame_number + 1
    after_end = min(max_frame + 1, target_frame_number + window_frames + 1)

    if before_start < before_end:
        before_points = gps_data.iloc[before_start:before_end]
    else:
        before_points = pd.DataFrame()
    if after_start < after_end:
        after_points = gps_data.iloc[after_start:after_end]
    else:
        after_points = pd.DataFrame()

    if before_points.empty or after_points.empty:
        return None, None

    before_point = (
        before_points.select_dtypes(include="number").iloc[-(min(len(before_points), 3)) :].mean()
    )
    after_point = (
        after_points.select_dtypes(include="number").iloc[: min(len(after_points), 3)].mean()
    )
    return before_point, after_point


def calculate_orientations_frame_based(
    buildings_csv: str,
    gps_folder: str,
    output_csv: Optional[str] = None,
    window_frames: int = 15,
) -> pd.DataFrame:
    """Compute per-building vehicle orientation (bearing) using frame numbers.

    Reads a CSV containing at least [matched_file, frame_number]. Looks up the
    GPS file `{matched_file}_GoPro Max-GPS5.csv` (with GH/GL fallback) in `gps_folder`.
    Adds an `orientation` column to the buildings and writes to `output_csv` if set.
    """
    footprints = pd.read_csv(buildings_csv)
    orientations: list[float | None] = []

    for _, row in footprints.iterrows():
        frame_number = int(row["frame_number"]) if not pd.isna(row["frame_number"]) else None
        video_file = str(row["matched_file"]) if "matched_file" in footprints.columns else None
        if frame_number is None or video_file is None:
            orientations.append(None)
            continue

        gps_file = os.path.join(gps_folder, f"{video_file}_GoPro Max-GPS5.csv")
        if "GH" in video_file:
            gps_file = gps_file.replace("GH", "GL")
        if not os.path.exists(gps_file):
            orientations.append(None)
            continue

        gps = pd.read_csv(gps_file)
        if gps.empty or frame_number >= len(gps):
            orientations.append(None)
            continue

        before_point, after_point = find_points_before_after_frame(
            gps, frame_number, window_frames=window_frames
        )
        if before_point is None or after_point is None:
            orientations.append(None)
            continue

        bearing = calculate_bearing(
            before_point["GPS (Lat.) [deg]"],
            before_point["GPS (Long.) [deg]"],
            after_point["GPS (Lat.) [deg]"],
            after_point["GPS (Long.) [deg]"],
        )
        orientations.append(bearing)

    footprints["orientation"] = orientations
    if output_csv:
        footprints.to_csv(output_csv, index=False)
    return footprints

