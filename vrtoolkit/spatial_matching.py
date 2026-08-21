import os
import glob
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer


def _detect_ref_columns(ref_df: pd.DataFrame) -> tuple[str, str]:
    """Detect longitude/latitude columns from common variants.

    Returns a tuple (lon_col, lat_col).
    Raises ValueError if not found.
    """
    candidates = [
        ("long", "lat"),
        ("longitude", "latitude"),
        ("x", "y"),
        ("Center_Longitude", "Center_Latitude"),
    ]
    cols = set(c.strip() for c in ref_df.columns)
    for lon, lat in candidates:
        if lon in cols and lat in cols:
            return lon, lat
    raise ValueError(
        "Could not detect reference longitude/latitude columns. "
        f"Available columns: {list(ref_df.columns)}"
    )


def run_spatial_index_matching(
    ref_coords_path: str,
    coords_folder: str,
    buffer_distance: float = 25.0,
    add_object_ids: bool = True,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Match building footprint points to nearest GPS track points using a spatial index.

    - Uses frame numbers (row indices in the GPS CSV) instead of timestamps.
    - Returns a DataFrame containing original footprint data plus columns:
      [matched_file, frame_number, vehicle_x, vehicle_y, ObjectId (optional)].

    Args:
        ref_coords_path: CSV with building footprint coordinates (columns long/lat or similar).
        coords_folder: Folder containing GoPro GPS CSV files: `*_GoPro Max-GPS5.csv`.
        buffer_distance: Buffer distance in meters around footprints for candidate matching.
        add_object_ids: If True, adds unique sequential ObjectId values.
        output_csv: If provided, saves the merged result to this CSV.
    """
    # Load reference points and convert to projected CRS for buffering
    ref_df = pd.read_csv(ref_coords_path)
    lon_col, lat_col = _detect_ref_columns(ref_df)
    ref_gdf = gpd.GeoDataFrame(
        ref_df,
        geometry=gpd.points_from_xy(ref_df[lon_col], ref_df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=5070)

    # Build buffered geometries and spatial index
    ref_buffered = ref_gdf.copy()
    ref_buffered["geometry"] = ref_buffered.geometry.buffer(buffer_distance)
    sindex = ref_buffered.sindex

    # Prepare GPS file discovery and transformer
    gps_pattern = os.path.join(coords_folder, "*_GoPro Max-GPS5.csv")
    files_to_process = sorted(os.path.basename(p) for p in glob.glob(gps_pattern))
    transformer = Transformer.from_crs("EPSG:4326", ref_gdf.crs, always_xy=True)

    matches = []
    for filename in files_to_process:
        file_path = os.path.join(coords_folder, filename)
        if not os.path.isfile(file_path):
            continue
        df = pd.read_csv(file_path)
        if df.empty:
            continue

        # Project GPS points to ref CRS
        x_coords, y_coords = transformer.transform(
            df["GPS (Long.) [deg]"].values, df["GPS (Lat.) [deg]"].values
        )
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(x_coords, y_coords), crs=ref_gdf.crs
        )

        for idx, gps_point in gdf.iterrows():
            geom = gps_point.geometry
            candidates = list(sindex.intersection(geom.bounds))
            if not candidates:
                continue
            containing = ref_buffered.loc[candidates]
            actual = containing[containing.contains(geom)]
            for ref_idx in actual.index:
                ref_point = ref_gdf.loc[ref_idx]
                dist = geom.distance(ref_point.geometry)
                matches.append(
                    {
                        "ref_index": ref_idx,
                        "frame_number": idx,
                        "matched_file": filename.split("_GoPro Max-GPS5.csv")[0],
                        "vehicle_x": df.loc[idx, "GPS (Long.) [deg]"],
                        "vehicle_y": df.loc[idx, "GPS (Lat.) [deg]"],
                        "distance": dist,
                    }
                )

    if not matches:
        result = ref_gdf.reset_index(drop=False)
        result = result.drop(columns=["geometry"]).rename(columns={"index": "ref_index"})
        if add_object_ids:
            result["ObjectId"] = range(1, len(result) + 1)
        if output_csv:
            result.to_csv(output_csv, index=False)
        return result

    matches_df = pd.DataFrame(matches)
    best = matches_df.loc[matches_df.groupby("ref_index")["distance"].idxmin()].drop(
        columns=["distance"]
    )
    merged = (
        ref_gdf.reset_index()
        .merge(best, left_on="index", right_on="ref_index", how="inner")
        .drop(columns=["geometry", "ref_index"])
    )
    # Optional clean matched_file like original code did
    if "matched_file" in merged.columns:
        merged["matched_file"] = merged["matched_file"].astype(str).str.split("_").str[0]

    # Ensure unique ObjectId if requested
    if add_object_ids or "ObjectId" not in merged.columns:
        merged["ObjectId"] = range(1, len(merged) + 1)

    if output_csv:
        merged.to_csv(output_csv, index=False)

    return merged

