import os
import glob
import math
import subprocess
from typing import Optional, Union

import pandas as pd

from .orientation import calculate_bearing


def calculate_vertical_fov(horizontal_fov_deg: float, aspect_ratio_w_h: float) -> float:
    h_fov_rad = math.radians(horizontal_fov_deg)
    v_fov_rad = 2 * math.atan(math.tan(h_fov_rad / 2) / aspect_ratio_w_h)
    return math.degrees(v_fov_rad)


def create_output_directory(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def dewarp_single_frame(
    input_path: str,
    output_path: str,
    yaw: float,
    pitch: float = 0.0,
    roll: float = 0.0,
    output_width: int = 1920,
    aspect_ratio_str: str = "16:9",
    horizontal_fov: float = 90.0,
) -> bool:
    """Dewarp a single equirectangular image frame using FFmpeg v360 filter."""
    # If the image is not panoramic, just copy it
    try:
        from PIL import Image

        with Image.open(input_path) as img:
            w, h = img.size
            if h == 0 or w / h < 2.0:
                import shutil

                shutil.copy2(input_path, output_path)
                return True
    except Exception:
        # Continue anyway; try running FFmpeg
        pass

    try:
        w_ratio, h_ratio = map(int, aspect_ratio_str.split(":"))
        aspect_ratio = w_ratio / h_ratio
    except Exception:
        aspect_ratio = 16 / 9

    output_height = int(output_width / aspect_ratio)
    vertical_fov = calculate_vertical_fov(horizontal_fov, aspect_ratio)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        (
            f"v360=e:rectilinear:h_fov={horizontal_fov}:v_fov={vertical_fov}:"
            f"yaw={yaw}:pitch={pitch}:roll={roll}:w={output_width}:h={output_height}"
        ),
        "-q:v",
        "2",
        output_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode == 0
    except Exception:
        return False


def run_frame_based_dewarping(
    csv_path: str = "./matched_filtered_buildings_orientation.csv",
    frames_input_dir: str = "./frames_output",
    frames_output_dir: str = "./frames_dewarped",
    h_fov: float = 90.0,
    pitch_angle: float = 0.0,
    output_width: int = 1920,
    aspect_ratio: str = "16:9",
    yaw_offset: float = -90.0,
) -> None:
    """Dewarp frames using camera-to-building yaw computed per row.

    Requires CSV columns: ObjectId, vehicle_x, vehicle_y, Center_Longitude, Center_Latitude, orientation.
    Accepts common variants via column mapping.
    """
    create_output_directory(frames_output_dir)
    df = pd.read_csv(csv_path)

    mapping = {
        "ObjectId": ["ObjectId", "objectid", "object_id"],
        "Center_Longitude": ["Center_Longitude", "x", "longitude", "long"],
        "Center_Latitude": ["Center_Latitude", "y", "latitude", "lat"],
        "vehicle_x": ["vehicle_x", "camera_lon", "camera_longitude", "vehicle_x_y"],
        "vehicle_y": ["vehicle_y", "camera_lat", "camera_latitude", "vehicle_y_y"],
        "orientation": ["orientation", "bearing", "heading"],
    }

    # Resolve actual columns
    actual: dict[str, str] = {}
    for key, candidates in mapping.items():
        for c in candidates:
            if c in df.columns:
                actual[key] = c
                break
        if key not in actual:
            raise ValueError(f"Missing required column for {key}. Candidates: {candidates}")

    for _, row in df.iterrows():
        object_id = row[actual["ObjectId"]]
        building_lon = row[actual["Center_Longitude"]]
        building_lat = row[actual["Center_Latitude"]]
        camera_lon = row[actual["vehicle_x"]]
        camera_lat = row[actual["vehicle_y"]]
        vehicle_orientation = row[actual["orientation"]]

        if any(pd.isna(v) for v in [building_lon, building_lat, camera_lon, camera_lat, vehicle_orientation]):
            continue

        bearing_to_building = calculate_bearing(camera_lat, camera_lon, building_lat, building_lon)
        required_yaw = (bearing_to_building - float(vehicle_orientation)) % 360
        normalized_yaw = required_yaw + yaw_offset
        if normalized_yaw > 180:
            normalized_yaw -= 360
        elif normalized_yaw <= -180:
            normalized_yaw += 360

        # Find input frames by patterns
        patterns = [
            os.path.join(frames_input_dir, f"{object_id}.jpg"),
            os.path.join(frames_input_dir, f"{object_id}_*.jpg"),
            os.path.join(frames_input_dir, f"*/{object_id}.jpg"),
            os.path.join(frames_input_dir, f"*/{object_id}_*.jpg"),
        ]
        found_frames: list[str] = []
        for pat in patterns:
            matches = glob.glob(pat)
            if matches:
                found_frames.extend(matches)
                break
        for input_frame in found_frames:
            out_path = os.path.join(frames_output_dir, os.path.basename(input_frame))
            dewarp_single_frame(
                input_path=input_frame,
                output_path=out_path,
                yaw=normalized_yaw,
                pitch=pitch_angle,
                roll=0.0,
                horizontal_fov=h_fov,
                output_width=output_width,
                aspect_ratio_str=aspect_ratio,
            )


def dewarp_single_frame_skip_existing(
    input_path: str,
    output_path: str,
    yaw: float,
    pitch: float = 0.0,
    roll: float = 0.0,
    output_width: int = 1920,
    aspect_ratio_str: str = "16:9",
    horizontal_fov: float = 90.0,
) -> Union[bool, str]:
    if os.path.exists(output_path):
        return "skipped"
    return dewarp_single_frame(
        input_path,
        output_path,
        yaw,
        pitch=pitch,
        roll=roll,
        output_width=output_width,
        aspect_ratio_str=aspect_ratio_str,
        horizontal_fov=horizontal_fov,
    )


def run_smart_dewarping(
    csv_path: str = "./matched_filtered_buildings_orientation.csv",
    frames_input_dir: str = "./frames_output",
    frames_output_dir: str = "./frames_dewarped",
    h_fov: float = 90.0,
    pitch_angle: float = 0.0,
    output_width: int = 1920,
    aspect_ratio: str = "16:9",
    yaw_offset: float = -90.0,
) -> None:
    os.makedirs(frames_output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Basic mapping for specific CSV variant
    required = ["ObjectId", "lat", "long", "vehicle_x_y", "vehicle_y_y", "orientation"]
    if not set(required).issubset(df.columns):
        # Fall back to generic runner if the CSV isn't in the "fixed" shape
        run_frame_based_dewarping(
            csv_path,
            frames_input_dir,
            frames_output_dir,
            h_fov,
            pitch_angle,
            output_width,
            aspect_ratio,
            yaw_offset,
        )
        return

    for _, row in df.iterrows():
        object_id = row["ObjectId"]
        building_lat = row["lat"]
        building_lon = row["long"]
        camera_lat = row["vehicle_y_y"]
        camera_lon = row["vehicle_x_y"]
        vehicle_orientation = row["orientation"]
        if any(pd.isna(v) for v in [building_lat, building_lon, camera_lat, camera_lon, vehicle_orientation]):
            continue
        out_frame = os.path.join(frames_output_dir, f"{object_id}.jpg")
        if os.path.exists(out_frame):
            continue
        in_frame = os.path.join(frames_input_dir, f"{object_id}.jpg")
        if not os.path.exists(in_frame):
            continue
        bearing = calculate_bearing(camera_lat, camera_lon, building_lat, building_lon)
        required_yaw = (bearing - float(vehicle_orientation)) % 360
        normalized_yaw = required_yaw + yaw_offset
        if normalized_yaw > 180:
            normalized_yaw -= 360
        elif normalized_yaw <= -180:
            normalized_yaw += 360
        dewarp_single_frame_skip_existing(
            input_path=in_frame,
            output_path=out_frame,
            yaw=normalized_yaw,
            pitch=pitch_angle,
            roll=0.0,
            horizontal_fov=h_fov,
            output_width=output_width,
            aspect_ratio_str=aspect_ratio,
        )
