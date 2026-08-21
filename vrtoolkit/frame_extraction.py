import os
import time
import shutil
import subprocess
from typing import Tuple, Optional

import pandas as pd


def check_ffmpeg() -> bool:
    """Return True if `ffmpeg` binary is available in PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def find_video_file(video_root: str, video_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Try multiple naming patterns and extensions to locate a video file.

    Returns (video_path, found_pattern) or (None, None).
    """
    patterns = [
        f"{video_name}.mp4",
        f"{video_name}.MP4",
        f"{video_name}.mov",
        f"{video_name}.MOV",
        f"{video_name}.avi",
        f"{video_name.replace('GH', 'GL')}.mp4",
        f"{video_name.replace('GH', 'GL')}.MP4",
        f"{video_name.replace('GL', 'GH')}.mp4",
        f"{video_name.replace('GL', 'GH')}.MP4",
        f"{video_name.replace('GS', 'GL')}.mp4",
        f"{video_name.replace('GS', 'GL')}.MP4",
    ]
    for pattern in patterns:
        p = os.path.join(video_root, pattern)
        if os.path.isfile(p):
            return p, pattern
    return None, None


def extract_frame_ffmpeg(
    video_path: str, timestamp_sec: float, output_path: str, max_retries: int = 2
) -> Tuple[bool, Optional[str]]:
    """Extract a single frame using FFmpeg with detailed error reporting."""
    for attempt in range(max_retries + 1):
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-accurate_seek",
                "-ss",
                str(timestamp_sec),
                "-i",
                video_path,
                "-vframes",
                "1",
                "-q:v",
                "1",
                "-pix_fmt",
                "yuvj420p",
                "-loglevel",
                "warning",
                output_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and os.path.exists(output_path):
                return True, None
            details = []
            if result.stderr:
                details.append(f"stderr: {result.stderr.strip()}")
            if result.stdout:
                details.append(f"stdout: {result.stdout.strip()}")
            details.append(f"return_code: {result.returncode}")
            details.append(f"timestamp: {timestamp_sec}s")
            if attempt < max_retries:
                continue
            return False, " | ".join(details) if details else "Unknown FFmpeg error"
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                continue
            return False, "FFmpeg timeout"
        except Exception as e:
            if attempt < max_retries:
                continue
            return False, str(e)
    return False, "Max retries exceeded"


def get_video_info_ffmpeg(video_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (duration_seconds, fps) using ffprobe, or (None, None) on failure."""
    try:
        duration_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        fps_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "stream=r_frame_rate",
            "-select_streams",
            "v:0",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        dur = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
        fps = subprocess.run(fps_cmd, capture_output=True, text=True, timeout=10)
        duration = float(dur.stdout.strip()) if dur.returncode == 0 and dur.stdout.strip() else None
        fps_val = None
        if fps.returncode == 0 and fps.stdout.strip():
            s = fps.stdout.strip()
            if "/" in s:
                n, d = s.split("/")
                fps_val = float(n) / float(d)
            else:
                fps_val = float(s)
        return duration, fps_val
    except Exception:
        return None, None


def validate_frame_numbers(
    frames_df: pd.DataFrame, video_path: str
) -> Tuple[pd.DataFrame, float]:
    """Filter frames with out-of-range frame numbers relative to video FPS/duration.

    Returns (valid_frames_df, fps_used).
    """
    duration, fps = get_video_info_ffmpeg(video_path)
    if duration is None or fps is None or fps <= 0:
        return frames_df, 30.0
    total_frames = int(duration * fps)
    valid = frames_df[frames_df["frame_number"] < total_frames]
    return valid, fps


def run_ffmpeg_extraction(
    csv_path: str,
    video_root: str,
    output_dir: str,
) -> None:
    """Extract frames listed by (matched_file, frame_number, ObjectId) using FFmpeg.

    Reads the CSV, groups by video, validates frame numbers, and writes output JPEGs
    to `output_dir` named as `{ObjectId}.jpg` (auto-incrementing suffixes for duplicates).
    """
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg/ffprobe not found in PATH. Please install FFmpeg.")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if not {"matched_file", "frame_number", "ObjectId"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns: matched_file, frame_number, ObjectId"
        )

    unique_frames = df.drop_duplicates(subset=["matched_file", "frame_number"])

    grouped = unique_frames.groupby("matched_file")
    for video_name, video_frames in grouped:
        video_path, _ = find_video_file(video_root, video_name)
        if video_path is None:
            continue
        valid_frames, fps = validate_frame_numbers(video_frames, video_path)
        if valid_frames.empty:
            continue
        for _, row in valid_frames.iterrows():
            frame_number = row["frame_number"]
            timestamp_sec = frame_number / fps
            panoid = row["ObjectId"]
            # handle duplicates
            base_filename = f"{panoid}.jpg"
            out_path = os.path.join(output_dir, base_filename)
            counter = 1
            while os.path.exists(out_path) and counter <= 10:
                base_filename = f"{panoid}_{counter}.jpg"
                out_path = os.path.join(output_dir, base_filename)
                counter += 1
            if counter > 10:
                continue
            extract_frame_ffmpeg(video_path, timestamp_sec, out_path)
