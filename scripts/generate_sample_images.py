import argparse
import os
import re
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def sanitize_id(value) -> str:
    s = str(value).strip()
    return re.sub(r"[^0-9A-Za-z]+", "_", s)


def parse_size(size_str: str) -> Tuple[int, int]:
    if "x" not in size_str:
        raise ValueError("--size must be in WxH form, e.g., 640x360")
    w, h = size_str.lower().split("x", 1)
    return int(w), int(h)


def resolve_ids_from_csv(csv_path: str, id_col: str | None = None) -> List[str]:
    if pd is None:
        raise RuntimeError("pandas is required to read CSV. Install pandas.")
    df = pd.read_csv(csv_path)
    if id_col is None:
        for cand in ("ObjectId", "objectid"):
            if cand in df.columns:
                id_col = cand
                break
    if id_col is None or id_col not in df.columns:
        raise ValueError(
            f"Could not find id column. Provide --id-col. Available: {list(df.columns)}"
        )
    return [str(v) for v in df[id_col].tolist()]


def draw_placeholder(img: Image.Image, text: str, text_color: str, bg_color: str) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([(0, 0), (w, h)], fill=bg_color)
    house_w, house_h = int(w * 0.5), int(h * 0.5)
    house_x0 = (w - house_w) // 2
    house_y0 = (h - house_h) // 2 + int(h * 0.05)
    house_x1 = house_x0 + house_w
    house_y1 = house_y0 + house_h
    d.rectangle([(house_x0, house_y0), (house_x1, house_y1)], outline="#ffffff", width=3)
    roof_peak = (w // 2, house_y0 - int(h * 0.1))
    d.polygon([roof_peak, (house_x0, house_y0), (house_x1, house_y0)], outline="#ffffff")
    door_w = int(house_w * 0.15)
    door_h = int(house_h * 0.35)
    door_x0 = w // 2 - door_w // 2
    door_y1 = house_y1 - int(house_h * 0.05)
    door_x1 = door_x0 + door_w
    door_y0 = door_y1 - door_h
    d.rectangle([(door_x0, door_y0), (door_x1, door_y1)], outline="#ffffff", width=2)
    text_str = f"ObjectId: {text}"
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, w // 20))
    except Exception:
        font = ImageFont.load_default()
    tw, th = d.textsize(text_str, font=font)
    tx = (w - tw) // 2
    ty = max(5, (house_y0 // 2) - th // 2)
    d.text((tx, ty), text_str, fill=text_color, font=font)


def generate_images(
    ids: Sequence[str],
    out_dir: str,
    size: Tuple[int, int] = (640, 360),
    bg_color: str = "#2b6cb0",
    text_color: str = "#ffffff",
    overwrite: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for id_val in ids:
        sid = sanitize_id(id_val)
        out_path = os.path.join(out_dir, f"{sid}.jpg")
        if os.path.exists(out_path) and not overwrite:
            continue
        img = Image.new("RGB", size)
        draw_placeholder(img, sid, text_color=text_color, bg_color=bg_color)
        img.save(out_path, format="JPEG", quality=90)


def main():
    parser = argparse.ArgumentParser(
        description="Generate placeholder images named <ObjectId>.jpg for testing the VLM pipeline."
    )
    parser.add_argument("--csv", help="CSV path containing ids", default=None)
    parser.add_argument("--id-col", help="Column name with ids", default=None)
    parser.add_argument("--ids", nargs="*", help="Explicit list of ids", default=None)
    parser.add_argument(
        "--out",
        default="samples/images/frames_dewarped",
        help="Output directory for generated images",
    )
    parser.add_argument("--size", default="640x360", help="Image size WxH, e.g., 640x360")
    parser.add_argument("--bg", default="#2b6cb0", help="Background color hex")
    parser.add_argument("--text", default="#ffffff", help="Text color hex")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    ids: List[str] = []
    if args.csv:
        ids.extend(resolve_ids_from_csv(args.csv, args.id_col))
    if args.ids:
        ids.extend([str(x) for x in args.ids])
    if not ids:
        raise SystemExit("No ids resolved. Provide --csv or --ids.")

    size = parse_size(args.size)
    generate_images(
        ids,
        args.out,
        size=size,
        bg_color=args.bg,
        text_color=args.text,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
