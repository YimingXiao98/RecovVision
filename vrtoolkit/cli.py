import argparse

from .spatial_matching import run_spatial_index_matching
from .frame_extraction import run_ffmpeg_extraction
from .orientation import calculate_orientations_frame_based
from .dewarp import run_frame_based_dewarping, run_smart_dewarping
from .VLMpipeline import run_vlm_pipeline


def main():
    parser = argparse.ArgumentParser(
        prog="videoprocessing",
        description="Video processing toolkit: matching, extracting frames, orientation, dewarping.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_match = sub.add_parser("match", help="Match footprints to GPS frames")
    p_match.add_argument("ref_csv", help="Reference footprints CSV (lon/lat)")
    p_match.add_argument("gps_folder", help="Folder containing *_GoPro Max-GPS5.csv files")
    p_match.add_argument("out_csv", help="Output CSV for matches")
    p_match.add_argument("--buffer", type=float, default=25.0, help="Buffer meters")

    p_extract = sub.add_parser("extract", help="Extract frames via FFmpeg")
    p_extract.add_argument("matches_csv", help="CSV including matched_file, frame_number, ObjectId")
    p_extract.add_argument("video_root", help="Folder containing source videos")
    p_extract.add_argument("output_dir", help="Folder to save frames as JPG")

    p_orient = sub.add_parser("orient", help="Compute vehicle orientations per building")
    p_orient.add_argument("matches_csv", help="CSV including matched_file, frame_number")
    p_orient.add_argument("gps_folder", help="Folder containing GPS CSVs")
    p_orient.add_argument("out_csv", help="Output CSV with orientation column")
    p_orient.add_argument("--window", type=int, default=15, help="Frame window for averaging")

    p_dewarp = sub.add_parser("dewarp", help="Dewarp frames using orientation")
    p_dewarp.add_argument("csv", help="CSV with ObjectId, coords, vehicle coords, orientation")
    p_dewarp.add_argument("input_dir", help="Input frames folder")
    p_dewarp.add_argument("output_dir", help="Output dewarped frames folder")
    p_dewarp.add_argument("--h_fov", type=float, default=90.0)
    p_dewarp.add_argument("--pitch", type=float, default=0.0)
    p_dewarp.add_argument("--width", type=int, default=1920)
    p_dewarp.add_argument("--aspect", type=str, default="16:9")
    p_dewarp.add_argument("--yaw_offset", type=float, default=-90.0)

    p_sdewarp = sub.add_parser("dewarp-smart", help="Dewarp, skipping existing outputs")
    p_sdewarp.add_argument("csv", help="CSV with ObjectId, coords, vehicle coords, orientation")
    p_sdewarp.add_argument("input_dir", help="Input frames folder")
    p_sdewarp.add_argument("output_dir", help="Output dewarped frames folder")
    p_sdewarp.add_argument("--h_fov", type=float, default=90.0)
    p_sdewarp.add_argument("--pitch", type=float, default=0.0)
    p_sdewarp.add_argument("--width", type=int, default=1920)
    p_sdewarp.add_argument("--aspect", type=str, default="16:9")
    p_sdewarp.add_argument("--yaw_offset", type=float, default=-90.0)

    p_vlm = sub.add_parser("vlm", help="Run VLM occupancy pipeline over images")
    p_vlm.add_argument("csv", help="CSV with 'ObjectId' or 'objectid' column")
    p_vlm.add_argument("image_dir", help="Directory containing images named '<id>.jpg'")
    p_vlm.add_argument("out_csv", help="Output CSV path for predictions")
    p_vlm.add_argument("--no-llm", action="store_true", help="Use rule-based classifier instead of LLM")
    p_vlm.add_argument("--gt_csv", default=None, help="Optional ground truth CSV for evaluation")
    p_vlm.add_argument("--image_id_column", default=None, help="Explicit image id column name if not standard")

    args = parser.parse_args()

    if args.cmd == "match":
        run_spatial_index_matching(
            args.ref_csv,
            args.gps_folder,
            buffer_distance=args.buffer,
            add_object_ids=True,
            output_csv=args.out_csv,
        )
    elif args.cmd == "extract":
        run_ffmpeg_extraction(args.matches_csv, args.video_root, args.output_dir)
    elif args.cmd == "orient":
        calculate_orientations_frame_based(
            args.matches_csv, args.gps_folder, output_csv=args.out_csv, window_frames=args.window
        )
    elif args.cmd == "dewarp":
        run_frame_based_dewarping(
            args.csv,
            args.input_dir,
            args.output_dir,
            h_fov=args.h_fov,
            pitch_angle=args.pitch,
            output_width=args.width,
            aspect_ratio=args.aspect,
            yaw_offset=args.yaw_offset,
        )
    elif args.cmd == "dewarp-smart":
        run_smart_dewarping(
            args.csv,
            args.input_dir,
            args.output_dir,
            h_fov=args.h_fov,
            pitch_angle=args.pitch,
            output_width=args.width,
            aspect_ratio=args.aspect,
            yaw_offset=args.yaw_offset,
        )
    elif args.cmd == "vlm":
        run_vlm_pipeline(
            csv_path=args.csv,
            base_image_dir=args.image_dir,
            output_csv_path=args.out_csv,
            use_llm=not args.no_llm,
            ground_truth_csv=args.gt_csv,
            image_id_column=args.image_id_column,
        )


if __name__ == "__main__":
    main()
