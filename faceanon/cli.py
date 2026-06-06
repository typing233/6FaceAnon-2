import argparse
import os
import sys
import time

import cv2

from .config import EngineConfig, DetectorConfig, TrackerConfig, AnonymizerConfig, AnonymizationType
from .engine import FaceAnonEngine
from .utils import get_bundled_model_path


def main():
    parser = argparse.ArgumentParser(
        prog="faceanon",
        description="Video privacy protection via CenterFace detection and anonymization",
    )
    parser.add_argument("--model", type=str, default=None, help="Path to CenterFace ONNX model")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--method", type=str, choices=["gaussian_blur", "mosaic"], default="gaussian_blur")
    parser.add_argument("--intensity", type=float, default=0.8, help="Anonymization intensity 0.0-1.0")
    parser.add_argument("--expand-ratio", type=float, default=0.1)
    parser.add_argument("--detect-every-n", type=int, default=None)

    subparsers = parser.add_subparsers(dest="command")

    # Image subcommand
    img_parser = subparsers.add_parser("image", help="Process a single image")
    img_parser.add_argument("-i", "--input", required=True)
    img_parser.add_argument("-o", "--output", required=True)

    # Video subcommand
    vid_parser = subparsers.add_parser("video", help="Process a video file")
    vid_parser.add_argument("-i", "--input", required=True)
    vid_parser.add_argument("-o", "--output", required=True)

    # Camera subcommand
    cam_parser = subparsers.add_parser("camera", help="Real-time camera/stream processing")
    cam_parser.add_argument("--source", type=str, default="0", help="Camera index (int) or stream URL")
    cam_parser.add_argument("--output", type=str, default=None, help="Output video file path")
    cam_parser.add_argument("--display", action="store_true", default=True, dest="display")
    cam_parser.add_argument("--no-display", action="store_false", dest="display")

    # Batch subcommand
    batch_parser = subparsers.add_parser("batch", help="Batch process a directory of files")
    batch_parser.add_argument("--input-dir", required=True, help="Source directory")
    batch_parser.add_argument("--output-dir", required=True, help="Destination directory")
    batch_parser.add_argument("--recursive", action="store_true", default=True, dest="recursive")
    batch_parser.add_argument("--no-recursive", action="store_false", dest="recursive")
    batch_parser.add_argument("--report", type=str, default=None, help="Path for JSON report (default: <output-dir>/report.json)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    model_path = args.model or get_bundled_model_path()

    detect_every_n = args.detect_every_n
    input_size = (640, 640)
    if args.command == "camera":
        if detect_every_n is None:
            detect_every_n = 3
        input_size = (320, 320)
    else:
        if detect_every_n is None:
            detect_every_n = 1

    config = EngineConfig(
        detector=DetectorConfig(
            model_path=model_path,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            input_size=input_size,
        ),
        tracker=TrackerConfig(),
        anonymizer=AnonymizerConfig(
            method=AnonymizationType(args.method),
            intensity=args.intensity,
            expand_ratio=args.expand_ratio,
        ),
        detect_every_n=detect_every_n,
    )

    engine = FaceAnonEngine(config)

    if args.command == "image":
        _process_image(engine, args.input, args.output)
    elif args.command == "video":
        _process_video(engine, args.input, args.output)
    elif args.command == "camera":
        _run_camera(engine, args)
    elif args.command == "batch":
        _run_batch(engine, args)


def _process_image(engine: FaceAnonEngine, input_path: str, output_path: str):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: cannot read image {input_path}", file=sys.stderr)
        sys.exit(1)

    result = engine.process_image(image)
    cv2.imwrite(output_path, result.anonymized_frame)
    print(f"Processed: {len(result.detections)} face(s) detected -> {output_path}")


def _process_video(engine: FaceAnonEngine, input_path: str, output_path: str):
    start_time = time.time()

    def callback(idx, total, result):
        if idx % 30 == 0:
            elapsed = time.time() - start_time
            fps = (idx + 1) / max(elapsed, 0.001)
            faces = len(result.tracks)
            progress = f"{idx + 1}/{total}" if total > 0 else str(idx + 1)
            print(f"\r  [{progress}] {fps:.1f} fps, {faces} face(s)", end="", file=sys.stderr)

    total_frames = engine.process_video(input_path, output_path, callback=callback)

    elapsed = time.time() - start_time
    avg_fps = total_frames / max(elapsed, 0.001)
    print(f"\nDone: {total_frames} frames in {elapsed:.1f}s ({avg_fps:.1f} fps avg) -> {output_path}", file=sys.stderr)


def _run_camera(engine: FaceAnonEngine, args):
    from .camera import CameraProcessor, CameraConfig

    source = _parse_source(args.source)
    cam_config = CameraConfig(
        source=source,
        output_path=args.output,
        display=args.display,
    )

    processor = CameraProcessor(engine, cam_config)
    print(f"Starting camera (source={source}, display={args.display})")
    print("Press 'q' in the window or Ctrl+C to stop.")
    processor.run()
    print("Camera stopped.")


def _run_batch(engine: FaceAnonEngine, args):
    from .batch import BatchProcessor, BatchConfig

    batch_config = BatchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
    )

    processor = BatchProcessor(engine, batch_config)
    report = processor.run()

    report_path = args.report or os.path.join(args.output_dir, "report.json")
    processor.save_report(report, report_path)

    print(f"\nBatch complete: {report.successes} succeeded, {report.failures} failed "
          f"(total: {report.total_files}) in {report.total_time:.1f}s")
    print(f"Report saved to: {report_path}")


def _parse_source(source_str: str):
    try:
        return int(source_str)
    except ValueError:
        return source_str


if __name__ == "__main__":
    main()
