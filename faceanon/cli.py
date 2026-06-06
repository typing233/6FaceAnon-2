import argparse
import sys
import time
import cv2

from .config import EngineConfig, DetectorConfig, TrackerConfig, AnonymizerConfig, AnonymizationType
from .engine import FaceAnonEngine


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
    parser.add_argument("--detect-every-n", type=int, default=1)

    subparsers = parser.add_subparsers(dest="command")

    img_parser = subparsers.add_parser("image", help="Process a single image")
    img_parser.add_argument("-i", "--input", required=True)
    img_parser.add_argument("-o", "--output", required=True)

    vid_parser = subparsers.add_parser("video", help="Process a video file")
    vid_parser.add_argument("-i", "--input", required=True)
    vid_parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = EngineConfig(
        detector=DetectorConfig(
            model_path=args.model,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
        ),
        tracker=TrackerConfig(),
        anonymizer=AnonymizerConfig(
            method=AnonymizationType(args.method),
            intensity=args.intensity,
            expand_ratio=args.expand_ratio,
        ),
        detect_every_n=args.detect_every_n,
    )

    engine = FaceAnonEngine(config)

    if args.command == "image":
        _process_image(engine, args.input, args.output)
    elif args.command == "video":
        _process_video(engine, args.input, args.output)


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


if __name__ == "__main__":
    main()
