"""Example: process a video file with face anonymization and tracking."""
import time
import faceanon


def main():
    config = faceanon.EngineConfig(
        anonymizer=faceanon.AnonymizerConfig(
            method=faceanon.AnonymizationType.MOSAIC,
            intensity=0.7,
        ),
        detect_every_n=2,  # Detect every other frame for speed
    )
    engine = faceanon.FaceAnonEngine(config)

    input_path = "input.mp4"
    output_path = "output_anonymized.mp4"

    start = time.time()

    def progress(idx, total, result):
        if idx % 50 == 0:
            elapsed = time.time() - start
            fps = (idx + 1) / max(elapsed, 0.001)
            print(f"  Frame {idx + 1}/{total} | {fps:.1f} fps | {len(result.tracks)} faces")

    total = engine.process_video(input_path, output_path, callback=progress)

    elapsed = time.time() - start
    print(f"\nDone: {total} frames in {elapsed:.1f}s ({total / elapsed:.1f} fps)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
