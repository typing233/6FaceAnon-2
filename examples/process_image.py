"""Example: process a single image with face anonymization."""
import cv2
import faceanon


def main():
    engine = faceanon.FaceAnonEngine(faceanon.EngineConfig(
        anonymizer=faceanon.AnonymizerConfig(
            method=faceanon.AnonymizationType.GAUSSIAN_BLUR,
            intensity=0.8,
        ),
    ))

    image = cv2.imread("test_face.jpg")
    if image is None:
        print("Error: cannot read test_face.jpg")
        return

    result = engine.process_image(image)
    print(f"Detected {len(result.detections)} face(s)")

    cv2.imwrite("output_blur.jpg", result.anonymized_frame)
    print("Saved: output_blur.jpg")

    # Switch to mosaic
    engine.anonymizer.config.method = faceanon.AnonymizationType.MOSAIC
    result = engine.process_image(image)
    cv2.imwrite("output_mosaic.jpg", result.anonymized_frame)
    print("Saved: output_mosaic.jpg")


if __name__ == "__main__":
    main()
