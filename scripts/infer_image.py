from ultralytics import YOLO
import cv2

CLASS_NAMES = ["resistor", "capacitor", "transistor"]

def run_inference_image(image_path,
                        weights_path="runs/detect/train/weights/best.pt",
                        conf=0.35,
                        iou=0.5):
    """
    Runs inference on an image and displays per-class counts.
    """
    model = YOLO(weights_path)
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)

    # annotated image with boxes
    annotated = results[0].plot()

    # per-class counts
    counts = {name: 0 for name in CLASS_NAMES}
    for cls_id in results[0].boxes.cls.cpu().numpy().astype(int):
        counts[CLASS_NAMES[cls_id]] += 1

    print("Counts:", counts)

    # write counts on the image
    y_offset = 30
    for cls, count in counts.items():
        text = f"{cls}: {count}"
        cv2.putText(annotated, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow("Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
