from ultralytics import YOLO
import cv2
from collections import Counter

CLASS_NAMES = ["resistor", "capacitor", "transistor"]

def run_inference_realtime(source=0,
                           weights_path="runs/detect/train/weights/best.pt",
                           conf=0.35,
                           iou=0.5,
                           interval=4):
    """
    Real-time inference from camera (USB, IP, etc.)
    Shows bounding boxes + count of each class in the frame.
    """
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % interval != 0:
            current_frame += 1
            continue

        results = model(frame, conf=conf, iou=iou, verbose=False)

        # Count classes in the current frame
        counts = Counter()
        for result in results:
            boxes = result.boxes
            for cls in boxes.cls.tolist():
                cls_name = CLASS_NAMES[int(cls)]
                counts[cls_name] += 1

        current_frame += 1

        # Frame annotated with bounding boxes
        annotated_frame = results[0].plot()

        # Write the count in the top left corner
        y_offset = 30
        for comp, qtd in counts.items():
            cv2.putText(
                annotated_frame,
                f"{comp}: {qtd}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            y_offset += 30

        cv2.imshow("Real-time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
