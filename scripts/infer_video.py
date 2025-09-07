from ultralytics import YOLO
import cv2
import time

CLASS_NAMES = ["resistor", "capacitor", "transistor"]

def run_inference_video(source=0,
                        weights_path="runs/detect/train/weights/best.pt",
                        conf=0.35,
                        iou=0.5):
    """
    Faz inferência em vídeo/webcam com contagem por frame.
    """
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {source}")

    t0 = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        annotated = results[0].plot()

        counts = {name: 0 for name in CLASS_NAMES}
        for cls_id in results[0].boxes.cls.cpu().numpy().astype(int):
            counts[CLASS_NAMES[cls_id]] += 1

        frames += 1
        fps = frames / (time.time() - t0 + 1e-6)

        cv2.putText(annotated,
                    f"Resistores: {counts['resistor']} | Capacitores: {counts['capacitor']} | Transistores: {counts['transistor']} | FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        cv2.imshow("YOLO - Vídeo", annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
