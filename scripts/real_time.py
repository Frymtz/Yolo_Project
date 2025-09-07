from ultralytics import YOLO
import cv2
from collections import Counter

CLASS_NAMES = ["resistor", "capacitor", "transistor"]

def run_inference_realtime(source=0,
                           weights_path="runs/detect/train/weights/best.pt",
                           conf=0.35,
                           iou=0.5,
                           intervalo=4):
    """
    Inferência em tempo real de câmera (USB, IP, etc.)
    Mostra bounding boxes + contagem de cada classe no frame.
    """
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {source}")

    frame_atual = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_atual % intervalo != 0:
            frame_atual += 1
            continue

        results = model(frame, conf=conf, iou=iou, verbose=False)

        # Contagem das classes no frame atual
        counts = Counter()
        for result in results:
            boxes = result.boxes
            for cls in boxes.cls.tolist():
                cls_name = CLASS_NAMES[int(cls)]
                counts[cls_name] += 1

        frame_atual += 1

        # Frame anotado com bounding boxes
        annotated_frame = results[0].plot()

        # Escrever a contagem no canto superior esquerdo
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

        cv2.imshow("Detecção em tempo real", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
