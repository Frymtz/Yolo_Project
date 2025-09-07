from ultralytics import YOLO
import cv2

CLASS_NAMES = ["resistor", "capacitor", "transistor"]

def run_inference_image(image_path,
                        weights_path="runs/detect/train/weights/best.pt",
                        conf=0.35,
                        iou=0.5):
    """
    Faz inferência em uma imagem e mostra contagem por classe.
    """
    model = YOLO(weights_path)
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)

    # anotado com boxes
    annotated = results[0].plot()

    # contagem por classe
    counts = {name: 0 for name in CLASS_NAMES}
    for cls_id in results[0].boxes.cls.cpu().numpy().astype(int):
        counts[CLASS_NAMES[cls_id]] += 1

    print("Contagens:", counts)

    cv2.imshow("Detecção", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
