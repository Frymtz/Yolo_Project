from ultralytics import YOLO

def train_model(data="datasets/components.yaml",
                model="yolov8s.pt",
                epochs=100,
                imgsz=640,
                batch=16,
                device=0):
    """
    Trains a YOLO model with the provided parameters.
    """
    yolo = YOLO(model)
    yolo.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cuda"
    )
