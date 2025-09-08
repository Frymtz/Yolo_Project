import argparse
from train import train_model
from infer_image import run_inference_image
from infer_video import run_inference_video
from real_time import run_inference_realtime  

def main():
    parser = argparse.ArgumentParser(description="YOLO Componentes Eletrônicos")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "image", "video", "realtime"],
                        help="Escolha o modo de execução: train, image, video, realtime")

    parser.add_argument("--source", type=str, default=None,
                        help="Caminho da imagem/vídeo ou índice da webcam (ex.: 0, 1) ou URL RTSP")

    parser.add_argument("--weights", type=str,
                        default="runs/detect/train/weights/best.pt",
                        help="Caminho para os pesos treinados")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()

    elif args.mode == "image":
        if not args.source:
            raise ValueError("Precisa passar --source com o caminho da imagem")
        run_inference_image(args.source, weights_path=args.weights)

    elif args.mode == "video":
        if not args.source:
            raise ValueError("Precisa passar --source com o caminho do vídeo")
        run_inference_video(source=args.source, weights_path=args.weights)

    elif args.mode == "realtime":
        src = 0 if args.source is None else args.source
        try:
            src = int(src)
        except ValueError:
            pass  
        run_inference_realtime(source=src, weights_path=args.weights)

if __name__ == "__main__":
    main()
