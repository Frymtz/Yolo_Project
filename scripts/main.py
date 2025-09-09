import argparse
from train import train_model
from infer_image import run_inference_image
from infer_video import run_inference_video
from real_time import run_inference_realtime

def main():
    parser = argparse.ArgumentParser(description="YOLO Electronic Components")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "image", "video", "realtime"],
                        help="Choose execution mode: train, image, video, realtime")

    parser.add_argument("--source", type=str, default=None,
                        help="Path to image/video, webcam index (e.g.: 0, 1), or RTSP URL")

    parser.add_argument("--weights", type=str,
                        default="runs/detect/train/weights/best.pt",
                        help="Path to trained weights")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()

    elif args.mode == "image":
        if not args.source:
            raise ValueError("You must provide --source with the image path")
        run_inference_image(args.source, weights_path=args.weights)

    elif args.mode == "video":
        if not args.source:
            raise ValueError("You must provide --source with the video path")
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
