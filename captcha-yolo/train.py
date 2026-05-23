import argparse
import os

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for captcha detection.")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model or checkpoint path.")
    parser.add_argument("--data", default="dataset.yaml", help="Dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=352, help="Training image size.")
    parser.add_argument("--batch", type=int, default=48, help="Batch size.")
    parser.add_argument("--device", default="mps", help="Training device, e.g. mps, 0, cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="captcha_train", help="Run name.")
    parser.add_argument("--cache", action="store_true", help="Enable dataset cache.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=args.project,
        name=args.name,
        pretrained=True,
        verbose=True,
    )

    print("Training finished.")
    print(f"Results directory: {args.project}/{args.name}")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
