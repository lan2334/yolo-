import os
import cv2
import shutil
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/text_detect_round2/weights/best.pt"

INPUT_DIR = "easy_real_captcha/dataset/images"

OUTPUT_LABEL_DIR = "easy_real_captcha/dataset/labels_auto"
OUTPUT_PREVIEW_DIR = "easy_real_captcha/dataset/previews_auto"

REVIEW_BASE_DIR = "easy_real_captcha/dataset/review_not_9"
REVIEW_IMAGE_DIR = os.path.join(REVIEW_BASE_DIR, "images/train")
REVIEW_LABEL_DIR = os.path.join(REVIEW_BASE_DIR, "labels/train")
REVIEW_PREVIEW_DIR = os.path.join(REVIEW_BASE_DIR, "previews/train")

CLASS_ID = 0
CONF_THRES = 0.35
EXPECTED_BOXES = 6

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs():
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PREVIEW_DIR, exist_ok=True)

    os.makedirs(REVIEW_IMAGE_DIR, exist_ok=True)
    os.makedirs(REVIEW_LABEL_DIR, exist_ok=True)
    os.makedirs(REVIEW_PREVIEW_DIR, exist_ok=True)


def yolo_line(xc, yc, w, h):
    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def list_images(input_dir):
    files = []
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(os.path.join(input_dir, f))
    files.sort()
    return files


def save_label(txt_path, boxes_xywhn):
    with open(txt_path, "w", encoding="utf-8") as f:
        for box in boxes_xywhn:
            x, y, w, h = box
            f.write(yolo_line(float(x), float(y), float(w), float(h)) + "\n")


def draw_preview(img_path, boxes_xyxy):
    img = cv2.imread(img_path)
    if img is None:
        return None

    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img


def save_review(img_path, stem, img, boxes_xywhn):

    img_dst = os.path.join(REVIEW_IMAGE_DIR, os.path.basename(img_path))
    lbl_dst = os.path.join(REVIEW_LABEL_DIR, stem + ".txt")
    pre_dst = os.path.join(REVIEW_PREVIEW_DIR, stem + ".jpg")

    shutil.copy2(img_path, img_dst)

    save_label(lbl_dst, boxes_xywhn)

    if img is not None:
        cv2.imwrite(pre_dst, img)


def main():
    ensure_dirs()

    if not os.path.exists(MODEL_PATH):
        print("模型不存在:", MODEL_PATH)
        return

    model = YOLO(MODEL_PATH)

    images = list_images(INPUT_DIR)

    total = len(images)
    good = 0
    review = 0

    for i, img_path in enumerate(images, start=1):

        stem = os.path.splitext(os.path.basename(img_path))[0]

        results = model.predict(
            source=img_path,
            conf=CONF_THRES,
            save=False,
            verbose=False
        )

        if not results:
            print(f"[{i}/{total}] 无结果 -> review")
            review += 1
            continue

        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            print(f"[{i}/{total}] 无检测框 -> review")
            review += 1
            continue

        boxes_xywhn = r.boxes.xywhn.cpu().numpy()
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()

        count = len(boxes_xywhn)

        img = draw_preview(img_path, boxes_xyxy)

        if count == EXPECTED_BOXES:

            txt_path = os.path.join(OUTPUT_LABEL_DIR, stem + ".txt")
            preview_path = os.path.join(OUTPUT_PREVIEW_DIR, stem + ".jpg")

            save_label(txt_path, boxes_xywhn)

            if img is not None:
                cv2.imwrite(preview_path, img)

            good += 1
            print(f"[{i}/{total}] OK -> {count} boxes")

        else:

            save_review(img_path, stem, img, boxes_xywhn)

            review += 1
            print(f"[{i}/{total}] REVIEW -> {count} boxes")

    print("\n完成")
    print("可用训练数据:", good)
    print("待复核数据:", review)


if __name__ == "__main__":
    main()