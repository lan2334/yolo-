import os
import shutil
import random

SRC_IMG_DIR = "real_captcha/dataset/images"
SRC_LBL_DIR = "real_captcha/dataset/labels_auto"

DST_BASE = "real_dataset"

TRAIN_RATIO = 0.8

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def ensure_dirs():
    for p in [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]:
        os.makedirs(os.path.join(DST_BASE, p), exist_ok=True)


def find_image(stem):
    for ext in IMAGE_EXTS:
        path = os.path.join(SRC_IMG_DIR, stem + ext)
        if os.path.exists(path):
            return path
    return None


def main():
    ensure_dirs()

    labels = [f for f in os.listdir(SRC_LBL_DIR) if f.endswith(".txt")]
    labels.sort()

    stems = [os.path.splitext(f)[0] for f in labels]

    random.seed(42)
    random.shuffle(stems)

    split = int(len(stems) * TRAIN_RATIO)

    train_stems = stems[:split]
    val_stems = stems[split:]

    def copy_subset(stems, subset):
        for stem in stems:

            img = find_image(stem)
            lbl = os.path.join(SRC_LBL_DIR, stem + ".txt")

            if img is None:
                print("找不到图片:", stem)
                continue

            dst_img = os.path.join(DST_BASE, "images", subset, os.path.basename(img))
            dst_lbl = os.path.join(DST_BASE, "labels", subset, stem + ".txt")

            shutil.copy2(img, dst_img)
            shutil.copy2(lbl, dst_lbl)

            print(f"{subset}: {stem}")

    copy_subset(train_stems, "train")
    copy_subset(val_stems, "val")

    print("\n完成")
    print("train:", len(train_stems))
    print("val:", len(val_stems))


if __name__ == "__main__":
    main()