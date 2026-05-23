import os
import cv2

INPUT_DIR = "yolo_data/images"
OUTPUT_LABEL_DIR = "yolo_data/labels"
OUTPUT_PREVIEW_DIR = "yolo_data/previews"

CLASS_ID = 0  # 单类别 text

drawing = False
start_x, start_y = -1, -1
current_rect = None
boxes = []


def ensure_dirs():
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_PREVIEW_DIR, exist_ok=True)


def yolo_line(x, y, w, h, img_w, img_h):
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    ww = w / img_w
    hh = h / img_h
    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"


def clamp_box(x, y, w, h, img_w, img_h):
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return int(x), int(y), int(w), int(h)


def point_in_box(px, py, box):
    x, y, w, h = box
    return x <= px <= x + w and y <= py <= y + h


def find_box_at_point(px, py, boxes):
    candidates = []
    for i, box in enumerate(boxes):
        if point_in_box(px, py, box):
            x, y, w, h = box
            area = w * h
            candidates.append((area, i))

    if not candidates:
        return -1

    candidates.sort()
    return candidates[0][1]


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_rect, boxes

    img_h, img_w = param[:2]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        current_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x1, y1 = min(start_x, x), min(start_y, y)
        x2, y2 = max(start_x, x), max(start_y, y)
        w, h = x2 - x1, y2 - y1
        current_rect = clamp_box(x1, y1, w, h, img_w, img_h)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_x, x), min(start_y, y)
        x2, y2 = max(start_x, x), max(start_y, y)
        w, h = x2 - x1, y2 - y1

        if w >= 3 and h >= 3:
            rect = clamp_box(x1, y1, w, h, img_w, img_h)
            boxes.append(rect)

        current_rect = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        idx = find_box_at_point(x, y, boxes)
        if idx != -1:
            boxes.pop(idx)


def save_yolo_label(img_path, img, boxes):
    img_h, img_w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]

    txt_path = os.path.join(OUTPUT_LABEL_DIR, base + ".txt")
    preview_path = os.path.join(OUTPUT_PREVIEW_DIR, base + ".jpg")

    lines = [yolo_line(x, y, w, h, img_w, img_h) for (x, y, w, h) in boxes]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    preview = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(preview_path, preview)


def load_existing_label(img_path, img_shape):
    img_h, img_w = img_shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(OUTPUT_LABEL_DIR, base + ".txt")

    loaded_boxes = []
    if not os.path.exists(txt_path):
        return loaded_boxes

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        _, xc, yc, ww, hh = map(float, parts)

        bw = int(round(ww * img_w))
        bh = int(round(hh * img_h))
        x = int(round(xc * img_w - bw / 2))
        y = int(round(yc * img_h - bh / 2))

        x, y, bw, bh = clamp_box(x, y, bw, bh, img_w, img_h)
        loaded_boxes.append((x, y, bw, bh))

    return loaded_boxes


def draw_canvas(img, boxes, current_rect):
    canvas = img.copy()

    for x, y, w, h in boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if current_rect is not None:
        x, y, w, h = current_rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return canvas


def annotate_one(img_path):
    global boxes, current_rect, drawing

    img = cv2.imread(img_path)
    if img is None:
        print(f"[跳过] 无法读取: {img_path}")
        return "next"

    boxes = load_existing_label(img_path, img.shape)
    current_rect = None
    drawing = False

    win_name = "Manual YOLO Annotator"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback, img.shape)

    while True:
        canvas = draw_canvas(img, boxes, current_rect)
        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("u"):
            if boxes:
                boxes.pop()

        elif key == ord("c"):
            boxes = []

        elif key == ord("s"):
            save_yolo_label(img_path, img, boxes)
            print(f"[保存] {os.path.basename(img_path)} -> {len(boxes)} 个框")
            return "next"

        elif key == ord("n"):
            print(f"[跳过] {os.path.basename(img_path)}")
            return "next"

        elif key == ord("r"):
            print(f"[上一张] {os.path.basename(img_path)}")
            return "prev"

        elif key == ord("q"):
            return "quit"


def main():
    ensure_dirs()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f.lower())[1] in exts
    ]
    files.sort()

    if not files:
        print(f"输入目录没有图片: {INPUT_DIR}")
        return

    idx = 0
    while 0 <= idx < len(files):
        img_path = files[idx]
        action = annotate_one(img_path)

        if action == "quit":
            break
        elif action == "prev":
            idx = max(0, idx - 1)
        else:  # next
            idx += 1

    cv2.destroyAllWindows()
    print("结束。")
    print(f"标签目录: {OUTPUT_LABEL_DIR}")
    print(f"预览目录: {OUTPUT_PREVIEW_DIR}")


if __name__ == "__main__":
    main()