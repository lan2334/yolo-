# yolo-
心血来潮写的yolo标注工具，个人认为还是非常好用的，手动标200张左右就可以进行自动标注然后人工复核一下即可
# captcha-yolo

这是一个用于验证码图片采集、目标框标注、数据集整理以及 YOLOv8 训练的小工具项目。

## 文件说明

- `fetch_captcha_bg.py`：从固定地址抓取验证码图片并保存到本地。
- `label_text.py`：基于 OpenCV 的手工标注工具，输出 YOLO 格式标签和预览图。
- `auto_label_yolov8.py`：使用训练好的 YOLO 模型自动标注图片，并把不确定样本分到复核目录。
- `prepare_dataset.py`：将已标注数据拆分为训练集和验证集。
- `train.py`：统一的 YOLOv8 训练入口脚本。

## 使用流程

1. 使用 `fetch_captcha_bg.py` 采集原始图片。
2. 使用 `label_text.py` 手工标注，或先用 `auto_label_yolov8.py` 自动标注后再复核。
3. 使用 `prepare_dataset.py` 整理出训练集和验证集。
4. 使用 `train.py` 开始训练模型。

## 环境依赖

- Python 3.10 及以上
- `ultralytics`
- `opencv-python`
- `requests`
- `beautifulsoup4`
- `numpy`

安装示例：

```bash
pip install ultralytics opencv-python requests beautifulsoup4 numpy
```

## 数据集格式

`train.py` 默认读取 YOLO 数据集配置文件，例如 `dataset.yaml`：

```yaml
path: /absolute/path/to/real_dataset
train: images/train
val: images/val

names:
  0: text
```

推荐目录结构如下：

```text
real_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## 训练方法

最基本的运行方式：

```bash
python train.py
```

常用参数示例：

```bash
python train.py \
  --data dataset.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 352 \
  --batch 48 \
  --device mps \
  --project runs/detect \
  --name captcha_train
```

训练完成后，最佳权重通常保存在：

```text
runs/detect/captcha_train/weights/best.pt
```

## 说明

- 目前部分脚本仍然使用了硬编码路径，换机器或换目录时需要先修改脚本内常量。
- `auto_label_yolov8.py` 依赖已经训练好的模型权重。
- 在 macOS 上，`--device mps --workers 0` 通常会更稳定。
