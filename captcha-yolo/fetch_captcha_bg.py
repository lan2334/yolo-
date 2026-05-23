import requests
import time
import cv2
import numpy as np
import os

URL = ""

SAVE_DIR = "easy_real_captcha/dataset/images"

os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 配置 =====
CURRENT_MAX_INDEX = 0   # 当前最后一个图片编号
NEED_COUNT = 50         # 需要新增多少张
SLEEP_TIME = 0.1          # 每次请求间隔
# =================


def fetch_captcha():

    ts = int(time.time() * 1000)
    url = f"{URL}?t={ts}"

    try:
        r = requests.get(url, verify=False, timeout=10)
    except Exception as e:
        print("请求异常:", e)
        return None

    if r.status_code != 200:
        print("请求失败:", r.status_code)
        return None

    img_array = np.frombuffer(r.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img


def save_bg(index):

    img = fetch_captcha()

    if img is None:
        return False

    filename = f"{SAVE_DIR}/img{index}.jpg"

    cv2.imwrite(filename, img)

    print("保存:", filename)

    return True


if __name__ == "__main__":

    start_index = CURRENT_MAX_INDEX + 1
    end_index = CURRENT_MAX_INDEX + NEED_COUNT

    print("开始抓取")
    print("起始编号:", start_index)
    print("结束编号:", end_index)

    for i in range(start_index, end_index + 1):

        success = save_bg(i)

        if not success:
            print("抓取失败，重试下一张")

        time.sleep(SLEEP_TIME)

    print("完成")