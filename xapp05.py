import av
import cv2
import numpy as np
# import logging
import queue
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import sys

# ダウンロードモジュール
from sample_utils.download import download_file

# クラス名（英語）
CLASSES_E = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# クラス名（日本語）
CLASSES_J = [
    "背景",
    "飛行機",
    "自転車",
    "鳥",
    "船",
    "ボトル",
    "バス",
    "車",
    "猫",
    "椅子",
    "牛",
    "テーブル",
    "犬",
    "馬",
    "バイク",
    "人間",
    "植木",
    "羊",
    "ソファー",
    "列車",
    "モニター",
]

# パス名
HERE = Path(__file__).parent
ROOT = HERE.parent

MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

# 検出精度初期値
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# 色
color_white = (255, 255, 255) # 白
color_red = (255, 0, 0) # 赤
color_blue = (0, 0, 255) # 青

# フォント
# Windows
# font_name_win = "C:\\Windows\\Fonts\\msgothic.ttc" # MSゴシック
# font_name_win = "C:\\Windows\\Fonts\\msmincho.ttc" # MS明朝
# font_name_win = "C:\\Windows\\Fonts\\meiryo.ttc" # MEIRYO
# font_name_win = "C:\\Windows\\Fonts\\meiryob.ttc" # MEIRYO（太字）
font_name_win = "msgothic.ttc" # MSゴシック
# font_name_win = "meiryo.ttc" # MEIRYO

font_name_mac = "ヒラギノ丸ゴ ProN W4.ttc" # Mac
# font_name_lnx = "/usr/share/fonts/OTF/TakaoPMincho.ttf" # Linux

# streamlit Cloud（Linux）
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# font_name_lnx = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_name_lnx = "DejaVuSerif.ttf"
# font_name_lnx = "DejaVuSerif-Bold.ttf"
# font_name_lnx = "DejaVuSansMono.ttf"
# font_name_lnx = "DejaVuSansMono-Bold.ttf"
# font_name_lnx = "DejaVuSans.ttf"
# font_name_lnx = "DejaVuSans-Bold.ttf"

# ラベル
label_font_size = 16 # ラベルフォントサイズ

# ロゴマーク
logo_path = "./images/forex_logo_a.png" # ロゴパス名
logo_rate = 0.15 # 倍率
logo_margin = 5 # ロゴ表示マージン

# 合計人数
total_num = 0

# キュー
result_queue: queue.Queue = (
    queue.Queue()
)  # TODO: A general-purpose shared state object may be more useful.

# ロガー
# logger = logging.getLogger(__name__)

# ファイルダウンロード
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564) # 学習モデル
download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353) # プロトコル

# Session-specific caching
cache_key = "object_detection_dnn"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
    st.session_state[cache_key] = net

# 合計人数
total_key = "total_number"
if total_key in st.session_state:
    total_num = st.session_state[total_key]
else:
    st.session_state[total_key] = total_num

# フォント
if sys.platform == "win32": # Windows
    font_name = font_name_win
if sys.platform == "darwin": # Mac
    font_name = font_name_mac
if sys.platform in ("linux", "linux2"): # Linux
    font_name = font_name_lnx

# ラベルフォント
labelfont = ImageFont.truetype(font_name, label_font_size)

# ロゴマーク読み込み
logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
logo_image = cv2.resize(logo_image, dsize=None, fx=logo_rate, fy=logo_rate)
logo_height, logo_width = logo_image.shape[:2]
# PIL形式に変換
logo_image = cv2.cvtColor(logo_image, cv2.COLOR_BGRA2RGBA)
logo_pil = Image.fromarray(logo_image)

# タイトル表示
st.subheader("みまもりくん")

# 状態表示
labels_placeholder = st.empty()
# 映像表示
streaming_placeholder = st.empty()
# スライダー表示
confidence_threshold = st.slider(
    "精度", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

# 物体抽出
def extractionObject(cimage, detections):
    num = 0 # 人数
    (h, w) = cimage.shape[:2]
    objects = [] # 物体配列

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] # 精度

        if confidence > confidence_threshold: # 設定精度以上？
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            idx = int(detections[0, 0, i, 1])
            ename = CLASSES_E[idx]
            jname = CLASSES_J[idx]

            if ename == "person": # 人間？
                num += 1
                col = color_red
            else:
                col = color_blue

            # 物体追加
            objects.append((startX, startY, endX, endY, ename, jname, col, confidence))

    return objects, num

# 結果描画
def drawingResult(src, objects):
    # （参考）
    # https://note.com/npaka/n/nddb33be1b782

    # 背景をPIL形式に変換
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)
    draw = ImageDraw.Draw(pil_src)

    # 物体取得
    for (startX, startY, endX, endY, ename, jname, col, confidence) in objects:
        # ラベル
        if sys.platform in ("linux", "linux2"): # Linux？（日本語フォントなし）
            label = ename # 英語
        else:
            label = jname # 日本語

        # ラベル描画
        y = startY - (label_font_size+1) if startY - (label_font_size+1) > (label_font_size+1) else startY + (label_font_size+1)
        draw.text(xy = (startX, y), text = label, fill = col, font = labelfont)

        # 枠描画
        draw.rectangle([(startX, startY), (endX, endY)], outline=col, width=2)

    # ロゴマークを合成
    src_height, src_width = src.shape[:2]
    logo_pos = (src_width - logo_width - logo_margin, logo_margin)
    pil_src.paste(logo_pil, logo_pos, logo_pil)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(pil_src), cv2.COLOR_RGB2BGR)

# コールバック処理
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    # 画像変換
    cimage = frame.to_ndarray(format="bgr24")
    blob = cv2.dnn.blobFromImage(cv2.resize(cimage, (300, 300)), 0.007843, (300, 300), 127.5)

    # 物体検出
    net.setInput(blob)
    detections = net.forward()

    # 物体抽出
    objects, num = extractionObject(cimage, detections)

    # 結果描画
    cimage = drawingResult(cimage, objects)

    # NOTE: This `recv` method is called in another thread,
    # so it must be thread-safe.
    # result_queue.put(result)  # TODO:
    result_queue.put(num)  # TODO:

    return av.VideoFrame.from_ndarray(cimage, format="bgr24")

# 映像表示
with streaming_placeholder.container():
    # WEBカメラ
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        translations={
            "start": "開始",
            "stop": "停止",
            "select_device": "カメラ切替",
            "media_api_not_available": "Media APIが利用できない環境です",
            "device_ask_permission": "メディアデバイスへのアクセスを許可してください",
            "device_not_available": "メディアデバイスを利用できません",
            "device_access_denied": "メディアデバイスへのアクセスが拒否されました",
        },
    )

if webrtc_ctx.state.playing: # 映像配信中？
    labels = labels_placeholder
    while True: # 繰り返し
        try:
            # キューの取得
            result = result_queue.get(timeout=1.0) # 人数取得
        except queue.Empty:
            result = 0

        if result > 0: # 人がいる？
            labels.error('人を発見！' + str(total_num))

            # 合計人数更新
            total_num += 1
            total_num = (total_num % 10000) + 1
            st.session_state[total_key] = total_num

        else: # 人がいない
            labels.info('安全です' + str(total_num))

            # # 合計人数リセット
            # total_num = 0
            # st.session_state[total_key] = total_num
