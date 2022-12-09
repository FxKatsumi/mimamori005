import av
import cv2
import numpy as np
# import logging
import queue
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ダウンロードモジュール
from sample_utils.download import download_file

# ロガー
# logger = logging.getLogger(__name__)

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

# ロゴマーク
logo_path = "./images/forex_logo_a.png" # ロゴパス名
# logo_rate = 0.12 # 倍率
logo_rate = 0.15 # 倍率

# ロゴマーク読み込み
logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
logo_image = cv2.resize(logo_image, dsize=None, fx=logo_rate, fy=logo_rate)
logo_height, logo_width = logo_image.shape[:2]
logo_margin = 5 # ロゴ表示マージン

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
    "精度調節", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

# 検出結果描画
def _annotate_image(cimage, detections):
    num = 0 # 人数
    # loop over the detections
    (h, w) = cimage.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] # 精度

        if confidence > confidence_threshold: # 設定精度以上？
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            name = CLASSES_E[idx]

            if name == "person": # 人間？
                num += 1
                col = (0, 0, 255) # 赤
            else:
                col = (255, 0, 0) # 青

            # display the prediction
            label = f"{name}: {round(confidence * 100, 2)}%"
            # 枠描画
            cv2.rectangle(cimage, (startX, startY), (endX, endY), col, 2)
            # テキスト描画
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                cimage,
                label,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                2,
            )
    return cimage, num

# キュー
result_queue: queue.Queue = (
    queue.Queue()
)  # TODO: A general-purpose shared state object may be more useful.

# 画像のオーバーレイ
def overlayImage(src, pil_logo, location):
    # （参考）
    # https://note.com/npaka/n/nddb33be1b782

    # 背景をPIL形式に変換
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_src = Image.fromarray(src)


    # 色
    color_white = (255, 255, 255) # 白
    color_red = (0,0,255) # 赤

    # フォント
    font_name = "C:\\Windows\\Fonts\\msgothic.ttc" # MSゴシック
    # font_name = "C:\\Windows\\Fonts\\msmincho.ttc" # MS明朝
    # font_name = "C:\\Windows\\Fonts\\meiryo.ttc" # MEIRYO
    # font_name = "C:\\Windows\\Fonts\\meiryob.ttc" # MEIRYO（太字）

    # ラベル
    label_font_size = 14 # ラベルフォントサイズ
    label_bg_color = color_red # ラベル背景色
    label_text_color = color_white # ラベル文字色

    draw = ImageDraw.Draw(pil_src)
    label_text_color = color_white # ラベル文字色

    # ラベルフォント
    labelfont = ImageFont.truetype(font_name, label_font_size)

    # テキスト描画
    draw.text(xy = (100,100), text = "フォレックス", fill = label_bg_color, font = labelfont)


    # 画像を合成
    pil_src.paste(pil_logo, location, pil_logo)

    # OpenCV形式に変換
    return cv2.cvtColor(np.asarray(pil_src), cv2.COLOR_RGB2BGR)

# コールバック処理
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    cimage = frame.to_ndarray(format="bgr24")
    blob = cv2.dnn.blobFromImage(
        cv2.resize(cimage, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward() # 物体検出
    # 検出結果描画
    annotated_image, num = _annotate_image(cimage, detections)

    # NOTE: This `recv` method is called in another thread,
    # so it must be thread-safe.
    # result_queue.put(result)  # TODO:
    result_queue.put(num)  # TODO:

    # 画像のオーバーレイ（ロゴマーク表示）
    src_height, src_width = annotated_image.shape[:2]
    logo_pos = (src_width - logo_width - logo_margin, logo_margin)
    annotated_image = overlayImage(annotated_image, logo_pil, logo_pos)

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

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
            labels.error('人を発見！')
        else: # 人がいない
            labels.info('安全です')
