import av
import cv2
import numpy as np
# import logging
import queue
from pathlib import Path
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ダウンロードモジュール
from sample_utils.download import download_file

# ロガー
# logger = logging.getLogger(__name__)

# クラス名
CLASSES = [
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
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564) # モデル
download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353) # プロトコル

# Session-specific caching
cache_key = "object_detection_dnn"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
    st.session_state[cache_key] = net

# タイトル表示
col1, col2, col3 = st.columns(3)
with col1:
    # ロゴマーク
    # st.image("./images/forex_logo.png", width=70)
    st.image("./images/forex_logo.png", width=10)
with col2:
    # タイトル
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
def _annotate_image(image, detections):
    num = 0 # 人数
    # loop over the detections
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] # 精度

        if confidence > confidence_threshold: # 設定精度以上？
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            name = CLASSES[idx]

            if name == "person": # 人間？
                num += 1
                col = (0, 0, 255) # 赤
            else:
                col = (255, 0, 0) # 青

            # display the prediction
            label = f"{name}: {round(confidence * 100, 2)}%"
            # 枠描画
            cv2.rectangle(image, (startX, startY), (endX, endY), col, 2)
            # テキスト描画
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image,
                label,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                2,
            )
    return image, num

# キュー
result_queue: queue.Queue = (
    queue.Queue()
)  # TODO: A general-purpose shared state object may be more useful.

# コールバック処理
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward() # 物体検出
    # 検出結果描画
    annotated_image, num = _annotate_image(image, detections)

    # NOTE: This `recv` method is called in another thread,
    # so it must be thread-safe.
    # result_queue.put(result)  # TODO:
    result_queue.put(num)  # TODO:

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
