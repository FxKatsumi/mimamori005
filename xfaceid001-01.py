import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1

# 定数定義
app_title = "にんしきくん" # アプリ名
face_data_path = "FaceData" # 顔データパス名
cascade_path = "haarcascades/haarcascade_frontalface_alt.xml" # カスケードファイル

# 画像のサイズの指定
# image_width = 500
# image_height = 300
image_width = 640
image_height = 360

expand_rate = 1.5 # 拡大率
face_threshold = 0.7 # 顔閾値

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

# ロゴマーク
logo_path = "forex_logo.png" # ロゴパス名
logo_rate = 0.11 # 倍率
logo_image = None # ロゴイメージ

# 顔データID変換（イメージ）
def GetFaceIDImage(img):
    try:
        # 顔データを160×160に切り抜き
        img_crop = mtcnn(img)
        # 切り抜いた顔データを512個の数字に
        img_emb = resnet(img_crop.unsqueeze(0))
        # 512個の数字にしたものはpytorchのtensorという型なので、numpyの型に変換
        return img_emb.squeeze().to('cpu').detach().numpy().copy()

    except Exception as e:
        #print('GetFaceIDImage:', e)
        return None

# 顔データID変換（ファイル名）
def GetFaceIDFile(fname):
    try:
        # イメージファイルパス名
        image_path = os.path.join(face_data_path, fname)
        # 画像データ取得
        img = Image.open(image_path) 
        # 顔データID変換（イメージ）
        return GetFaceIDImage(img)

    except Exception as e:
        print('GetFaceIDFile:', e)

# 顔データクラス
class FaceDataClass:
    # コンストラクタ 
    def __init__(self, fname):
        try:
            # 名前（ファイル名）設定
            self.name = os.path.splitext(os.path.basename(fname))[0]
            # 顔データID変換
            self.id = GetFaceIDFile(fname)

        except Exception as e:
            print('FaceDataClass(__init__):', e)

# 顔データ配列
FaceDatas = []

# 顔データ読み込み
def FaceDataRead():
    try:
        # 顔データフォルダー検索
        folderfiles = os.listdir(face_data_path)
        # 顔データファイル名のみ取得
        files = [f for f in folderfiles if os.path.isfile(os.path.join(face_data_path, f))]

        # 顔データ取得
        for fname in files:
            # 顔データ追加
            FaceDatas.append(FaceDataClass(fname))

    except Exception as e:
        print('FaceDataRead:', e)

# コサイン類似度算出
def cos_similarity(p1, p2):
    try:
        return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

    except Exception as e:
        print('cos_similarity:', e)

# 型変換（PIL→CV）
def pil2cv(imgPIL):
    try:
        imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
        return imgCV_BGR

    except Exception as e:
        print('pil2cv:', e)

# 型変換（CV→PIL）
def cv2pil(imgCV):
    try:
        imgCV_RGB = imgCV[:, :, ::-1]
        imgPIL = Image.fromarray(imgCV_RGB)
        return imgPIL

    except Exception as e:
        print('cv2pil:', e)

# ロゴイメージ取得
def loadLogo():
    try:
        logo_image = Image.open(logo_path)
        logo_w, logo_h  = logo_image.size
        logo_scale = logo_rate
        return logo_image.resize((int(logo_w * logo_scale), int(logo_h * logo_scale))) # リサイズ

    except Exception as e:
        print('loadLogo:', e)

# ロゴイメージ描画
def drawLogo(imgPIL):
    try:
        base_w, base_h  = imgPIL.size # ウインドウサイズ
        logo_w, logo_h  = logo_image.size # ロゴサイズ
        # ロゴ描画
        imgPIL.paste(logo_image, (base_w - logo_w - 10, 0), logo_image)

    except Exception as e:
        print('drawLogo:', e)


# テキスト出力
def cv2_putText(img, labels, labelfont):
    try:
        # イメージ変換
        imgPIL = cv2pil(img)
        draw = ImageDraw.Draw(imgPIL)

        # 背景色
        b, g, r = label_bg_color
        rectcolor = (r, g, b) # 背景色

        # ラベル取得
        for (x, y, facename) in labels:
            facename = ' ' + facename # テキスト補正

            # 描画サイズ取得
            x1, y1, x2, y2 = draw.textbbox((x, y), facename, font=labelfont, anchor='md')
            w = x2 - x1
            h = y2 - y1

            label_width = int(h / 2) + 1 # ラベル幅（高さ）

            # 枠描画
            draw.rectangle([(x, y-h), (x+w, y)], \
                        outline=rectcolor, width=label_width)

            # テキスト描画
            draw.text(xy = (x,y-h), text = facename, fill = label_text_color, font = labelfont)

        # タイトル描画
        draw.text(xy = (5, 5), text = app_title, fill = (0, 0, 255), font = labelfont)
        # ロゴ描画
        drawLogo(imgPIL)

        # イメージ変換
        imgCV = pil2cv(imgPIL)

        return imgCV

    except Exception as e:
        print('cv2_putText:', e)

# クリップ拡大
def clipExpand(x,y,w,h):
    try:
        w2 = int(w * expand_rate)
        x2 = int(x - (w2 - w) / 2)
        h2 = int(h * expand_rate)
        y2 = int(y - (h2 - h) / 2)

        # 補正
        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0
        if x2 + w2 > image_width:
            w2 = image_width - x2
        if y2 + h2 > image_height:
            h2 = image_height - y2

        return (x2, y2, w2, h2)

    except Exception as e:
        print('clipExpand:', e)

# 顔認識
def faceRecognition(fid):
    try:
        facename = '???' # 名前
        score = 0 # スコア

        # 顔データ検索
        for fd in FaceDatas:
            # 類似度取得
            res = cos_similarity(fid, fd.id)

            if res >= face_threshold: # 一致？
                if res > score: # より類似？
                    score = res
                    facename = fd.name

        return facename

    except Exception as e:
        print('faceRecognition:', e)


# メイン処理
try:
    # 顔検出のAI
    # image_size: 顔を検出して切り取るサイズ
    # margin: 顔まわりの余白
    mtcnn = MTCNN(image_size=160, margin=10)

    # 切り取った顔を512個の数字にするAI
    # 1回目の実行では学習済みのモデルをダウンロードしますので、少し時間かかります。
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # 顔データ読み込み
    FaceDataRead()

    #Webカメラから入力を開始
    cap = cv2.VideoCapture(0)

    #顔の検出器を作成
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # ロゴイメージ取得
    logo_image = loadLogo()

    # ラベルフォント
    labelfont = ImageFont.truetype(font_name, label_font_size)

    #カメラから連続して映像を取得
    while True:
        #カメラの画像を読み込む
        ret, frame = cap.read()

        #画像のサイズを変更（リサイズ）
        frame = cv2.resize(frame, (image_width, image_height))
        #グレイスケールに変換
        gray_flame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #顔認証を実行(minNeighboreは信頼性のパラメータ)
        #face_list = face_cascade.detectMultiScale(gray_flame, minNeighbors=20)
        face_list = face_cascade.detectMultiScale(gray_flame, minNeighbors=3)

        # ラベル配列
        Labels = []

        # 検出された顔の処理
        for (x,y,w,h) in face_list:
            #赤色の枠で囲む
            cv2.rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), label_bg_color, 1)

            # クリップ拡大
            (x2, y2, w2, h2) = clipExpand(x,y,w,h)
            #顔のみ切り取る
            trim_face = frame[y2:y2+h2, x2:x2+w2]
            # 顔データID変換（イメージ）
            fid = GetFaceIDImage(trim_face)

            if fid is not None: # 顔検出あり？
                # 顔認識
                facename = faceRecognition(fid)
                # ラベル追加
                Labels.append((x, y, facename))

        # ラベル一括描画
        frame = cv2_putText(img = frame,
                            labels = Labels,
                            labelfont = labelfont)
        
        # ウィンドウに画像を出力
        cv2.imshow("FOREX", frame)

        # 1msec確認
        exit_wind = cv2.waitKey(1)
        # Enterキーが押されたら終了
        if exit_wind == 13: break
    
    #------------------------------------------------
        
    #カメラを終了
    cap.release()
    
    #ウィンドウを閉じる
    cv2.destroyAllWindows()

except Exception as e:
    print('main:', e)
