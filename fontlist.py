import shutil
import sys
import os
import streamlit as st

from PIL import ImageDraw, Image, ImageFont

def font_installed_dirs() -> list[str]:
    """
    インストールされているフォントのディレクトリ取得
    """

    if sys.platform == "win32":
        d = os.environ.get("WINDIR")
        return [os.path.join(d, "fonts")]

    if sys.platform in ("linux", "linux2"):
        dirs = os.environ.get("XDG_DATA_DIRS", "")
        if not dirs:
            dirs = "/usr/share"
        return [os.path.join(d, "fonts") for d in dirs.split(":")]

    if sys.platform == "darwin":
        return ["/Library/Fonts",
                "/System/Library/Fonts",
                os.path.expanduser("~/Library/Fonts")]

    raise Exception(f"unsupported platform:{sys.platform}")

def font_path_list() -> list[str]:
    """
    インストールされているフォントのPATHリストを取得
    """

    # フォントがインストールされているディレクトリリスト取得
    dirs = font_installed_dirs()

    # os.walkを使って、ファイル一覧を取得する
    l: list[str] = []
    for d in dirs:
        for parent, _, filenames in os.walk(d):
            for name in filenames:
                l.append(os.path.join(parent, name))

    return l

def show_fonts():
    """
    フォント毎のサンプル画像を出力します。
    """
    # font_size = 28

    # # 出力ディレクトリの再作成
    # output_dir = os.path.join("out", "font_samples")
    # shutil.rmtree(output_dir)
    # os.mkdir(output_dir)

    for f in font_path_list():
        # font_name = os.path.basename(f)
        # image = Image.new('RGBA', (600, 200), 'white')
        # draw = ImageDraw.Draw(image)

        # try:
        #     font = ImageFont.truetype(f, font_size)
        #     draw.text((0, 0),
        #               f"{font_name}\nサンプルのテキストを表示\nfont_size:{font_size}",
        #               font=font,
        #               fill='black')
        # except OSError as err:
        #     # エラーがでた場合、スキップ
        #     print(f"failed to draw with font:{f} caused by {err}", file=sys.stderr)
        #     continue

        # # 画像保存
        # image.save(os.path.join(output_dir, font_name + ".png"), "PNG")

        # # パスをテキストで保存
        # with open(os.path.join(output_dir, font_name + ".txt"), "w") as fp:
        #     fp.write(f"{f}\n")
        #     fp.write(f"{font_name}\n")

        # print(f)
        st.write(f)

if __name__ == '__main__':
    show_fonts()
