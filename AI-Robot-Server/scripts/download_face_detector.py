"""
Скачивание модели детектора лиц OpenCV DNN (SSD, res10_300x300).

Это более точный детектор лиц, чем встроенный Haar Cascade — лучше
работает при повороте головы и плохом освещении. Если файлы не скачаны,
vision.py автоматически откатывается на Haar Cascade (ничего не сломается).

Использование:
    python scripts/download_face_detector.py
"""
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEST_DIR = BASE_DIR / "models" / "face_detector"

FILES = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
}


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        dest_path = DEST_DIR / filename
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"✅ Уже скачано: {dest_path}")
            continue
        print(f"⬇️  Скачиваю {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"✅ Готово: {dest_path} ({dest_path.stat().st_size} байт)")
        except Exception as e:
            print(f"❌ Не удалось скачать {filename}: {e}")
            print("   Скачай вручную по ссылке выше и положи в models/face_detector/")

    ok = all((DEST_DIR / f).exists() for f in FILES)
    if ok:
        print("\n🎉 Детектор лиц OpenCV DNN готов к использованию.")
    else:
        print("\n⚠️  Не все файлы скачаны — vision.py будет использовать Haar Cascade как запасной вариант.")


if __name__ == "__main__":
    main()
