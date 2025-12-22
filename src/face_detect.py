# pip install retina-face
# we recommand tensorflow==2.15
from PIL import Image
import numpy as np
from retinaface import RetinaFace
import sys

def _clip(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def _pad(x1, y1, x2, y2, w, h, pad=0.12):
    bw, bh = (x2 - x1), (y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    return _clip(x1 - px, y1 - py, x2 + px, y2 + py, w, h)

def _refine_for_mask(x1, y1, x2, y2, w, h, style="generic"):
    # keep it simple & safe: small pad only
    # (too aggressive "tighten" can hurt real faces)
    if style == "anime":
        return _pad(x1, y1, x2, y2, w, h, pad=0.18)
    return _pad(x1, y1, x2, y2, w, h, pad=0.12)

def get_mask_coord(image_path):
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = img_rgb.shape

    # 1) Try RetinaFace (best for real faces)
    try:
        img_bgr = img_rgb[:, :, ::-1]
        resp = RetinaFace.detect_faces(img_bgr)
        if isinstance(resp, dict) and len(resp) > 0:
            best = max(resp.values(), key=lambda v: float(v.get("score", 0.0)))
            x1, y1, x2, y2 = best["facial_area"]
            x1, y1, x2, y2 = _refine_for_mask(x1, y1, x2, y2, w, h, style="generic")
            return y1, y2, x1, x2, h, w
    except Exception:
        pass

    # 2) Fallback: anime-face-detector (works for anime, sometimes ok for real)
    try:
        import cv2
        from anime_face_detector import create_detector
        detector = create_detector("yolov3")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        faces = detector(img_bgr)
        if faces:
            f = max(faces, key=lambda f: (f.face.w * f.face.h))
            x1, y1 = f.face.x, f.face.y
            x2, y2 = f.face.x + f.face.w, f.face.y + f.face.h
            x1, y1, x2, y2 = _refine_for_mask(x1, y1, x2, y2, w, h, style="anime")
            return y1, y2, x1, x2, h, w
    except Exception:
        print("anime-face-detector not available, skipping...")

    # 3) Last resort: center crop (never crash)
    # x1, x2 = int(0.22 * w), int(0.78 * w)
    # y1, y2 = int(0.10 * h), int(0.92 * h)
    # return y1, y2, x1, x2, h, w


if __name__ == "__main__":
  image_path = sys.argv[1]
  y,y2,x,x2,height,width = get_mask_coord(image_path)
  print (y,y2,x,x2,height,width)