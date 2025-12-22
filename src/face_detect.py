# pip install retina-face
# we recommand tensorflow==2.15
import sys
from PIL import Image
import numpy as np

# Optional: keep RetinaFace if you already use it (best for real faces)
try:
    from retinaface import RetinaFace
    HAS_RETINAFACE = True
except Exception:
    HAS_RETINAFACE = False

# Ultralytics YOLO
from ultralytics import YOLO

# Cache model so we don't reload every call
_YOLO_MODEL = None

def _get_yolo(device="cuda"):
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        # General model (will auto-download on first use)
        _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL

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

def _face_box_from_person_box(px1, py1, px2, py2, w, h):
    """
    Heuristic: face/head region is roughly upper part of person box.
    Works well when YOLO returns a person box (upper-body/full-body).
    """
    pw, ph = (px2 - px1), (py2 - py1)
    # take upper ~55% as head/face region, center horizontally
    x1 = px1 + 0.18 * pw
    x2 = px2 - 0.18 * pw
    y1 = py1 + 0.02 * ph
    y2 = py1 + 0.58 * ph
    return _clip(x1, y1, x2, y2, w, h)

def _cv_fallback_bbox_white_bg(img_rgb):
    """
    Extremely reliable for images like yours (frontal anime on plain/white background).
    Finds largest non-background component and returns its bbox.
    """
    import cv2

    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Invert threshold: keep "non-white" pixels
    # (tune 245->235 if your background isn't pure white)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None

    # Pick largest component (ignore label 0 background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    ww = stats[idx, cv2.CC_STAT_WIDTH]
    hh = stats[idx, cv2.CC_STAT_HEIGHT]

    x1, y1, x2, y2 = x, y, x + ww, y + hh

    # Refine to face-ish region (drop a bit of torso if present)
    # For head-only images, this barely changes; for half-body it helps.
    bh = y2 - y1
    y2 = y1 + int(0.92 * bh)

    return _pad(x1, y1, x2, y2, w, h, pad=0.06)

def get_mask_coord(image_path, device="cuda"):
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = img_rgb.shape

    print(f"Detecting face in image of size (h={h}, w={w})",  HAS_RETINAFACE)

    # 1) Real-face expert (RetinaFace)
    if HAS_RETINAFACE:
        try:
            img_bgr = img_rgb[:, :, ::-1]
            resp = RetinaFace.detect_faces(img_bgr)
            if isinstance(resp, dict) and len(resp) > 0:
                best = max(resp.values(), key=lambda v: float(v.get("score", 0.0)))
                x1, y1, x2, y2 = best["facial_area"]
                x1, y1, x2, y2 = _pad(x1, y1, x2, y2, w, h, pad=0.10)
                return y1, y2, x1, x2, h, w
        except Exception:
            pass

    # 2) YOLO fallback (general)
    try:
        model = _get_yolo(device=device)
        # If cuda isn't available, ultralytics will silently run on CPU if device="cpu"
        results = model.predict(img_rgb, imgsz=640, conf=0.25, verbose=False, device=device)
        r0 = results[0]
        if r0.boxes is not None and len(r0.boxes) > 0:
            # Prefer "person" class (COCO class 0) if exists
            boxes = r0.boxes
            cls = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            person_idxs = np.where(cls == 0)[0]
            if len(person_idxs) > 0:
                # choose highest confidence person
                i = person_idxs[int(np.argmax(conf[person_idxs]))]
                px1, py1, px2, py2 = xyxy[i]
                x1, y1, x2, y2 = _face_box_from_person_box(px1, py1, px2, py2, w, h)
                x1, y1, x2, y2 = _pad(x1, y1, x2, y2, w, h, pad=0.08)
                return y1, y2, x1, x2, h, w

            # Otherwise just take the biggest box (sometimes head-only anime triggers a different class)
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            i = int(np.argmax(areas))
            x1, y1, x2, y2 = xyxy[i]
            x1, y1, x2, y2 = _pad(x1, y1, x2, y2, w, h, pad=0.10)
            return y1, y2, x1, x2, h, w
    except Exception:
        pass

    # 3) CV fallback for your style (white/plain background anime) — very effective
    cv_box = _cv_fallback_bbox_white_bg(img_rgb)
    if cv_box is not None:
        x1, y1, x2, y2 = cv_box
        return y1, y2, x1, x2, h, w

    # 4) Last resort: center crop (never crash)
    x1, x2 = int(0.22 * w), int(0.78 * w)
    y1, y2 = int(0.10 * h), int(0.92 * h)
    return y1, y2, x1, x2, h, w

if __name__ == "__main__":
  image_path = sys.argv[1]
  y,y2,x,x2,height,width = get_mask_coord(image_path)
  print (y,y2,x,x2,height,width)