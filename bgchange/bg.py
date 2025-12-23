import cv2
import numpy as np
import random
import os

# ---------------- CONFIG ----------------
FG_PATH = "inputs/resized_image.png"   # RGBA
BG_PATH = "inputs/premium_photo-1664547606209-fb31ec979c85.jpeg"
OUTPUT_DIR = "outputs"

HORIZON_BAND_RATIO = 0.3

BASE_SCALE = 1.0
K = 15.0
MIN_SCALE = 0.1
MAX_SCALE = 2.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD ----------------
fg_rgba = cv2.imread(FG_PATH, cv2.IMREAD_UNCHANGED)
bg = cv2.imread(BG_PATH)

if fg_rgba is None or bg is None:
    raise FileNotFoundError("Foreground or background image not found")

fg_rgb = fg_rgba[:, :, :3]
fg_alpha = fg_rgba[:, :, 3]
orig_h, orig_w = fg_rgb.shape[:2]

bg_h, bg_w = bg.shape[:2]
horizon_y = int(bg_h * 0.5)

def on_horizon(val):
    global horizon_y
    horizon_y = max(10, min(val, bg_h - 10))

# ---------------- OBJECT CLASS ----------------
class Obj:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.base_scale = BASE_SCALE
        self.selected = False

# ---------------- OBJECT HELPERS ----------------
def create_random_object():
    band_half = int(bg_h * HORIZON_BAND_RATIO / 2)
    top = max(50, horizon_y - band_half)
    bottom = min(bg_h - 50, horizon_y + band_half)

    x = random.randint(100, bg_w - 100)
    y = random.randint(top, bottom)
    return Obj(x, y)

# ---------------- INIT OBJECTS ----------------
objects = [create_random_object() for _ in range(5)]

# ---------------- INTERACTION ----------------
selected_obj = None
drag_offset = (0, 0)

def mouse_cb(event, mx, my, flags, param):
    global selected_obj, drag_offset

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_obj = None

        for obj in objects[::-1]:
            dy = obj.y - horizon_y
            scale = obj.base_scale * (1 + K * dy / bg_h)
            scale = np.clip(scale, MIN_SCALE, MAX_SCALE)

            fw = int(orig_w * scale)
            fh = int(orig_h * scale)

            x0 = int(obj.x - fw // 2)
            y0 = int(obj.y - fh // 2)
            x1 = x0 + fw
            y1 = y0 + fh

            if x0 <= mx <= x1 and y0 <= my <= y1:
                for o in objects:
                    o.selected = False
                obj.selected = True
                selected_obj = obj
                drag_offset = (mx - obj.x, my - obj.y)
                break

    elif event == cv2.EVENT_MOUSEMOVE and selected_obj:
        selected_obj.x = mx - drag_offset[0]
        selected_obj.y = my - drag_offset[1]

    elif event == cv2.EVENT_LBUTTONUP:
        selected_obj = None

# ---------------- WINDOW ----------------
cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Editor", mouse_cb)

cv2.createTrackbar(
    "Horizon",
    "Editor",
    horizon_y,
    bg_h,
    on_horizon
)

# ---------------- SAVE COUNTER ----------------
save_index = 1

def save_image(img):
    global save_index
    while True:
        filename = f"output_{save_index:04d}.jpg"
        path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(path):
            cv2.imwrite(path, img)
            print(f"✅ Saved {path}")
            save_index += 1
            break
        save_index += 1

# ---------------- MAIN LOOP ----------------
while True:
    canvas = bg.copy()

    # draw horizon
    cv2.line(canvas, (0, horizon_y), (bg_w, horizon_y), (0, 0, 255), 2)

    for obj in objects:
        dy = obj.y - horizon_y
        scale = obj.base_scale * (1 + K * dy / bg_h)
        scale = np.clip(scale, MIN_SCALE, MAX_SCALE)

        fw = int(orig_w * scale)
        fh = int(orig_h * scale)

        fg_r = cv2.resize(fg_rgb, (fw, fh), cv2.INTER_LANCZOS4)
        a_r = cv2.resize(fg_alpha, (fw, fh), cv2.INTER_LINEAR)

        x = int(obj.x - fw // 2)
        y = int(obj.y - fh // 2)

        if x < 0 or y < 0 or x + fw > bg_w or y + fh > bg_h:
            continue

        roi = canvas[y:y+fh, x:x+fw]
        alpha = (a_r / 255.0)[..., None]

        canvas[y:y+fh, x:x+fw] = (
            alpha * fg_r + (1 - alpha) * roi
        ).astype(np.uint8)

        if obj.selected:
            cv2.rectangle(canvas, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

    cv2.imshow("Editor", canvas)

    key = cv2.waitKey(20) & 0xFF

    if key == 27:  # ESC
        break
        
    elif key in (ord('d'), ord('D')):
        if objects:
            objects[:] = [o for o in objects if not o.selected]
            selected_obj = None

    elif key in (ord('n'), ord('N')):
        objects.append(create_random_object())

    elif key in (ord('s'), ord('S')):
        save_image(canvas)

cv2.destroyAllWindows()
