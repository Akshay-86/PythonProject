import cv2
import numpy as np
import random

# ---------------- CONFIG ----------------
FG_PATH = "bgchange/screenshots_20250906_145328_15.00m_5X_p1_v0_a0-removebg-preview(1).png"   # RGBA
BG_PATH = "inputs/screenshots_20251218_145223_500.00m_1X_p0_v0_a0.png"

BASE_SCALE = 1.0
K = 2.0          # perspective strength
MIN_SCALE = 0.1
MAX_SCALE = 2.0

# ---------------- LOAD ----------------
fg_rgba = cv2.imread(FG_PATH, cv2.IMREAD_UNCHANGED)
bg = cv2.imread(BG_PATH)

fg_rgb = fg_rgba[:, :, :3]
fg_alpha = fg_rgba[:, :, 3]
orig_h, orig_w = fg_rgb.shape[:2]

bg_h, bg_w = bg.shape[:2]
horizon_y = bg_h // 2

# ---------------- OBJECT CLASS ----------------
class Obj:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.base_scale = BASE_SCALE
        self.selected = False

objects = []

# create random objects
for _ in range(5):
    x = random.randint(100, bg_w - 100)
    y = random.randint(100, bg_h - 100)
    objects.append(Obj(x, y))

# ---------------- INTERACTION ----------------
selected_obj = None
drag_offset = (0, 0)

def mouse_cb(event, x, y, flags, param):
    global selected_obj, drag_offset

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_obj = None
        for obj in objects[::-1]:
            if abs(x - obj.x) < 50 and abs(y - obj.y) < 50:
                obj.selected = True
                selected_obj = obj
                drag_offset = (x - obj.x, y - obj.y)
            else:
                obj.selected = False

    elif event == cv2.EVENT_MOUSEMOVE and selected_obj:
        selected_obj.x = x - drag_offset[0]
        selected_obj.y = y - drag_offset[1]

    elif event == cv2.EVENT_LBUTTONUP:
        if selected_obj:
            selected_obj.selected = False
        selected_obj = None

    # mouse wheel resize
    if event == cv2.EVENT_MOUSEWHEEL and selected_obj:
        if flags > 0:
            selected_obj.base_scale *= 1.05
        else:
            selected_obj.base_scale *= 0.95

# ---------------- WINDOW ----------------
cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Editor", mouse_cb)

# ---------------- LOOP ----------------
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

        canvas[y:y+fh, x:x+fw] = (alpha * fg_r + (1 - alpha) * roi).astype(np.uint8)

        if obj.selected:
            cv2.rectangle(canvas, (x, y), (x+fw, y+fh), (0, 255, 0), 2)

    cv2.imshow("Editor", canvas)

    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyAllWindows()
