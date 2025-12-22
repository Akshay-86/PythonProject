import cv2
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
FG_PATH = "bgchange/LS20251222120154.png"      # RGBA PNG
BG_PATH = "inputs/screenshots_20251218_145223_500.00m_1X_p0_v0_a0.png"
SAVE_PATH = "captured_image.jpg"

# ----------------------------
# LOAD IMAGES
# ----------------------------
fg_rgba = cv2.imread(FG_PATH, cv2.IMREAD_UNCHANGED)
bg = cv2.imread(BG_PATH)

if fg_rgba is None or bg is None:
    raise FileNotFoundError("Image not found")

bg_h, bg_w = bg.shape[:2]

# Split foreground
fg_rgb = fg_rgba[:, :, :3]
fg_alpha = fg_rgba[:, :, 3]

orig_fg_h, orig_fg_w = fg_rgb.shape[:2]

# ----------------------------
# STATE
# ----------------------------
scale = 100  # percent
pos_x = (bg_w - orig_fg_w) // 2
pos_y = (bg_h - orig_fg_h) // 2

dragging = False
offset_x = 0
offset_y = 0

# ----------------------------
# CALLBACKS
# ----------------------------
def on_scale(val):
    global scale
    scale = max(val, 1)

def mouse_cb(event, x, y, flags, param):
    global pos_x, pos_y, dragging, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        offset_x = x - pos_x
        offset_y = y - pos_y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        pos_x = x - offset_x
        pos_y = y - offset_y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# ----------------------------
# WINDOW
# ----------------------------
cv2.namedWindow("Editor", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Scale %", "Editor", scale, 150, on_scale)
cv2.setMouseCallback("Editor", mouse_cb)

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    canvas = bg.copy()

    # Resize foreground safely (never exceed background)
    s = scale / 100.0

    fg_w = int(orig_fg_w * s)
    fg_h = int(orig_fg_h * s)

    # Clamp size to background
    fg_w = min(fg_w, bg_w)
    fg_h = min(fg_h, bg_h)

    fg_resized = cv2.resize(
        fg_rgb, (fg_w, fg_h), interpolation=cv2.INTER_LANCZOS4
    )
    alpha_resized = cv2.resize(
        fg_alpha, (fg_w, fg_h), interpolation=cv2.INTER_LINEAR
    )


    # Clamp position to canvas
    x = max(0, min(pos_x, bg_w - fg_w))
    y = max(0, min(pos_y, bg_h - fg_h))

    # Alpha blending
    roi = canvas[y:y+fg_h, x:x+fg_w]
    alpha = alpha_resized / 255.0
    alpha = alpha[..., None]

    blended = (alpha * fg_resized + (1 - alpha) * roi).astype(np.uint8)
    canvas[y:y+fg_h, x:x+fg_w] = blended

    cv2.imshow("Editor", canvas)

    key = cv2.waitKey(20) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('s'):  # SAVE
        cv2.imwrite(SAVE_PATH, canvas)
        print(f"✅ Saved: {SAVE_PATH}")

cv2.destroyAllWindows()
