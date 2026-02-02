import cv2
import numpy as np

# -----------------------------
# Load calibration
# -----------------------------
data = np.load("camera_calib_a4.npz")
K = data["K"]
dist = data["dist"]

# A4 dimensions (portrait, meters)
A4_W, A4_H = 0.210, 0.297

# A4 world coordinates (same order you used consistently)
objp = np.array([
    [0,      0,      0],     # corner 1
    [A4_W,   0,      0],     # corner 2
    [A4_W,   A4_H,   0],     # corner 3
    [0,      A4_H,   0],     # corner 4
], dtype=np.float32)

# -----------------------------
# Load image
# -----------------------------
img = cv2.imread("/Users/smim2/Downloads/code for private and secure ai/CV_course_Project/a4_calib_images/m3.JPG")
if img is None:
    raise RuntimeError("Could not read measure.jpg")

img = cv2.undistort(img, K, dist)

clicked = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked.append((x, y))
        print(f"Clicked: {x}, {y}")

# -----------------------------
# Step 1: click A4 corners
# -----------------------------
print("\nCLICK A4 CORNERS (same order as calibration)")
print("Then press 'n' to continue\n")

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_mouse)

while True:
    vis = img.copy()
    for p in clicked:
        cv2.circle(vis, p, 6, (0,255,0), -1)
    cv2.imshow("image", vis)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('n') and len(clicked) >= 4:
        break

a4_img_pts = np.array(clicked[:4], dtype=np.float32)

clicked = []

# -----------------------------
# Step 2: click object points
# -----------------------------
print("\nCLICK TWO POINTS on the object (e.g., left & right edge)")
print("Then press 'q'\n")

while True:
    vis = img.copy()
    for p in clicked:
        cv2.circle(vis, p, 6, (0,0,255), -1)
    cv2.imshow("image", vis)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q') and len(clicked) >= 2:
        break

obj_img_pts = np.array(clicked[:2], dtype=np.float32)

cv2.destroyAllWindows()

# -----------------------------
# Compute homography
# -----------------------------
H, _ = cv2.findHomography(a4_img_pts, objp[:, :2])

def img_to_world(pt, H):
    p = np.array([pt[0], pt[1], 1.0])
    w = H @ p
    return w[:2] / w[2]

P1 = img_to_world(obj_img_pts[0], H)
P2 = img_to_world(obj_img_pts[1], H)

dist_m = np.linalg.norm(P1 - P2)

print("\n===== MEASUREMENT RESULT =====")
print(f"Measured distance: {dist_m:.4f} meters")
print(f"Measured distance: {dist_m*100:.2f} cm")
