import cv2
import numpy as np
import os

# =========================
# SETTINGS
# =========================
CALIB_NPZ  = "/Users/smim2/Downloads/code for private and secure ai/CV_course_Project/camera_calib_checkerboard.npz"
IMAGE_PATH = "/Users/smim2/Downloads/code for private and secure ai/CV_course_Project/measure_images/m13.JPG"

# A4 size in meters (use correct orientation!)
A4_WIDTH_M  = 0.210
A4_HEIGHT_M = 0.297
# =========================


def load_calibration(npz_path):
    """Load camera intrinsic matrix K and lens distortion coefficients dist."""
    data = np.load(npz_path)
    return data["K"], data["dist"]


def undistort_points(pts_px, K, dist):
    """
    Undistort pixel points to remove lens distortion.
    Using P=K keeps the output in pixel coordinates (undistorted).
    """
    pts_px = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    pts_ud = cv2.undistortPoints(pts_px, K, dist, P=K)
    return pts_ud.reshape(-1, 2)


def compute_homography_img_to_a4(K, dist, a4_corners_px):
    """
    Compute homography that maps UNDISTORTED image pixels -> A4 coordinates (meters).
    We do:
      - undistort clicked A4 corners
      - define their real-world A4 coordinates in meters
      - estimate H (img->A4)
    """
    # Undistort the clicked A4 corners first
    a4_corners_ud = undistort_points(a4_corners_px, K, dist)

    # Real-world A4 coordinates (meters), MUST match click order:
    # TL, TR, BR, BL
    a4_world = np.array([
        [0.0,       0.0],        # TL
        [A4_WIDTH_M, 0.0],        # TR
        [A4_WIDTH_M, A4_HEIGHT_M],# BR
        [0.0,       A4_HEIGHT_M], # BL
    ], dtype=np.float32)

    # Homography maps (img undistorted) -> (A4 meters)
    H, _ = cv2.findHomography(a4_corners_ud, a4_world, method=0)
    if H is None:
        raise RuntimeError("Homography failed. Make sure you clicked corners correctly.")
    return H


def pixel_to_a4_xy(K, dist, H_img_to_a4, u, v):
    """
    Convert a clicked pixel (u,v) to A4 (x,y) in meters:
      - undistort the point
      - apply H to map to meters
    """
    pt_ud = undistort_points([(u, v)], K, dist)[0]
    x, y = float(pt_ud[0]), float(pt_ud[1])

    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H_img_to_a4 @ p
    q /= q[2]
    return float(q[0]), float(q[1])


def main():
    if not os.path.exists(CALIB_NPZ):
        raise FileNotFoundError(f"Calibration file not found: {CALIB_NPZ}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # Load calibration
    K, dist = load_calibration(CALIB_NPZ)

    # Load image
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("Could not read image. Check IMAGE_PATH.")
    vis = img.copy()

    # -------------------------
    # Step 1: Click paper corners
    # -------------------------
    a4_clicks = []  # TL, TR, BR, BL

    def on_mouse_a4(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(a4_clicks) < 4:
            a4_clicks.append((x, y))

    win1 = "Click paper corners: 1)TL 2)TR 3)BR 4)BL  (press r reset, q quit)"
    cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win1, on_mouse_a4)

    print("\nSTEP 1: Click the 4 paper corners in order:")
    print("1) Top-Left, 2) Top-Right, 3) Bottom-Right, 4) Bottom-Left")
    print("Keys: r=reset, q=quit\n")

    while True:
        show = vis.copy()
        for i, (cx, cy) in enumerate(a4_clicks):
            cv2.circle(show, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(show, str(i+1), (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow(win1, show)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            a4_clicks = []
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("Quit.")
            return

        if len(a4_clicks) == 4:
            break

    cv2.destroyWindow(win1)

    # Compute mapping from image pixels -> A4 meters
    H_img_to_a4 = compute_homography_img_to_a4(K, dist, a4_clicks)

    # -------------------------
    # Step 2: Click mug bottom + top
    # -------------------------
    mug_clicks = []  # bottom, top

    def on_mouse_mug(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(mug_clicks) < 2:
            mug_clicks.append((x, y))

    win2 = "Click mug points: 1)BOTTOM (on A4 plane) 2)TOP (rim)  (q quit)"
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win2, on_mouse_mug)

    print("STEP 2: Click mug points:")
    print("1) Bottom point (where mug meets the A4 sheet), 2) Top rim point\n")

    while True:
        show = vis.copy()

        # draw A4 corner clicks
        for (cx, cy) in a4_clicks:
            cv2.circle(show, (cx, cy), 6, (255, 0, 0), -1)

        # draw mug clicks
        for i, (cx, cy) in enumerate(mug_clicks):
            cv2.circle(show, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(show, str(i+1), (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow(win2, show)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            print("Quit.")
            return

        if len(mug_clicks) == 2:
            break

    cv2.destroyAllWindows()

    # Convert clicked mug pixels to A4 coordinates (meters)
    (u1, v1), (u2, v2) = mug_clicks
    x_bottom, y_bottom = pixel_to_a4_xy(K, dist, H_img_to_a4, u1, v1)
    x_top,    y_top    = pixel_to_a4_xy(K, dist, H_img_to_a4, u2, v2)

    # Height estimate: difference along the A4 vertical axis (meters -> cm)
    height_m  = abs(y_top - y_bottom)
    height_cm = height_m * 100.0

    print("\n===== RESULTS =====")
    print(f"Bottom : x={x_bottom:.4f}, y={y_bottom:.4f}")
    print(f"Top    : x={x_top:.4f},    y={y_top:.4f}")
    print(f"Estimated mug height: {height_cm:.2f} cm")

    print("\nNOTE:")
    print("This works best when the mug is touching (or extremely close to) the A4 plane.\n")


if __name__ == "__main__":
    main()

