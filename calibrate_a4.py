import cv2
import glob
import numpy as np
import os

# =========================
# SETTINGS (edit if needed)
# =========================
IMAGE_DIR = "/Users/smim2/Downloads/code for private and secure ai/CV_course_Project/a4_calib_images"     # folder with your A4 photos
OUT_FILE  = "/Users/smim2/Downloads/code for private and secure ai/CV_course_Project/camera_calib_a4.npz" # output calibration file

# A4 dimensions in meters (portrait or landscape doesn't matter as long as you click correctly)
A4_WIDTH_M  = 0.210  # 297 mm
A4_HEIGHT_M = 0.297  # 210 mm

# Corner click order required:
# 1) Top-Left, 2) Top-Right, 3) Bottom-Right, 4) Bottom-Left
# =========================


def get_image_list(image_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.HEIC")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(image_dir, e)))
    files = sorted(files)
    return files


def draw_instructions(img):
    lines = [
        "Click A4 corners in THIS ORDER:",
        "1) Top-Left",
        "2) Top-Right",
        "3) Bottom-Right",
        "4) Bottom-Left",
        "",
        "Keys: [r]=reset points  [s]=skip image  [q]=quit",
    ]
    y = 30
    for t in lines:
        cv2.putText(img, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28


def reprojection_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_err2 = 0.0
    total_n = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        gt = imgpoints[i].reshape(-1, 2)
        err = np.linalg.norm(gt - proj, axis=1)  # per-point pixel error
        total_err2 += np.sum(err ** 2)
        total_n += len(err)
    return float(np.sqrt(total_err2 / max(total_n, 1)))


def main():
    images = get_image_list(IMAGE_DIR)
    if len(images) < 10:
        raise RuntimeError(f"Found only {len(images)} images. Use at least ~10, ideally 20–30.")

    # 3D object points for A4 corners (Z=0 plane), in meters
    # (0,0) top-left corner of the paper in "paper coordinates"
    objp = np.array([
        [0.0,       0.0,        0.0],          # TL
        [A4_WIDTH_M, 0.0,        0.0],          # TR
        [A4_WIDTH_M, A4_HEIGHT_M, 0.0],         # BR
        [0.0,       A4_HEIGHT_M, 0.0],          # BL
    ], dtype=np.float32)

    objpoints = []  # list of (4,3)
    imgpoints = []  # list of (4,1,2) float32
    img_size = None

    clicked = []  # current image clicks

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked) < 4:
                clicked.append((x, y))

    print("\nINSTRUCTIONS:")
    print("For each image, click the A4 corners in order: TL -> TR -> BR -> BL")
    print("Press 'r' to reset points, 's' to skip image, 'q' to quit.\n")

    for idx, path in enumerate(images):
        img = cv2.imread(path)
        if img is None:
            print("Could not read:", path, "Skipping.")
            continue

        if img_size is None:
            img_size = (img.shape[1], img.shape[0])  # (width, height)

        clicked = []
        win = "A4 Corner Clicker"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            vis = img.copy()
            draw_instructions(vis)

            # draw clicked points
            for i, (cx, cy) in enumerate(clicked):
                cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
                cv2.putText(vis, str(i + 1), (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(vis, f"Image {idx+1}/{len(images)}: {os.path.basename(path)}",
                        (20, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('r'):
                clicked = []
            elif key == ord('s'):
                print("Skipped:", os.path.basename(path))
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print("Quit early.")
                # proceed to calibrate with what we have
                idx = len(images)
                break

            # if 4 points collected, accept this image
            if len(clicked) == 4:
                pts = np.array(clicked, dtype=np.float32).reshape(-1, 1, 2)
                objpoints.append(objp.copy())
                imgpoints.append(pts)
                print("Accepted:", os.path.basename(path))
                break

        cv2.destroyWindow(win)

        # if user quit early, stop looping
        if key == ord('q'):
            break

    if len(objpoints) < 8:
        raise RuntimeError(f"Only {len(objpoints)} valid images collected. Need at least ~8–10.")

    # Calibrate
    # Initialize distortion with 5 coefficients model (common)
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None, criteria=criteria, flags=flags
    )

    rmse = reprojection_rmse(objpoints, imgpoints, rvecs, tvecs, K, dist)

    print("\n===== CALIBRATION RESULTS =====")
    print("RMS (OpenCV reported):", ret)
    print("Reprojection RMSE (pixels):", rmse)
    print("\nCamera matrix K:\n", K)
    print("\nDistortion coeffs (k1,k2,p1,p2,k3,...):\n", dist.ravel())

    np.savez(
        OUT_FILE,
        K=K, dist=dist, img_size=np.array(img_size), rmse=np.array(rmse),
        a4_width_m=np.array(A4_WIDTH_M), a4_height_m=np.array(A4_HEIGHT_M),
        n_images=np.array(len(objpoints))
    )
    print("\nSaved calibration to:", OUT_FILE)
   

if __name__ == "__main__":
    main()
