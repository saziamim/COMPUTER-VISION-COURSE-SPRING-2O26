import cv2
import glob
import numpy as np
import os

IMAGE_DIR = "/Users/smim2/Downloads/code for private and secure ai/CV_course_Project/a4_calib_images2"
OUT_FILE  = "/Users/smim2/Downloads/code for private and secure ai/CV_course_Project/camera_calib_checkerboard.npz"

# 1) IMPORTANT: inner corners (not squares)
CHECKERBOARD = (9, 6)  # (cols, rows) of inner corners

# 2) IMPORTANT: measure one square side and set it in meters
SQUARE_SIZE_M = 0.025  # example: 25mm squares -> 0.025m

def get_image_list(image_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.HEIC")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(image_dir, e)))
    return sorted(files)

def main():
    images = get_image_list(IMAGE_DIR)
    if len(images) < 10:
        raise RuntimeError(f"Found only {len(images)} images. Use at least ~10, ideally 20–30.")

    # Create object points grid: (0..8, 0..5) scaled by square size, Z=0
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp[:, :2] *= SQUARE_SIZE_M

    objpoints = []
    imgpoints = []
    img_size = None

    # For sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

    good = 0
    for path in images:
        img = cv2.imread(path)
        if img is None:
            print("Could not read:", path)
            continue

        if img_size is None:
            img_size = (img.shape[1], img.shape[0])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print("No corners found:", os.path.basename(path))
            continue

        # Refine corners to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        good += 1
        print("Accepted:", os.path.basename(path))

        # (Optional) preview detections quickly
        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, found)
        cv2.imshow("Corners", vis)
        cv2.waitKey(150)

    cv2.destroyAllWindows()

    if good < 8:
        raise RuntimeError(f"Only {good} valid images found. Take more / improve views.")

    # Calibrate camera
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    print("\n===== CALIBRATION RESULTS =====")
    print("RMS (OpenCV reported):", ret)
    print("\nCamera matrix K:\n", K)
    print("\nDistortion coeffs:\n", dist.ravel())
    print("Used images:", good)

    np.savez(
        OUT_FILE,
        K=K, dist=dist, img_size=np.array(img_size),
        checkerboard=np.array(CHECKERBOARD),
        square_size_m=np.array(SQUARE_SIZE_M),
        n_images=np.array(good),
        rms=np.array(ret)
    )
    print("\nSaved calibration to:", OUT_FILE)

if __name__ == "__main__":
    main()

