"""
Probe which mask mode (color / dark) auto-selection picks for every image
in left_image_selected. No modifications to dissection_extractor.py.
Replicates only the mask-selection logic from _segment_dark_marker.
"""
import os
import sys
import cv2
import numpy as np

# ── defaults (mirror DissectionTrajectoryExtractor.__init__) ────────────────
MARKER_H_MIN          = 112
MARKER_H_MAX          = 160
MARKER_S_MIN          = 90
MARKER_V_MIN          = 40
MIN_COLOR_PIXELS      = 10000
LIGHT_V_MIN           = 200
LIGHT_S_MAX           = 60
MV_VOTES_NEEDED       = 2
MV_RATIO_THRESHOLD    = 0.5748
MV_NARROW_THRESHOLD   = 147
MV_NARROW_RATIO_THR   = 2.402
MV_WHITE_THRESHOLD    = 188112
TISSUE_V_MIN          = 40
TISSUE_L_MIN          = 40
CLOTH_H_MIN           = 85
CLOTH_H_MAX           = 135
CLOTH_S_MIN           = 25
TISSUE_DILATE_KERNEL  = 40


def enhance_contrast(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def build_tissue_roi(lab_l, hsv_v, hsv_h, hsv_s):
    bright     = (hsv_v >= TISSUE_V_MIN) & (lab_l >= TISSUE_L_MIN)
    blue_cloth = (hsv_h >= CLOTH_H_MIN) & (hsv_h <= CLOTH_H_MAX) & (hsv_s >= CLOTH_S_MIN)
    tissue     = (bright & ~blue_cloth).astype(np.uint8) * 255
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN,  k4)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, k5, iterations=2)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TISSUE_DILATE_KERNEL, TISSUE_DILATE_KERNEL))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kd)
    tissue = cv2.dilate(tissue, kd, iterations=1)
    return tissue


def probe_mask_mode(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return "ERROR", {}

    img  = enhance_contrast(img_bgr)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l, h, s, v = lab[:,:,0], hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    tissue_roi = build_tissue_roi(l, v, h, s)

    mask_color = (
        (h >= MARKER_H_MIN) & (h <= MARKER_H_MAX) &
        (s >= MARKER_S_MIN) & (v >= MARKER_V_MIN)
    ).astype(np.uint8) * 255
    mask_color = cv2.bitwise_and(mask_color, tissue_roi)

    mask_dark = (
        (v <= 100) & (s >= 190) & (h >= 150) & (l <= 60)
    ).astype(np.uint8) * 255
    mask_dark = cv2.bitwise_and(mask_dark, tissue_roi)

    n_color = int(np.count_nonzero(mask_color))
    n_dark  = int(np.count_nonzero(mask_dark))

    # fast-path
    if n_color >= MIN_COLOR_PIXELS:
        return "color (fast-path)", {"n_color": n_color, "n_dark": n_dark}

    # MV-D voting
    mask_narrow = (
        (v <= 80) & (s >= 190) & (h >= 150) & (h <= 165) & (l <= 60)
    ).astype(np.uint8) * 255
    n_narrow = int(np.count_nonzero(cv2.bitwise_and(mask_narrow, tissue_roi)))

    mask_white = (
        (v >= LIGHT_V_MIN) & (s <= LIGHT_S_MAX)
    ).astype(np.uint8) * 255
    n_white = int(np.count_nonzero(cv2.bitwise_and(mask_white, tissue_roi)))

    ratio        = n_color / (n_dark   + 1)
    narrow_ratio = n_color / (n_narrow + 1)

    v1 = ratio        >= MV_RATIO_THRESHOLD
    v2 = n_narrow      < MV_NARROW_THRESHOLD
    v3 = narrow_ratio >= MV_NARROW_RATIO_THR
    v4 = n_white      >= MV_WHITE_THRESHOLD
    votes = sum([v1, v2, v3, v4])

    chosen = "color (MV-D)" if votes >= MV_VOTES_NEEDED else "dark"
    stats = {
        "n_color": n_color, "n_dark": n_dark, "n_narrow": n_narrow,
        "n_white": n_white, "votes": votes,
        "v1(ratio)": f"{ratio:.3f}>={MV_RATIO_THRESHOLD}={v1}",
        "v2(narrow<thr)": f"{n_narrow}<{MV_NARROW_THRESHOLD}={v2}",
        "v3(narr_ratio)": f"{narrow_ratio:.3f}>={MV_NARROW_RATIO_THR}={v3}",
        "v4(white)": f"{n_white}>={MV_WHITE_THRESHOLD}={v4}",
    }
    return chosen, stats


def main():
    root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "left_image_selected")
    )
    results = []
    for folder in sorted(os.listdir(root), key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path), key=lambda x: int(x.replace("-1","").replace(".png","")) if x.replace("-1","").replace(".png","").isdigit() else x):
            if not fname.lower().endswith(".png"):
                continue
            img_path = os.path.join(folder_path, fname)
            mode, stats = probe_mask_mode(img_path)
            results.append((folder, fname.replace(".png",""), mode, stats))

    # print summary
    print(f"{'folder':>8}  {'frame':<10}  {'mode':<20}  votes  details")
    print("-" * 90)
    for folder, frame, mode, stats in results:
        votes = stats.get("votes", "-")
        n_color = stats.get("n_color", 0)
        n_dark  = stats.get("n_dark", 0)
        print(f"{folder:>8}  {frame:<10}  {mode:<20}  {votes!s:<6} n_color={n_color:6d}  n_dark={n_dark:6d}")

    print()
    from collections import Counter
    counts = Counter(m for _,_,m,_ in results)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} images")


if __name__ == "__main__":
    main()
