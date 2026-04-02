"""
Majority-vote threshold learner for dark vs color marker mode.

Three features per image:
  1. mean HSV-H of top-50%-by-S tissue ROI pixels
  2. raw color mask pixel count  (high -> color)
  3. raw dark  mask pixel count  (high -> dark)

Each feature is thresholded independently (brute-force optimal).
Final prediction: majority vote (>=2 of 3 votes for color -> color mode).

Label assignment (folder number):
    dark  = 0 : folders 6-14, 38-43, 67-68
    color = 1 : all other numbered folders
    skip      : folder 60 (ambiguous)
"""

import cv2
import numpy as np
import pathlib
import re
import sys

# -- load helpers from run_test.py --------------------------------------------
_rt  = pathlib.Path(__file__).parent / "run_test.py"
_src = _rt.read_text(encoding="utf-8")
_stop = re.search(r"^# .* main", _src, re.MULTILINE)
_ns = {}
exec(compile(_src[: _stop.start()], "run_test.py", "exec"), _ns)

build_tissue_roi     = _ns["build_tissue_roi"]
enhance_contrast_bgr = _ns["enhance_contrast_bgr"]
CFG                  = _ns["CFG"]

# -- label config -------------------------------------------------------------
DARK_FOLDERS = set(range(6, 15)) | set(range(38, 44)) | {67, 68}
SKIP_FOLDERS = {60}

IMG_BASE = pathlib.Path(
    "D:/University/SMARTS/Spring26/dissection_specific_scripts/left_image_selected"
)


# -- feature extraction -------------------------------------------------------
def extract_features(img_bgr):
    """
    Returns (h_top_s, n_color, n_dark):
      h_top_s : mean H of top-50%-by-S tissue ROI pixels
      n_color : raw color mask pixel count inside tissue ROI
      n_dark  : raw dark  mask pixel count inside tissue ROI
    """
    img = enhance_contrast_bgr(img_bgr)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l, h, s, v = lab[:, :, 0], hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    roi  = build_tissue_roi(l, v, h, s)
    mask = roi > 0

    # feature 1: mean H of top-50%-by-S tissue pixels
    h_roi  = h[mask].astype(np.float32)
    s_roi  = s[mask].astype(np.float32)
    s_mid  = np.median(s_roi) if len(s_roi) > 0 else 0
    top_s  = h_roi[s_roi >= s_mid]
    h_feat = float(np.mean(top_s)) if len(top_s) > 0 else 0.0

    # feature 2: raw color mask pixel count
    mc = (
        (h >= CFG["marker_h_min"]) & (h <= CFG["marker_h_max"]) &
        (s >= CFG["marker_s_min"]) & (v >= CFG["marker_v_min"])
    ).astype(np.uint8) * 255
    n_color = int(np.count_nonzero(cv2.bitwise_and(mc, roi)))

    # feature 3: raw dark mask pixel count
    md = (
        (v <= 100) & (s >= 190) & (h >= 150) & (l <= 60)
    ).astype(np.uint8) * 255
    n_dark = int(np.count_nonzero(cv2.bitwise_and(md, roi)))

    # feature 4: col_px / (drk_px + 1) ratio
    ratio = float(n_color) / (n_dark + 1)

    # feature 5: white tissue pixel count
    # Very bright (V>=180) + low saturation (S<=60) = white/pale fascia tissue.
    mw = (
        (v >= 180) & (s <= 60)
    ).astype(np.uint8) * 255
    n_white = int(np.count_nonzero(cv2.bitwise_and(mw, roi)))

    # feature 6: narrow dark ink pixel count
    # Same as drk_px but with H restricted to [150,165] and V tightened to <=80.
    # True dark ink lives at H=150-165 (purplish-dark).
    # Red meat that inflates drk_px lives at H=165-180 (red-wrap) with V=60-100.
    # Narrowing H and V excludes red meat false positives from the dark count.
    mn = (
        (v <= 80) & (s >= 190) & (h >= 150) & (h <= 165) & (l <= 60)
    ).astype(np.uint8) * 255
    n_narrow = int(np.count_nonzero(cv2.bitwise_and(mn, roi)))

    return h_feat, n_color, n_dark, ratio, n_white, n_narrow


# -- collect samples ----------------------------------------------------------
fh_list, fc_list, fd_list, fr_list, fw_list, fn_list, labels, names = [], [], [], [], [], [], [], []

all_dirs = sorted(
    (d for d in IMG_BASE.iterdir() if d.is_dir() and d.name.isdigit()),
    key=lambda p: int(p.name),
)

print(f"Scanning {len(all_dirs)} folders ...\n")
for folder_dir in all_dirs:
    fnum = int(folder_dir.name)
    if fnum in SKIP_FOLDERS:
        continue
    label = 0 if fnum in DARK_FOLDERS else 1

    for img_path in sorted(folder_dir.glob("*.png")):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [SKIP] {img_path}")
            continue
        h_f, n_c, n_d, ratio, n_w, n_n = extract_features(img)
        tag = "dark " if label == 0 else "color"
        print(f"  {fnum:>3}/{img_path.stem:<6}  h={h_f:5.2f}  col={n_c:>7}  drk={n_d:>7}  ratio={ratio:7.3f}  white={n_w:>7}  narrow={n_n:>7}  [{tag}]")
        fh_list.append(h_f)
        fc_list.append(n_c)
        fd_list.append(n_d)
        fr_list.append(ratio)
        fw_list.append(n_w)
        fn_list.append(n_n)
        labels.append(label)
        names.append(f"{fnum}/{img_path.stem}")

fh     = np.array(fh_list, dtype=np.float64)
fc     = np.array(fc_list, dtype=np.float64)
fd     = np.array(fd_list, dtype=np.float64)
fr     = np.array(fr_list, dtype=np.float64)
fw     = np.array(fw_list, dtype=np.float64)
fn     = np.array(fn_list, dtype=np.float64)
labels = np.array(labels,  dtype=np.float64)

n_lc = int(labels.sum())
n_ld = int((1 - labels).sum())
print(f"\n{'-'*65}")
print(f"  Samples: {len(labels)}  ({n_ld} dark, {n_lc} color)")
print(f"  h_top_s  dark={fh[labels==0].mean():.2f}    color={fh[labels==1].mean():.2f}")
print(f"  col_px   dark={fc[labels==0].mean():.0f}     color={fc[labels==1].mean():.0f}")
print(f"  drk_px   dark={fd[labels==0].mean():.0f}    color={fd[labels==1].mean():.0f}")
print(f"  ratio    dark={fr[labels==0].mean():.3f}    color={fr[labels==1].mean():.3f}")
print(f"  white    dark={fw[labels==0].mean():.0f}     color={fw[labels==1].mean():.0f}")
print(f"  narrow   dark={fn[labels==0].mean():.0f}     color={fn[labels==1].mean():.0f}")
print(f"{'-'*65}\n")


# -- brute-force threshold finder ---------------------------------------------
def best_threshold_color(feat):
    """feat >= T predicts color (label=1)."""
    cands = np.sort(np.unique(feat))
    best_t, best_a = cands[0], 0.0
    for t in cands:
        pred = (feat >= t).astype(int)
        a = float(np.mean(pred == labels.astype(int)))
        if a > best_a:
            best_a, best_t = a, t
    return best_t, best_a

def best_threshold_dark(feat):
    """feat >= T predicts dark (label=0), so color prediction = feat < T."""
    cands = np.sort(np.unique(feat))
    best_t, best_a = cands[0], 0.0
    for t in cands:
        pred = 1 - (feat >= t).astype(int)
        a = float(np.mean(pred == labels.astype(int)))
        if a > best_a:
            best_a, best_t = a, t
    return best_t, best_a


# -- individual classifiers ---------------------------------------------------
T_h, acc_h = best_threshold_color(fh)
T_c, acc_c = best_threshold_color(fc)
T_d, acc_d = best_threshold_dark(fd)
T_r, acc_r = best_threshold_color(fr)
T_w, acc_w = best_threshold_color(fw)
T_n, acc_n = best_threshold_dark(fn)   # narrow dark ink high -> dark

pred_h = (fh >= T_h).astype(int)
pred_c = (fc >= T_c).astype(int)
pred_d = 1 - (fd >= T_d).astype(int)
pred_r = (fr >= T_r).astype(int)
pred_w = (fw >= T_w).astype(int)
pred_n = 1 - (fn >= T_n).astype(int)  # narrow dark high -> dark mode

# narrow ratio: col_px / (drk_narrow + 1)
fn_ratio = fc / (fn + 1)
T_nr, acc_nr = best_threshold_color(fn_ratio)
pred_nr = (fn_ratio >= T_nr).astype(int)

def n_correct(pred):
    return int(np.sum(pred == labels.astype(int)))

print(f"  Individual classifiers (brute-force optimal threshold):")
print(f"    [1] h_top_s      >= {T_h:.2f}     -> color  acc={acc_h:.3f}  ({n_correct(pred_h)}/{len(labels)})")
print(f"    [2] col_px       >= {T_c:.0f}      -> color  acc={acc_c:.3f}  ({n_correct(pred_c)}/{len(labels)})")
print(f"    [3] drk_px       >= {T_d:.0f}      -> dark   acc={acc_d:.3f}  ({n_correct(pred_d)}/{len(labels)})")
print(f"    [4] ratio        >= {T_r:.4f}    -> color  acc={acc_r:.3f}  ({n_correct(pred_r)}/{len(labels)})")
print(f"    [5] white        >= {T_w:.0f}      -> color  acc={acc_w:.3f}  ({n_correct(pred_w)}/{len(labels)})")
print(f"    [6] drk_narrow   >= {T_n:.0f}      -> dark   acc={acc_n:.3f}  ({n_correct(pred_n)}/{len(labels)})")
print(f"    [7] narrow_ratio >= {T_nr:.4f}   -> color  acc={acc_nr:.3f}  ({n_correct(pred_nr)}/{len(labels)})")


# -- majority vote A: col, drk, ratio ------------------------------------------
votes_a   = pred_c + pred_d + pred_r
mv_a      = (votes_a >= 2).astype(int)
correct_a = mv_a == labels.astype(int)
acc_a     = float(np.mean(correct_a))

# -- majority vote B: drk, ratio, white ----------------------------------------
votes_b   = pred_d + pred_r + pred_w
mv_b      = (votes_b >= 2).astype(int)
correct_b = mv_b == labels.astype(int)
acc_b     = float(np.mean(correct_b))

# -- majority vote C: narrow_ratio, drk_narrow, white (>=2/3) -----------------
votes_c   = pred_nr + pred_n + pred_w
mv_c      = (votes_c >= 2).astype(int)
correct_c = mv_c == labels.astype(int)
acc_c_mv  = float(np.mean(correct_c))

# -- majority vote D: narrow_ratio, drk_narrow, white, ratio (>=2/4) ----------
votes_d   = pred_nr + pred_n + pred_w + pred_r
mv_d      = (votes_d >= 2).astype(int)
correct_d = mv_d == labels.astype(int)
acc_d_mv  = float(np.mean(correct_d))

print(f"\n{'='*65}")
print(f"  MV-A (2/3: col, drk, ratio)                  acc={acc_a:.3f}  ({correct_a.sum()}/{len(labels)})")
print(f"  MV-B (2/3: drk, ratio, white)                acc={acc_b:.3f}  ({correct_b.sum()}/{len(labels)})")
print(f"  MV-C (2/3: narrow_ratio, drk_narrow, white)  acc={acc_c_mv:.3f}  ({correct_c.sum()}/{len(labels)})")
print(f"  MV-D (>=2/4: narrow_ratio, drk_narrow, white, ratio) acc={acc_d_mv:.3f}  ({correct_d.sum()}/{len(labels)})")
print(f"{'='*65}")

# show misclassified for best vote
options = [("MV-A", correct_a), ("MV-B", correct_b), ("MV-C", correct_c), ("MV-D", correct_d)]
best_mv, best_correct = max(options, key=lambda x: x[1].mean())
print(f"\n  Best: {best_mv} -- misclassified:")
wrong = [(names[i], fc[i], fd[i], fn[i], fn_ratio[i], fw[i],
          pred_c[i], pred_d[i], pred_n[i], pred_nr[i], pred_w[i], int(labels[i]))
         for i in range(len(labels)) if not best_correct[i]]

if wrong:
    print(f"  {'name':<16}  {'col':>7}  {'drk':>7}  {'narrow':>7}  {'n_ratio':>8}  {'white':>7}  vc vd vn vnr vw  true")
    print(f"  {'-'*16}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  -- -- -- --- --  ----")
    for nm, fc_, fd_, fn_, fnr_, fw_, vc, vd, vn, vnr, vw, true in wrong:
        tl = "dark " if true == 0 else "color"
        print(f"  {nm:<16}  {fc_:7.0f}  {fd_:7.0f}  {fn_:7.0f}  {fnr_:8.3f}  {fw_:7.0f}   {vc}  {vd}  {vn}  {vnr:3}  {vw}  {tl}")
else:
    print("  All samples correctly classified.")

print(f"\n  Summary:")
print(f"    [1] h_top_s      T={T_h:.2f}       -> {acc_h:.1%}")
print(f"    [2] col_px       T={T_c:.0f}        -> {acc_c:.1%}")
print(f"    [3] drk_px       T={T_d:.0f}        -> {acc_d:.1%}")
print(f"    [4] ratio        T={T_r:.4f}      -> {acc_r:.1%}")
print(f"    [5] white        T={T_w:.0f}        -> {acc_w:.1%}")
print(f"    [6] drk_narrow   T={T_n:.0f}        -> {acc_n:.1%}")
print(f"    [7] narrow_ratio T={T_nr:.4f}     -> {acc_nr:.1%}")
print(f"    MV-A (col,drk,ratio)                   -> {acc_a:.1%}")
print(f"    MV-B (drk,ratio,white)                 -> {acc_b:.1%}")
print(f"    MV-C (narrow_ratio,drk_narrow,white)   -> {acc_c_mv:.1%}")
print(f"    MV-D (narrow_ratio,drk_narrow,white,ratio) -> {acc_d_mv:.1%}")
