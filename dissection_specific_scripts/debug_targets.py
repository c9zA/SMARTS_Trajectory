"""Quick debug run for the target images."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Patch run_test to not execute main loop
import builtins
_real_open = builtins.open

import importlib, types

# We'll exec run_test up to the main loop
import pathlib, re

src = pathlib.Path(__file__).parent.joinpath("run_test.py").read_text(encoding="utf-8")
# Stop before main loop
stop = re.search(r"^# ── main", src, re.MULTILINE)
src_funcs = src[:stop.start()]

ns = {}
exec(compile(src_funcs, "run_test.py", "exec"), ns)

CFG          = ns["CFG"]
CFG_OVERRIDES= ns["CFG_OVERRIDES"]
run_pipeline = ns["run_pipeline"]
import cv2, numpy as np

BASE     = "D:/University/SMARTS/Spring26/dissection_specific_scripts"
IMG_BASE = f"{BASE}/left_image_selected"

TARGETS = [
    ("6",  "107"),
    ("11", "119"),
    ("41", "20"),
    ("60", "95"),
    ("67", "3"),
    ("67", "8"),
    ("67", "113"),
]

for folder, frame in TARGETS:
    path = f"{IMG_BASE}/{folder}/{frame}.png"
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [SKIP] {path}")
        continue

    ovr = CFG_OVERRIDES.get((folder, frame), {})
    if ovr:
        saved = {k: CFG.get(k, None) for k in ovr}
        CFG.update(ovr)

    print(f"\n{'='*60}")
    print(f"  IMAGE: {folder}.{frame}")
    print(f"{'='*60}")
    run_pipeline(img, debug_label=f"{folder}.{frame}")

    if ovr:
        for k, v in saved.items():
            if v is None: CFG.pop(k, None)
            else: CFG[k] = v
