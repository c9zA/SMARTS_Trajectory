"""Sync CFG and functions from run_test.py into the notebook.

Extracts content directly from run_test.py so there is no duplication.
  - Notebook imports+CFG cell  ← CFG dict + standard imports from run_test.py
  - Notebook functions cell    ← all helper/pipeline functions from run_test.py
"""
import json, pathlib, re, textwrap

RUN_TEST = pathlib.Path(
    "D:/University/SMARTS/Spring26/dissection_specific_scripts"
    "/dissection_specific_scripts/run_test.py"
)
NB_PATH = pathlib.Path(
    "D:/University/SMARTS/Spring26/dissection_specific_scripts"
    "/dissection_specific_scripts/dissection_marker_to_single_path_mask.ipynb"
)

src = RUN_TEST.read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)

# ── 1. Extract CFG block ─────────────────────────────────────────────────────
# From "CFG = {" to the matching closing "}"
cfg_start = next(i for i, l in enumerate(lines) if l.strip().startswith("CFG = {"))
depth, cfg_end = 0, cfg_start
for i, l in enumerate(lines[cfg_start:], cfg_start):
    depth += l.count("{") - l.count("}")
    if depth == 0:
        cfg_end = i
        break
cfg_block = "".join(lines[cfg_start : cfg_end + 1])

# ── 2. Extract all helper/pipeline function definitions ──────────────────────
# Collect every top-level "def " block up to (but not including) the main loop.
# Stop at the "# ── main" sentinel or "for folder, frame in TEST_IMAGES".
func_lines = []
in_func = False
for line in lines:
    stripped = line.rstrip()
    # Stop at main execution block
    if re.match(r"^# ── main", stripped) or re.match(r"^for folder, frame", stripped):
        break
    if re.match(r"^def ", stripped) or re.match(r"^class ", stripped):
        in_func = True
    if in_func:
        func_lines.append(line)

functions_block = "".join(func_lines).rstrip() + "\n"

# ── 3. Build notebook cell sources ───────────────────────────────────────────
IMPORTS_AND_CFG = (
    "import cv2\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from skimage.morphology import skeletonize\n"
    "import networkx as nx\n"
    "\n"
    "\n"
    "# ------ Configurable parameters ------\n"
    "\n"
) + cfg_block + "\n"

FUNCTIONS_ONLY = functions_block

# ── 4. Patch notebook cells ───────────────────────────────────────────────────
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

def first_line(src):
    for line in src.split("\n"):
        s = line.strip()
        if s:
            return s
    return ""

def to_source_lines(code):
    ls = code.splitlines(keepends=True)
    while ls and ls[0].strip() == "":
        ls = ls[1:]
    return ls

imports_idx = func_idx = None
for idx, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    fl = first_line("".join(cell["source"]))
    if fl.startswith("import cv2") and imports_idx is None:
        imports_idx = idx
    if ("def enhance_contrast_bgr" in "".join(cell["source"])) and func_idx is None:
        func_idx = idx

assert imports_idx is not None, "Could not find imports+CFG cell (import cv2)"
assert func_idx    is not None, "Could not find functions cell (def enhance_contrast_bgr)"

cells[imports_idx]["source"] = to_source_lines(IMPORTS_AND_CFG)
cells[func_idx]["source"]    = to_source_lines(FUNCTIONS_ONLY)

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Synced imports+CFG (cell {imports_idx}) and functions (cell {func_idx}) to notebook.")
print(f"  CFG lines: {cfg_block.count(chr(10))}")
print(f"  Function lines: {functions_block.count(chr(10))}")
