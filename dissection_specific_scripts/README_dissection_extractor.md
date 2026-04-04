# dissection_extractor.py

Standalone module for extracting a single dissection trajectory line from a surgical image. Given a BGR image containing a tissue dissection with a drawn marker (color, dark, or light), the pipeline returns a skeletonized single-path mask where trajectory pixels = 255.

No dependency on `run_test.py` — all pipeline logic lives in the `DissectionTrajectoryExtractor` class.

---

## Dependencies

```
opencv-python
numpy
scikit-image   (skeletonize)
networkx
```

---

## Quick start

### As a Python module

```python
from dissection_extractor import DissectionTrajectoryExtractor
import cv2

extractor = DissectionTrajectoryExtractor()
img = cv2.imread("image.png")

# Simple: returns uint8 H×W mask, trajectory pixels = 255
mask = extractor(img)

# Override any parameter for one instance
extractor2 = DissectionTrajectoryExtractor(max_link_dist=200, marker_mode="dark")
mask2 = extractor2(img)
```

### Batch / reuse across images

```python
extractor = DissectionTrajectoryExtractor(max_link_dist=200)
for path in image_paths:
    mask = extractor(cv2.imread(path))
    # ...
```

### Convenience function (file in → file or arrays out)

```python
from dissection_extractor import run_trajectory_pipeline

# Returns trajectory mask (uint8 H×W numpy array)
trajectory = run_trajectory_pipeline("img.png")

# Saves a green-overlay PNG to disk, returns None
run_trajectory_pipeline("img.png", "out.png")

# Override any parameter
run_trajectory_pipeline("img.png", "out.png", marker_mode="dark", max_link_dist=200)
```

### Command line

```
python dissection_extractor.py input.png output.png
python dissection_extractor.py input.png output.png --marker_mode dark
python dissection_extractor.py input.png output.png --max_link_dist 200 --chain_max_angle_deg 60
python dissection_extractor.py --help
```

The CLI writes a binary mask (trajectory = 255) to `output.png` and prints the pixel count.

---

## Pipeline overview

```
img_bgr
  │
  ▼
_segment_dark_marker          # color/dark/light HSV mask, clipped to tissue ROI
  │  CLAHE → HSV+LAB → tissue ROI → marker mask → morph open/close → filter small/large components
  ▼
_extract_single_path
  ├─ _extract_anchors          # connected-component blobs with centroid, area, solidity, elongation
  ├─ _suppress_shadows         # multi-pass: border, right/bottom edge, B0 abs-area,
  │                            #   cluster-area, B1 (very large+irregular), B2 (large+dense tiny
  │                            #   neighbours), linearity PCA test, pass-2 re-test
  ├─ _build_chain              # Bellman-Ford DP: area-dominated score, direction constraint
  ├─ _trim_outlier_links       # removes endpoint blobs whose link is > outlier_ratio × P75
  ├─ bridge-node pruning       # removes tiny stepping-stone interior nodes
  ├─ sparkle filter            # if chain is dominated by tiny blobs, retry with sig-only blobs
  ├─ _extend_endpoints         # greedy endpoint extension along trajectory direction
  ├─ post-extension pruning    # prune long links whose small side is isolated noise
  ├─ _trim_outlier_links (ep)  # angle trim, adj-ratio trim, both-small-far trim
  ├─ extension pass 2          # only for loop-triggered sparkle: wider angle extension
  └─ draw links → morphClose → skeletonize → _extract_longest_path
```

Output is a 1-pixel-wide skeleton of the longest path through the drawn trajectory.

---

## Marker mode selection

| Mode | Behaviour |
|------|-----------|
| `auto` (default) | Uses color mask if ≥ `min_color_pixels` pixels; otherwise runs MV-D to decide dark vs. color |
| `color` | Always use purple/blue HSV range (H 112–160) |
| `dark` | Always use dark-ink mask (low V, high S, H ≥ 150) |
| `light` | Always use light/white mask (high V, low S) |

**MV-D** (Multi-Vote Dark/color) casts four independent votes in `auto` mode when the color mask is weak. Two or more votes out of four → use color mask.

---

## Key parameters

### Marker detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `marker_mode` | `"auto"` | Mode selection (see above) |
| `marker_h_min/max` | 112 / 160 | HSV hue range for color marker |
| `marker_s_min` | 90 | Min saturation for color marker |
| `min_color_pixels` | 10000 | Color mask pixel count threshold for fast-path |
| `use_clahe` | `True` | CLAHE contrast enhancement before processing |

### Chain / trajectory building

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_link_dist` | 165.0 | Max gap (px) between consecutive blobs |
| `chain_max_angle_deg` | 55 | Max turn angle in the DP chain |
| `ext_max_angle_deg` | 65 | Max angle for endpoint extension |
| `chain_outlier_ratio` | 3.0 | Trim links > ratio × P75 reference |
| `chain_endpoint_ext_dist` | 380 | Max distance for endpoint extension |
| `dp_link_scale` | 8 | Proportional gap bonus: `max_link + min(area_i,area_j)/scale` |

### Shadow / artifact suppression

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shadow_abs_area_ratio` | 0.008 | B0: blobs > ratio × image pixels are shadows |
| `shadow_large_ratio` | 8.0 | B1/B2 large-blob multiplier (vs. median area) |
| `shadow_solidity_min` | 0.65 | B1/B2 minimum solidity for large blobs |
| `shadow_cluster_area_ratio` | 0.020 | Cluster-area shadow threshold |
| `shadow_linearity_radius` | 200 | PCA neighbourhood radius for linearity test |
| `shadow_linearity_threshold` | 4.0 | PCA ratio below which neighbourhood is 2D → shadow |

### Border exclusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `border_margin_px` | 30 | Edge margin for small-blob suppression |
| `border_margin_max_area` | 300 | Only suppress border blobs smaller than this |
| `right_border_margin_px` | 0 | Suppress all blobs within N px of right edge (disabled by default) |
| `bottom_border_margin_px` | 0 | Suppress all blobs within N px of bottom edge (disabled by default) |

### Sparkle filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sparkle_filter_ratio` | 0.110 | Triggers retry if chain median < ratio × 3rd-largest blob |
| `sparkle_loop_ratio` | 0.35 | Higher threshold used when chain is nearly circular |
| `sparkle_sig_max_link_dist` | 320 | Max link distance for sig-only retry DP |

### Endpoint trimming

| Parameter | Default | Description |
|-----------|---------|-------------|
| `endpoint_trim_angle_deg` | 50 | Trim endpoint if turn angle exceeds this |
| `endpoint_trim_adj_ratio` | 0.04 | Trim endpoint if area < ratio × adjacent blob |
| `endpoint_trim_long_link_ratio` | 1.5 | Trim long-link outlier endpoints below this × chain median |
| `ep_small_max_dist` | 130 | Max link distance when both endpoint and neighbour are small |

### Post-extension pruning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ext_seg_prune_dist` | 218 | Links longer than this are checked for isolated small sides |
| `ext_seg_min_blobs` | 3 | Min blobs on small side to keep a long link |
| `ext_seg_protect_large_area` | 500 | Never prune a small side containing a blob ≥ this area |

### Bridge-node pruning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bridge_min_area_ratio` | 0.15 | Interior node pruned if area < ratio × skip distance |
| `bridge_min_skip_dist` | 200.0 | Only prune bridges when A→C skip exceeds this (px) |

---

## Per-image overrides

For images with instrument artifacts that can't be handled by global parameters, pass override values directly to the constructor or `run_trajectory_pipeline`:

```python
# Suppress instrument artifacts in the bottom 400 px
mask = run_trajectory_pipeline(
    "image.png",
    bottom_border_margin_px=400,
    max_link_dist=215,
)

# Suppress instrument clamp at the right edge
mask = run_trajectory_pipeline(
    "image.png",
    right_border_margin_px=255,
    tiny_blob_req_anchor_ratio=1.0,
)
```

---

## Shadow suppression detail

Blobs are suppressed in order:

1. **Border** — small blobs (≤ `border_margin_max_area`) within `border_margin_px` of any edge
2. **Right/bottom edge** — all blobs within `right_border_margin_px` / `bottom_border_margin_px` of the respective edge
3. **Tiny-no-anchor** — blobs below `median × tiny_blob_req_anchor_ratio` with no large neighbour within `max_link_dist`
4. **B0** — blobs with absolute area > `shadow_abs_area_ratio × total_px`
5. **Cluster-area** — blobs whose spatial cluster exceeds `shadow_cluster_area_ratio × total_px`
6. **B1** — very large (> `shadow_strict_large_ratio × median`) AND low solidity; linearity escape: if neighbourhood is linear AND a large non-shadow neighbour exists, blob is kept
7. **B2** — large (> `shadow_large_ratio × median`) AND low solidity AND many tiny neighbours; escape: if tiny neighbours are themselves linearly arranged, blob is kept
8. **Linearity (Type A)** — PCA over significant neighbours; if not linear, suppress
9. **Pass 2** — re-test linearity-suppressed blobs after removing pass-1 shadows; a blob near a large anchor that now looks linear is restored

---

## Scoring (chain DP)

The Bellman-Ford DP uses **area-dominated scoring**:

```
score = distance × 0.001 + NODE_BONUS + blob.area
NODE_BONUS = 0.1 × median_area   (tie-breaker)
```

Three large ink blobs beat thirty tiny sparkle dots. No cap on area contribution — large legitimate blobs are fully rewarded.
