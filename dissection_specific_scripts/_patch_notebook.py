import json

with open('dissection_marker_to_single_path_mask.ipynb') as f:
    nb = json.load(f)

# ── CELL 2: updated CFG ────────────────────────────────────────────────────────
cell2_new = r"""import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
import networkx as nx



# ------ Configurable parameters ------

CFG = {
    # Optional enhancement
    "use_clahe": True,

    # Tissue ROI gating (restrict marker search to tissue neighborhood)
    # "tissue_v_min": 55,
    # "tissue_l_min": 60,
    "tissue_v_min": 40,
    "tissue_l_min": 40,
    # "tissue_h_max":70,
    # "tissue_h_min":112,
    "cloth_h_min": 85,
    "cloth_h_max": 135,
    "cloth_s_min": 25,

    #"tissue_dilate_kernel": 25,
    "tissue_dilate_kernel": 40,



    # --- VERY IMPORTANT !!! ---
    # You need to define a set of color thresholds, different from the one below, if you
    # want to detect lines that are not marked by purple/blue color!!
    # There exists a dark fallback threshold set, but beware it's not guaranteed to work!

    #Purple/blue marker thresholds (primary mode)

    "use_color_marker": True,
    "marker_h_min": 112,
    "marker_h_max": 160,
    "marker_s_min": 90,
    "marker_v_min": 40,

    # "use_color_marker": True,
    # "marker_h_min": 112,
    # "marker_h_max": 175,
    # "marker_s_min": 80,
    # "marker_v_min": 40,

    # Dark fallback thresholds (used only when color marker pixels are too few)
    #"lab_l_max": 115,
    #"hsv_h_min": 125,
    "hsv_h_min": 112,
    #"hsv_h_max": 150,
    "hsv_h_max": 160,
    #"lab_l_max": 60,
    "lab_l_max": 80,
    "hsv_v_max": 125,
    "hsv_s_max": 100,
    "hsv_s_min": 158,

    "min_color_pixels": 10000,

    # denoise + dashed/dotted gap bridging
    "open_kernel": 2,
    "close_kernel": 11,

    "close_iters": 1,

    # Component filtering after segmentation
    "min_component_area": 25,
    "max_component_area_ratio": 0.03,

    # Branch pruning and final rendering
    "prune_branch_length": 10,
    "line_thickness": 3,

    # Longest smooth chain selection across dotted fragments
    # "max_link_dist": 180.0,
    "max_link_dist": 320.0,
    "max_proj_gap": 180.0,
    "max_lateral_dev": 90.0,
    # "min_chain_nodes": 3,
    "min_chain_nodes": 2,
    "link_thickness": 4,

    # Shadow suppression
    # "shadow_small_ratio":     2.0,  # replaced by shadow_linearity_threshold
    "shadow_density_radius":  250,
    # "shadow_density_count":   5,   # raised to 4: fire linearity test with fewer neighbors
    "shadow_density_count":   4,
    "shadow_linearity_threshold": 4.0,
    # "shadow_large_ratio":     4.0,  # replaced by shadow_linearity_threshold
    "shadow_large_ratio":      8.0,   # re-enabled for solidity-based Type B
    "shadow_solidity_min":     0.65,   # merged irregular blobs have low solidity
    "shadow_strict_large_ratio": 6.0, # Type B1: area > median*6 -> definitely merged shadow
    "shadow_strict_solidity":    0.35, # Type B1: solidity < 0.35 -> very irregular merged blob
    "shadow_abs_area_ratio":     0.0042, # Type B0: any blob > 0.42% of frame is shadow/artifact
    # "shadow_elongation_min":  2.5,  # replaced by shadow_linearity_threshold

    # Direction chain
    "chain_max_angle_deg":    75,
    # "chain_outlier_ratio":   2.5,
    # "chain_outlier_ratio":   3.0,
    "trim_ref_floor_ratio":  0.30,   # floor = max_link_dist * this (90 px) prevents over-trim on tight chains
    "border_margin_px":      30,    # blobs within this many px of any image edge are instrument artifacts
    "border_margin_max_area": 300,   # only suppress border blobs smaller than this (px); larger = valid ink
    "shadow_cluster_area_ratio": 0.020,  # suppress blob if sum of nearby blob areas > 2% of frame
    # "chain_outlier_ratio":   2.5,  # too strict: cuts legitimate end blobs (41.20)
    "chain_outlier_ratio":   3.0,
}

print("Config loaded.")"""

nb['cells'][2]['source'] = cell2_new

# ── CELL 3: updated functions ──────────────────────────────────────────────────
cell3_new = r"""def enhance_contrast_bgr(img_bgr):
    #increase the contrast of color img
    if not CFG["use_clahe"]:
        return img_bgr

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

#this is used in segment_dark_marker after
def filter_components(binary_mask, min_area, max_area_ratio):
    h, w = binary_mask.shape
    max_area = int(max_area_ratio * h * w)#this is a number, so that large blobs like blood is not confounded


    #returns labels_num to each component, then returns a matrix labels same size as img, each pixel labeled group ID, then
    # stats is a matrix with each label denoting bounding boxes width height, contain CC_STAT_AREA pixel count
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    #creates a blank img same size
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):#0 label = bg
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cleaned[labels == i] = 255#each pixel of this component marked white
    return cleaned#only has components of appropriate size


# def build_tissue_roi(lab_l, hsv_v, hsv_h):
#     # Approximate tissue area (bright enough), then dilate(expand border) so marker near boundaries is retained.
#     tissue = ((hsv_v >= CFG["tissue_v_min"]) & (lab_l >= CFG["tissue_l_min"]) & ((hsv_h<=CFG["tissue_h_max"]) | (hsv_h>=CFG["tissue_h_min"])))#identify the tissue
#     tissue_u8 = (tissue.astype(np.uint8) * 255)#saturate the whole tissue
#
#     k = cv2.getStructuringElement(
#         cv2.MORPH_ELLIPSE,
#         (CFG["tissue_dilate_kernel"], CFG["tissue_dilate_kernel"]),
#     )
#     tissue_u8 = cv2.dilate(tissue_u8, k, iterations=0)
#     return tissue_u8
def build_tissue_roi(lab_l, hsv_v, hsv_h, hsv_s):
    bright = (hsv_v >= CFG["tissue_v_min"]) & (lab_l >= CFG["tissue_l_min"])
    blue_cloth = (
        (hsv_h >= CFG["cloth_h_min"]) &
        (hsv_h <= CFG["cloth_h_max"]) &
        (hsv_s >= CFG["cloth_s_min"])
    )
    tissue = bright & ~blue_cloth
    tissue_u8 = (tissue.astype(np.uint8) * 255)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))#open kernel erodes then dilates
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))#dilates then erodes

    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_OPEN, k_open)#note we should always remove noise first
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, k_close, iterations=2)
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CFG["tissue_dilate_kernel"], CFG["tissue_dilate_kernel"]),
    )
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, k)
    tissue_u8 = cv2.dilate(tissue_u8, k, iterations=1)
    return tissue_u8




def segment_dark_marker(img_bgr):
    """
    Segment marker-like pixels and bridge dashed gaps.
    Primary mode: purple/blue marker detection.
    Fallback mode: dark-ink detection (for black marker datasets - this awaits testing and validation!)
    Returns uint8 mask in {0,255}.
    """
    img = enhance_contrast_bgr(img_bgr)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l = lab[:, :, 0]
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    #tissue_roi = build_tissue_roi(l, v, h)
    tissue_roi = build_tissue_roi(l, v, h, s)


    # Primary: purple/blue marker extraction
    mask_color = (
        (h >= CFG["marker_h_min"]) & (h <= CFG["marker_h_max"]) &
        (s >= CFG["marker_s_min"]) & (v >= CFG["marker_v_min"])
    )#creates a binary 01 array of whole img
    mask_color = (mask_color.astype(np.uint8) * 255)#creates binary 0,255 img used by cv2 mod
    mask_color = cv2.bitwise_and(mask_color, tissue_roi)

    # Fallback: dark marker extraction if color cue is weak

    # mask_dark = (
    #     (l <= CFG["lab_l_max"]) &
    #     (v <= CFG["hsv_v_max"]) &
    #     (s >= CFG["hsv_s_min"])
    # )
    mask_dark = (
        # (v <= 120) &
        # (s >= 130) &
        # (h >= 15)


        #((v <= 120) & (s >= 140) & (h >= 15) & (l <= 70)) | ((v <= 35) & (s >= 100))
        #((v <= 120) & (s >= 140) & (h >= 15) & (l <= 50)) | ((v <= 35) & (s >= 130))
        #((v <= 120) & (s >= 170) & (h >= 60) & (l <= 45)) | ((v <= 35) & (s >= 140))
        #((v <= 120) & (s >= 190) & (h >= 15)  & (l <= 70))
        #v th = 100, s recomm 140 and below 190, h threshold between 100 and 150, lgreater than 60
        #((v>=15) & (v <= 100) & (s >= 190) & (h<=180) & (h >= 100)  & (l <= 60) & (l>=5))
        ((v <= 100) & (s >= 190) & (h >= 150)  & (l <= 60))
    )
    mask_dark = (mask_dark.astype(np.uint8) * 255)
    mask_dark = cv2.bitwise_and(mask_dark, tissue_roi)
    print(f"  [dark fallback] mask_dark pixels: {np.count_nonzero(mask_dark)}")

    if (#CFG["use_color_marker"] and
            int(np.count_nonzero(mask_color)) >= CFG["min_color_pixels"]):
        mask = mask_color
        print(img_bgr.shape[0], img_bgr.shape[1])
        print(mask_color.shape[0], mask_color.shape[1])
        print('mask_color')
        print(np.count_nonzero(mask_color))
    else:
        mask = mask_dark
        print(img_bgr.shape[0], img_bgr.shape[1])
        print(mask_dark.shape[0], mask_dark.shape[1])
        print('mask_dark')
        print(np.count_nonzero(mask_dark))

    # Morphological open to denoise, close to bridge dashed/dotted segments.
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG["open_kernel"], CFG["open_kernel"]))#open kernel erodes then dilates
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG["close_kernel"], CFG["close_kernel"]))#dilates then erodes

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)#note we should always remove noise first
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=CFG["close_iters"])

    mask = filter_components(mask, CFG["min_component_area"], CFG["max_component_area_ratio"])
    return mask


def skeleton_to_graph(skel_bool):#a binary img only of center pixels of each component
    """
    Build an 8-neighbor undirected graph from a boolean skeleton image.
    Nodes are (y, x) tuples.
    """
    G = nx.Graph()#creates undirected graph
    ys, xs = np.where(skel_bool)#get coordinates of center pixel
    points = set(zip(ys, xs))#convert to set of coordinated points, fast look-up

    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)]
    #this loop creates edge with weight of actual distance between center of skeletons that are very close, 1 pixel layer apart
    for y, x in points:
        G.add_node((y, x))
        for dy, dx in nbrs:
            ny, nx_ = y + dy, x + dx
            if (ny, nx_) in points:
                w = np.sqrt(2.0) if (dy != 0 and dx != 0) else 1.0
                G.add_edge((y, x), (ny, nx_), weight=w)
    return G


def prune_short_branches(G, min_len=10):
    """
    Iteratively remove short branches attached to junctions(less than 10 nodes away).
    A branch is traced from an endpoint (deg=1) until first node with deg > 2 (branch).
    """
    changed = True
    while changed:
        changed = False
        deg = dict(G.degree())
        endpoints = [n for n, d in deg.items() if d == 1]

        for ep in endpoints:
            if ep not in G:
                continue
            path = [ep]
            prev = None
            cur = ep


            while True:
                nbrs = [n for n in G.neighbors(cur) if n != prev]
                if len(nbrs) == 0:#stop if dead end
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if G.degree(cur) != 2:
                    break

            # Remove only if this branch ends at a junction and is short.
            if len(path) <= min_len and G.degree(path[-1]) >= 3:
                for node in path[:-1]:
                    if node in G:
                        G.remove_node(node)
                changed = True
                break

    return G


def _extract_longest_path_from_graph(skel_bool):
    """Fallback: longest path inside one connected skeleton graph."""
    h, w = skel_bool.shape
    G = skeleton_to_graph(skel_bool)
    if G.number_of_nodes() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    G = prune_short_branches(G, min_len=CFG["prune_branch_length"])
    if G.number_of_nodes() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()

    deg = dict(H.degree())
    endpoints = [n for n, d in deg.items() if d == 1]

    if len(endpoints) >= 2:
        best_path = None
        best_dist = -1.0
        for i in range(len(endpoints)):
            source = endpoints[i]
            dists, paths = nx.single_source_dijkstra(H, source=source, weight="weight")
            for j in range(i + 1, len(endpoints)):
                target = endpoints[j]
                if target in dists and dists[target] > best_dist:
                    best_dist = dists[target]
                    best_path = paths[target]
    else:
        seed = next(iter(H.nodes()))
        d1, _ = nx.single_source_dijkstra(H, source=seed, weight="weight")
        u = max(d1, key=d1.get)
        d2, p2 = nx.single_source_dijkstra(H, source=u, weight="weight")
        v = max(d2, key=d2.get)
        best_path = p2[v]

    out = np.zeros((h, w), dtype=np.uint8)
    if best_path is None or len(best_path) < 2:
        return out

    for (y1, x1), (y2, x2) in zip(best_path[:-1], best_path[1:]):
        cv2.line(out, (x1, y1), (x2, y2), 255, thickness=CFG["line_thickness"])
    return out


# def _extract_component_anchors(binary_mask):
#     """Return component anchors with label ids for dotted-fragment linking."""
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
#     anchors = []
#     for lid in range(1, num_labels):
#         area = stats[lid, cv2.CC_STAT_AREA]
#         if area < CFG["min_component_area"]:
#             continue
#
#         ys, xs = np.where(labels == lid)
#         if len(xs) == 0:
#             continue
#
#         cx, cy = centroids[lid]
#         d2 = (xs - cx) ** 2 + (ys - cy) ** 2
#         k = int(np.argmin(d2))
#         ax, ay = float(xs[k]), float(ys[k])
#
#         anchors.append({
#             "label": lid,
#             "point": np.array([ax, ay], dtype=np.float32),  # x, y
#             "area": int(area),
#         })
#
#     return anchors, labels


def _trim_outlier_links(chain, outlier_ratio):
      """
      Remove endpoint blobs whose connecting link is an outlier vs. the reference distance.
      Reference is P75 of links where BOTH endpoints have area >= 25% of the 5th-largest blob
      (filters tiny noise blobs that artificially lower the median).
      A floor of max_link_dist * trim_ref_floor_ratio prevents over-trimming on tight chains.
      """
      if len(chain) < 3:
          return chain

      pts   = np.array([c["point"] for c in chain], dtype=np.float64)
      areas = np.array([c["area"]  for c in chain], dtype=np.float64)

      while len(chain) >= 3:
          dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)

          sorted_areas = np.sort(areas)[::-1]
          ref_area     = sorted_areas[min(4, len(sorted_areas) - 1)]
          sig_thresh   = ref_area * 0.25
          sig_mask     = (areas[:-1] >= sig_thresh) & (areas[1:] >= sig_thresh)
          ref_dists    = dists[sig_mask] if sig_mask.sum() >= 2 else dists
          med          = float(np.percentile(ref_dists, 75))

          floor = CFG["max_link_dist"] * CFG.get("trim_ref_floor_ratio", 0.30)
          med   = max(med, floor)

          bad = np.where(dists > outlier_ratio * med)[0]
          if len(bad) == 0:
              break
          split  = int(bad[0]) + 1
          piece1, piece2 = chain[:split], chain[split:]
          chain  = piece1 if len(piece1) >= len(piece2) else piece2
          pts    = np.array([c["point"] for c in chain], dtype=np.float64)
          areas  = np.array([c["area"]  for c in chain], dtype=np.float64)
      return chain


# def extract_single_path_mask(binary_mask):
#     """
#     Convert mask -> longest smooth chain over dotted fragments -> single path mask.
#     Fallback to skeleton graph longest path when chain extraction is weak.
#     """
#     h, w = binary_mask.shape
#     skel = skeletonize(binary_mask > 0)
#     skel_vis = skel.astype(np.uint8) * 255
#
#     anchors, labels = _extract_component_anchors(binary_mask)
#     chain = _longest_smooth_chain(anchors)
#
#     # If chain mode fails, fallback to prior graph method.
#     if len(chain) < 2:
#         return _extract_longest_path_from_graph(skel), skel_vis
#
#     out = np.zeros((h, w), dtype=np.uint8)
#
#     # Keep selected components and connect them to form one non-branching long path.
#     selected_labels = [c["label"] for c in chain]
#     keep = np.isin(labels, selected_labels)
#     out[keep] = 255
#
#     for a, b in zip(chain[:-1], chain[1:]):
#         x1, y1 = a["point"]
#         x2, y2 = b["point"]
#         cv2.line(
#             out,
#             (int(round(x1)), int(round(y1))),
#             (int(round(x2)), int(round(y2))),
#             255,
#             thickness=CFG["link_thickness"],
#         )
#
#     # Light cleanup and thickness normalization.
#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
#
#     # Finalize by reducing to one path inside this connected result.
#     out_skel = skeletonize(out > 0)
#     out = _extract_longest_path_from_graph(out_skel)
#
#     return out, skel_vis

def extract_single_path_mask(binary_mask):
      h, w = binary_mask.shape
      skel = skeletonize(binary_mask > 0)
      skel_vis = skel.astype(np.uint8) * 255

      anchors, labels = _extract_component_anchors_with_features(binary_mask)
      anchors = _suppress_shadow_components(anchors, total_px=h * w, img_shape=(h, w))
      chain   = _direction_constrained_chain(
                    anchors,
                    max_link_dist=CFG["max_link_dist"],
                    max_angle_deg=CFG["chain_max_angle_deg"])
      chain   = _trim_outlier_links(chain, CFG["chain_outlier_ratio"])

      if len(chain) < 2:
          return _extract_longest_path_from_graph(skel), skel_vis

      out = np.zeros((h, w), dtype=np.uint8)
      selected_labels = [c["label"] for c in chain]
      out[np.isin(labels, selected_labels)] = 255

      for a, b in zip(chain[:-1], chain[1:]):
          x1, y1 = a["point"];  x2, y2 = b["point"]
          cv2.line(out,
                   (int(round(x1)), int(round(y1))),
                   (int(round(x2)), int(round(y2))),
                   255, thickness=CFG["link_thickness"])

      k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
      out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
      out_skel = skeletonize(out > 0)
      return _extract_longest_path_from_graph(out_skel), skel_vis


def _compute_principal_axis(points_xy):
    """Compute dominant direction unit vector and its perpendicular."""
    pts = np.asarray(points_xy, dtype=np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = centered.T @ centered / max(1, len(pts) - 1)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, int(np.argmax(vals))]
    u = u / (np.linalg.norm(u) + 1e-8)
    n = np.array([-u[1], u[0]], dtype=np.float32)
    return mean, u.astype(np.float32), n




# def _longest_smooth_chain(anchors):
#     """
#     Longest path DP over component anchors sorted on dominant axis.
#     Enforces no branching and rejects sudden jumps with geometric constraints.
#     """
#     if len(anchors) < 2:
#         return []
#
#     pts = np.array([a["point"] for a in anchors], dtype=np.float32)
#     mean, u, n = _compute_principal_axis(pts)
#
#     # Sort anchors along principal direction.
#     proj = (pts - mean) @ u
#     order = np.argsort(proj)
#     pts_s = pts[order]
#     anchors_s = [anchors[int(i)] for i in order]
#
#     dp = np.zeros(len(anchors_s), dtype=np.float32)
#     prev = np.full(len(anchors_s), -1, dtype=np.int32)
#
#     for i in range(len(anchors_s)):
#         best_score = 0.0
#         best_prev = -1
#
#         for j in range(i):
#             dxy = pts_s[i] - pts_s[j]
#             dist = float(np.linalg.norm(dxy))
#             if dist < 1e-6:
#                 continue
#
#             dt = float(np.dot(dxy, u))
#             lat = float(abs(np.dot(dxy, n)))
#
#             # Reject backward links and abrupt jumps.
#             if dt <= 0:
#                 continue
#             if dist > CFG["max_link_dist"]:
#                 continue
#             if dt > CFG["max_proj_gap"]:
#                 continue
#             if lat > CFG["max_lateral_dev"]:
#                 continue
#
#             cand = dp[j] + dist
#             if cand > best_score:
#                 best_score = cand
#                 best_prev = j
#
#         dp[i] = best_score
#         prev[i] = best_prev
#
#     end_idx = int(np.argmax(dp))
#     chain = []
#     cur = end_idx
#     while cur >= 0:
#         chain.append(cur)
#         cur = int(prev[cur])
#     chain = chain[::-1]
#
#     if len(chain) < CFG["min_chain_nodes"]:
#         return []
#
#     return [anchors_s[i] for i in chain]


def _extract_component_anchors_with_features(binary_mask):
      """Like _extract_component_anchors but also returns elongation per component."""
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
      anchors = []
      for lid in range(1, num_labels):
          area = stats[lid, cv2.CC_STAT_AREA]
          if area < CFG["min_component_area"]:
              continue
          cx, cy = centroids[lid]
          ys, xs = np.where(labels == lid)
          if len(xs) == 0:
              continue

          # Closest pixel to centroid
          d2 = (xs - cx)**2 + (ys - cy)**2
          k = int(np.argmin(d2))
          ax, ay = float(xs[k]), float(ys[k])

          # Elongation via 2nd-order central moments
          comp_u8 = (labels == lid).astype(np.uint8)
          M = cv2.moments(comp_u8)
          if M["m00"] > 0:
              a_m = M["mu20"] / M["m00"]
              b_m = M["mu11"] / M["m00"]
              c_m = M["mu02"] / M["m00"]
              disc = np.sqrt(max(((a_m - c_m) / 2)**2 + b_m**2, 0))
              eig1 = (a_m + c_m) / 2 + disc
              eig2 = (a_m + c_m) / 2 - disc
              elongation = float(np.sqrt(eig1 / max(eig2, 1e-4)))
          else:
              elongation = 1.0

          # Convex hull solidity: compact ink blobs are solid, merged shadow blobs are not
          contours_c, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          if contours_c:
              hull_area = float(cv2.contourArea(cv2.convexHull(contours_c[0])))
              solidity = float(area) / hull_area if hull_area > 0 else 1.0
          else:
              solidity = 1.0

          anchors.append({
              "label": lid,
              "point": np.array([ax, ay], dtype=np.float32),
              "area": int(area),
              "elongation": elongation,
              "solidity": solidity,
          })
      return anchors, labels

# def _suppress_shadow_components(anchors, total_px=None, img_shape=None):
#       """
#       Remove shadow artifacts from component list.
#       Type A: small component surrounded by many other small components (dot cluster).
#       Type B: large blob with low elongation (irregular shadow mass).
#       """
#       if len(anchors) <= 1:
#           return anchors
#
#       areas = np.array([a["area"] for a in anchors], dtype=np.float32)
#       pts  = np.array([a["point"] for a in anchors], dtype=np.float32)
#       elongs = np.array([a["elongation"] for a in anchors], dtype=np.float32)
#
#       area_median  = float(np.median(areas))
#       small_thresh = area_median * CFG["shadow_small_ratio"]   # default 2.0
#       large_thresh = area_median * CFG["shadow_large_ratio"]   # default 4.0
#
#       kept = []
#       for i, anchor in enumerate(anchors):
#           area      = areas[i]
#           elongation = elongs[i]
#
#           # Type B: large irregular blob
#           if area > large_thresh and elongation < CFG["shadow_elongation_min"]:  # default 2.5
#               continue
#
#           # Type A: small component in a dense cluster of other small components
#           if area < small_thresh:
#               dists = np.linalg.norm(pts - pts[i], axis=1)
#               nearby_small = int(np.sum(
#                   (dists < CFG["shadow_density_radius"]) &   # default 250 px
#                   (areas < small_thresh) &
#                   (np.arange(len(anchors)) != i)
#               ))
#               if nearby_small >= CFG["shadow_density_count"]:  # default 5
#                   continue
#
#           kept.append(anchor)
#       return kept

def _suppress_shadow_components(anchors, total_px=None, img_shape=None):
      """
      Remove shadow artifacts using a neighborhood linearity test (replaces Type A/B).
      A component is shadow if it has >= shadow_density_count neighbors within
      shadow_density_radius AND those neighbors form a 2D cluster (low PCA linearity).
      - Ink dots along a trajectory have a LINEAR neighborhood  -> high linearity -> kept.
      - Shadow dots/blobs have a 2D circular neighborhood       -> low linearity  -> removed.
      This correctly handles:
        * large central mass of a shadow cluster (its neighbors are the surrounding dots, 2D)
        * small scattered shadow dots (their neighbors include the mass and other dots, 2D)
        * large ink blobs near the trajectory (neighbors are other trajectory dots, linear)
      """
      if len(anchors) <= 1:
          return anchors

      pts = np.array([a["point"] for a in anchors], dtype=np.float32)
      N   = len(pts)
      is_shadow = np.zeros(N, dtype=bool)
      shadow_reason = [""] * N

      areas      = np.array([a["area"]     for a in anchors], dtype=np.float32)
      solidities = np.array([a.get("solidity", 1.0) for a in anchors], dtype=np.float32)
      area_median  = float(np.median(areas))
      large_thresh = area_median * CFG["shadow_large_ratio"]

      border_margin    = CFG.get("border_margin_px", 0)
      border_max_area  = CFG.get("border_margin_max_area", 1e9)
      cluster_area_thr = (total_px * CFG.get("shadow_cluster_area_ratio", 0.0)
                          if total_px is not None else 0.0)

      for i in range(N):
          # Border exclusion: only SMALL blobs near the image edge are instrument
          # artifacts. Valid ink marks near the border (e.g. clip position) are
          # larger and must not be suppressed.
          if img_shape is not None and border_margin > 0 and areas[i] <= border_max_area:
              ih, iw = img_shape
              px, py = float(pts[i][0]), float(pts[i][1])
              if min(px, iw - px, py, ih - py) < border_margin:
                  is_shadow[i] = True; shadow_reason[i] = "border"; continue

          # Type B0 (absolute area): any single blob larger than X% of the full frame is
          # definitely a shadow or large artifact — real ink blobs are always small relative
          # to the frame. Independent of median or solidity; catches the 38.54 merged shadow
          # (1.55% of frame) which has solidity 0.445 and therefore escapes Type B1.
          if total_px is not None and areas[i] > total_px * CFG["shadow_abs_area_ratio"]:
              is_shadow[i] = True; shadow_reason[i] = "B0-abs"
              continue

          # Cluster-area check: if the total area of all blobs within
          # shadow_density_radius collectively exceeds shadow_cluster_area_ratio *
          # total_px, treat as shadow (catches fragmented shadow regions whose
          # individual components each pass the per-blob B0 threshold).
          if cluster_area_thr > 0:
              dists_cl = np.linalg.norm(pts - pts[i], axis=1)
              cluster_mask = dists_cl < CFG["shadow_density_radius"]
              if cluster_mask.sum() >= CFG["shadow_density_count"]:
                  cluster_total = float(np.sum(areas[cluster_mask]))
                  if cluster_total > cluster_area_thr:
                      is_shadow[i] = True; shadow_reason[i] = "cluster-area"; continue

          # Type B1 (strict): very large AND very irregular blob => definitely a merged shadow.
          # When morphological closing fuses many scattered shadow dots into one big blob,
          # no tiny neighbors remain, so the combined check (B2) misses it.
          # Solidity of a fused-dots blob is very low (convex hull >> filled area).
          # Ink blobs are compact and continuous, so their solidity stays high.
          if areas[i] > area_median * CFG["shadow_strict_large_ratio"] and solidities[i] < CFG["shadow_strict_solidity"]:
              is_shadow[i] = True; shadow_reason[i] = "B1"
              continue

          # Type B2 (combined): large + moderately irregular + surrounded by tiny scattered dots.
          # Both conditions must hold: ink blobs can be large or irregular, but they do NOT
          # have many small dots closely around them the way shadow regions do.
          # (Old B used only area+solidity, which incorrectly filtered narrow ink blobs.)
          # if areas[i] > large_thresh and solidities[i] < CFG["shadow_solidity_min"]:
          #     is_shadow[i] = True
          #     continue
          if areas[i] > large_thresh and solidities[i] < CFG["shadow_solidity_min"]:
              dists_i = np.linalg.norm(pts - pts[i], axis=1)
              n_tiny_nearby = int(np.sum(
                  (dists_i < CFG["shadow_density_radius"]) &
                  (areas < area_median) &
                  (np.arange(N) != i)
              ))
              if n_tiny_nearby >= CFG["shadow_density_count"] - 1:
                  is_shadow[i] = True; shadow_reason[i] = "B2"
                  continue

          dists      = np.linalg.norm(pts - pts[i], axis=1)
          # Filter neighbourhood to significant blobs (area >= area_median * 0.25).
          # Tiny noise dots near a trajectory bend drag the PCA cluster toward 2-D,
          # wrongly suppressing a valid ink blob sitting at that bend.
          lin_area_thresh = area_median * 0.25
          nearby_idx = np.where(
              (dists < CFG["shadow_density_radius"]) &
              (np.arange(N) != i) &
              (areas >= lin_area_thresh)
          )[0]

          if len(nearby_idx) < CFG["shadow_density_count"] - 1:
              continue          # too few neighbors to form a meaningful cluster

          # PCA linearity: ratio of largest to second eigenvalue of the point cloud
          group_pts = pts[np.append(nearby_idx, i)]
          centered  = group_pts - group_pts.mean(axis=0)
          cov       = centered.T @ centered
          vals      = np.sort(np.linalg.eigvalsh(cov))[::-1]   # descending
          linearity = vals[0] / (vals[1] + 1e-6)

          if linearity < CFG["shadow_linearity_threshold"]:
              is_shadow[i] = True; shadow_reason[i] = f"linearity={linearity:.1f}"

      # -- Pass 2: re-test small linearity-suppressed blobs ---------------------
      # A blob may be falsely suppressed in pass 1 because a large shadow blob
      # (e.g. blob#43 in 38.54) was in its neighbourhood making it look 2D.
      # After pass 1, that large shadow is suppressed; re-run the linearity test
      # for small ink dots (area < 2xmedian) excluding now-suppressed neighbours.
      small_thresh_p2 = area_median * 2.0
      for i in range(N):
          if not is_shadow[i]:
              continue
          if "linearity" not in shadow_reason[i]:
              continue
          if areas[i] > small_thresh_p2:
              continue
          lin_area_thresh = area_median * 0.25
          dists = np.linalg.norm(pts - pts[i], axis=1)
          # Only un-suppress if a large non-suppressed "anchor" blob is nearby.
          # This prevents floating shadow clusters (far from any real trajectory blob)
          # from being incorrectly freed in images like 8.47 and 7.56.
          anchor_thresh = area_median * 2.0
          anchor_mask = (~is_shadow) & (areas >= anchor_thresh) & (np.arange(N) != i)
          if not np.any((dists < CFG["max_link_dist"]) & anchor_mask):
              continue  # no large keep blob nearby -> leave as shadow
          nearby_idx = np.where(
              (dists < CFG["shadow_density_radius"]) &
              (~is_shadow) &                           # exclude pass-1 shadows
              (np.arange(N) != i) &
              (areas >= lin_area_thresh)
          )[0]
          if len(nearby_idx) < CFG["shadow_density_count"] - 1:
              # Too few non-shadow neighbours -> un-suppress
              is_shadow[i] = False
              shadow_reason[i] = ""
              continue
          group_pts = pts[np.append(nearby_idx, i)]
          centered  = group_pts - group_pts.mean(axis=0)
          cov       = centered.T @ centered
          vals      = np.sort(np.linalg.eigvalsh(cov))[::-1]
          linearity = vals[0] / (vals[1] + 1e-6)
          if linearity >= CFG["shadow_linearity_threshold"] * 0.875:
              is_shadow[i] = False
              shadow_reason[i] = ""
      # -------------------------------------------------------------------------

      return [a for i, a in enumerate(anchors) if not is_shadow[i]]


def _direction_constrained_chain(anchors, max_link_dist, max_angle_deg):
      """
      Longest chain with direction-continuity constraint.
      Handles curved paths (U-shapes, arcs) by tracking local direction.
      O(N^3) via Bellman-Ford relaxation.
      """
      if len(anchors) < 2:
          return anchors

      pts = np.array([a["point"] for a in anchors], dtype=np.float64)
      N = len(pts)
      cos_thresh = np.cos(np.radians(max_angle_deg))

      # Pairwise: vec[i,j] = direction i->j (unnorm), dist_mat[i,j] = distance
      vec      = pts[None, :, :] - pts[:, None, :]            # (N,N,2)
      dist_mat = np.linalg.norm(vec, axis=2)                  # (N,N)
      with np.errstate(invalid='ignore', divide='ignore'):
          uv = vec / (dist_mat[:, :, None] + 1e-8)            # unit vectors (N,N,2)

      link_ok = (dist_mat > 1) & (dist_mat <= max_link_dist)  # valid links (N,N)

      # dp[i,j]   = best total distance for chain ending at i, last step was j->i
      # back[i,j] = which k was before j in the best chain (i.e., k->j->i)
      # Old: dp   = np.where(link_ok.T, dist_mat.T, -1.0)            # init all single links
      # Old: HOP_BONUS = CFG["max_link_dist"]           # ensures node count beats raw distance
      # Old: dp   = np.where(link_ok.T, dist_mat.T + HOP_BONUS, -1.0)   # flat hop bonus
      areas_arr  = np.array([a["area"] for a in anchors], dtype=np.float64)
      NODE_BONUS = float(np.median(areas_arr))          # median area: large blobs beat many tiny ones
      # Cap per-node area at 4*NODE_BONUS: prevents extreme outlier blobs from dominating score
      # while still giving large legitimate blobs an advantage over tiny noise blobs.
      capped_areas = np.minimum(areas_arr, NODE_BONUS * 4)
      dp   = np.where(link_ok.T, dist_mat.T * 0.01 + NODE_BONUS + capped_areas[:, np.newaxis], -1.0)
      back = np.full((N, N), -1, dtype=np.int32)

      for _ in range(N - 2):          # at most N-2 extension steps
          improved = False
          new_dp   = dp.copy()
          new_back = back.copy()

          for j in range(N):
              for i in range(N):
                  if dp[i, j] < 0:
                      continue
                  dir_in = uv[j, i]                           # direction arriving at i

                  # Vectorised check for all outgoing links i->k
                  dots = uv[i] @ dir_in                       # (N,) cosines with i->k
                  ok   = link_ok[i] & (dots >= cos_thresh)
                  ok[i] = ok[j] = False

                  # Old: cands  = dp[i, j] + dist_mat[i]             # (N,)
                  # Old: cands  = dp[i, j] + dist_mat[i] + HOP_BONUS    # flat hop bonus
                  cands  = dp[i, j] + dist_mat[i] * 0.01 + NODE_BONUS + capped_areas  # area-weighted, capped
                  better = ok & (cands > new_dp[:, i])
                  new_dp[:, i]   = np.where(better, cands, new_dp[:, i])
                  new_back[:, i] = np.where(better, j, new_back[:, i])
                  improved |= bool(np.any(better))

          dp, back = new_dp, new_back
          if not improved:
              break

      if dp.max() < 0:
          return []

      # Best terminal state
      end_i, end_j = map(int, np.unravel_index(np.argmax(dp), dp.shape))

      # Backtrack
      path = [end_i, end_j]
      ci, cj = end_i, end_j
      while back[ci, cj] >= 0:
          pk = int(back[ci, cj])
          path.append(pk)
          ci, cj = cj, pk
      path.reverse()

      # Deduplicate (preserving order)
      seen, unique = set(), []
      for idx in path:
          if idx not in seen:
              seen.add(idx)
              unique.append(idx)

      if len(unique) < CFG["min_chain_nodes"]:
          return []
      return [anchors[i] for i in unique]


def run_pipeline(image_bgr):
    raw_mask = segment_dark_marker(image_bgr)
    single_path_mask, skeleton_vis = extract_single_path_mask(raw_mask)
    return {
        "raw_mask": raw_mask,
        "skeleton": skeleton_vis,
        "single_path_mask": single_path_mask,
    }


print("Core functions ready.")"""

nb['cells'][3]['source'] = cell3_new

with open('dissection_marker_to_single_path_mask.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook saved successfully.")
