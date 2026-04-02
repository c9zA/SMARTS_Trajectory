"""
dissection_extractor.py
=======================
Standalone, self-contained module for dissection trajectory extraction.
No dependency on run_test.py — all pipeline logic lives here as class methods.

Import usage
------------
    from dissection_extractor import DissectionTrajectoryExtractor
    import cv2

    extractor = DissectionTrajectoryExtractor()
    mask = extractor(cv2.imread("image.png"))           # uint8 H x W, path = 255
    raw, skel, mask = extractor.extract_all(img_bgr)    # all three outputs

Command-line usage
------------------
    python dissection_extractor.py input.png output.png
    python dissection_extractor.py input.png output.png --marker_mode dark
    python dissection_extractor.py input.png output.png --max_link_dist 200

All parameters listed in --help.
"""

import argparse
import pathlib
import sys

import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx


class DissectionTrajectoryExtractor:
    """Extract a single white trajectory line from a dissection image.

    Usage
    -----
    extractor = DissectionTrajectoryExtractor()          # all defaults
    mask = extractor(image_bgr)                          # uint8 H×W, path=255
    raw, skel, mask = extractor.extract_all(image_bgr)  # all three outputs

    All constructor parameters mirror CFG keys.  Passing a value overrides the
    default for that instance only.
    """

    def __init__(
        self,
        # --- marker mode ---
        marker_mode                 = "auto",
        # --- color marker mask ---
        marker_h_min                = 112,
        marker_h_max                = 160,
        marker_s_min                = 90,
        marker_v_min                = 40,
        min_color_pixels            = 10000,
        # --- light marker mask ---
        light_v_min                 = 200,
        light_s_max                 = 60,
        # --- MV-D dark/light auto toggle ---
        mv_votes_needed             = 2,
        mv_ratio_threshold          = 0.5748,
        mv_narrow_threshold         = 147,
        mv_narrow_ratio_threshold   = 2.402,
        mv_white_threshold          = 188112,
        # --- tissue ROI ---
        use_clahe                   = True,
        tissue_v_min                = 40,
        tissue_l_min                = 40,
        cloth_h_min                 = 85,
        cloth_h_max                 = 135,
        cloth_s_min                 = 25,
        tissue_dilate_kernel        = 40,
        # --- morphological cleanup ---
        open_kernel                 = 2,
        close_kernel                = 11,
        close_iters                 = 1,
        min_component_area          = 25,
        max_component_area_ratio    = 0.03,
        prune_branch_length         = 10,
        line_thickness              = 3,
        link_thickness              = 4,
        min_chain_nodes             = 2,
        # --- chain / DP ---
        max_link_dist               = 165.0,
        # --- border exclusion ---
        border_margin_px            = 30,
        border_margin_max_area      = 300,
        right_border_margin_px      = 0,
        bottom_border_margin_px     = 0,
        # --- tiny blob suppression ---
        tiny_blob_req_anchor_ratio  = 0.0,
        tiny_blob_anchor_large_ratio= 2.0,
        # --- shadow suppression ---
        shadow_density_radius       = 250,
        shadow_linearity_radius     = 200,
        shadow_density_count        = 4,
        shadow_linearity_threshold  = 4.0,
        shadow_large_ratio          = 8.0,
        shadow_solidity_min         = 0.65,
        shadow_strict_large_ratio   = 6.0,
        shadow_strict_solidity      = 0.55,
        b1_large_self_ratio         = 0.12,
        shadow_abs_area_ratio       = 0.008,
        shadow_cluster_area_ratio   = 0.020,
        # --- direction / chain ---
        chain_max_angle_deg         = 55,
        ext_max_angle_deg           = 65,
        chain_outlier_ratio         = 3.0,
        trim_ref_floor_ratio        = 0.30,
        chain_endpoint_ext_dist     = 380,
        # --- sparkle filter ---
        sparkle_filter_ratio        = 0.110,
        sparkle_loop_ratio          = 0.35,
        sparkle_sig_max_link_dist   = 320,
        dp_link_scale               = 8,
        ext_loop_angle_deg          = 90,
        # --- endpoint trimming ---
        endpoint_trim_angle_deg     = 50,
        endpoint_trim_adj_ratio     = 0.04,
        endpoint_trim_adj_min_area  = 100,
        endpoint_trim_median_ratio  = 0.03,
        endpoint_trim_long_link_ratio = 1.5,
        ep_large_area_thr           = 150,
        ep_small_max_dist           = 130,
        # --- extension ---
        ext_min_area_ratio          = 0.05,
        ext_ep_scale                = 1.0,
        # --- post-extension pruning ---
        ext_seg_prune_dist          = 218,
        ext_seg_min_blobs           = 3,
        ext_seg_protect_large_area  = 500,
        # --- bridge-node pruning ---
        bridge_min_area_ratio       = 0.15,
        bridge_min_skip_dist        = 200.0,
    ):
        self.marker_mode                  = marker_mode
        self.marker_h_min                 = marker_h_min
        self.marker_h_max                 = marker_h_max
        self.marker_s_min                 = marker_s_min
        self.marker_v_min                 = marker_v_min
        self.min_color_pixels             = min_color_pixels
        self.light_v_min                  = light_v_min
        self.light_s_max                  = light_s_max
        self.mv_votes_needed              = mv_votes_needed
        self.mv_ratio_threshold           = mv_ratio_threshold
        self.mv_narrow_threshold          = mv_narrow_threshold
        self.mv_narrow_ratio_threshold    = mv_narrow_ratio_threshold
        self.mv_white_threshold           = mv_white_threshold
        self.use_clahe                    = use_clahe
        self.tissue_v_min                 = tissue_v_min
        self.tissue_l_min                 = tissue_l_min
        self.cloth_h_min                  = cloth_h_min
        self.cloth_h_max                  = cloth_h_max
        self.cloth_s_min                  = cloth_s_min
        self.tissue_dilate_kernel         = tissue_dilate_kernel
        self.open_kernel                  = open_kernel
        self.close_kernel                 = close_kernel
        self.close_iters                  = close_iters
        self.min_component_area           = min_component_area
        self.max_component_area_ratio     = max_component_area_ratio
        self.prune_branch_length          = prune_branch_length
        self.line_thickness               = line_thickness
        self.link_thickness               = link_thickness
        self.min_chain_nodes              = min_chain_nodes
        self.max_link_dist                = max_link_dist
        self.border_margin_px             = border_margin_px
        self.border_margin_max_area       = border_margin_max_area
        self.right_border_margin_px       = right_border_margin_px
        self.bottom_border_margin_px      = bottom_border_margin_px
        self.tiny_blob_req_anchor_ratio   = tiny_blob_req_anchor_ratio
        self.tiny_blob_anchor_large_ratio = tiny_blob_anchor_large_ratio
        self.shadow_density_radius        = shadow_density_radius
        self.shadow_linearity_radius      = shadow_linearity_radius
        self.shadow_density_count         = shadow_density_count
        self.shadow_linearity_threshold   = shadow_linearity_threshold
        self.shadow_large_ratio           = shadow_large_ratio
        self.shadow_solidity_min          = shadow_solidity_min
        self.shadow_strict_large_ratio    = shadow_strict_large_ratio
        self.shadow_strict_solidity       = shadow_strict_solidity
        self.b1_large_self_ratio          = b1_large_self_ratio
        self.shadow_abs_area_ratio        = shadow_abs_area_ratio
        self.shadow_cluster_area_ratio    = shadow_cluster_area_ratio
        self.chain_max_angle_deg          = chain_max_angle_deg
        self.ext_max_angle_deg            = ext_max_angle_deg
        self.chain_outlier_ratio          = chain_outlier_ratio
        self.trim_ref_floor_ratio         = trim_ref_floor_ratio
        self.chain_endpoint_ext_dist      = chain_endpoint_ext_dist
        self.sparkle_filter_ratio         = sparkle_filter_ratio
        self.sparkle_loop_ratio           = sparkle_loop_ratio
        self.sparkle_sig_max_link_dist    = sparkle_sig_max_link_dist
        self.dp_link_scale                = dp_link_scale
        self.ext_loop_angle_deg           = ext_loop_angle_deg
        self.endpoint_trim_angle_deg      = endpoint_trim_angle_deg
        self.endpoint_trim_adj_ratio      = endpoint_trim_adj_ratio
        self.endpoint_trim_adj_min_area   = endpoint_trim_adj_min_area
        self.endpoint_trim_median_ratio   = endpoint_trim_median_ratio
        self.endpoint_trim_long_link_ratio = endpoint_trim_long_link_ratio
        self.ep_large_area_thr            = ep_large_area_thr
        self.ep_small_max_dist            = ep_small_max_dist
        self.ext_min_area_ratio           = ext_min_area_ratio
        self.ext_ep_scale                 = ext_ep_scale
        self.ext_seg_prune_dist           = ext_seg_prune_dist
        self.ext_seg_min_blobs            = ext_seg_min_blobs
        self.ext_seg_protect_large_area   = ext_seg_protect_large_area
        self.bridge_min_area_ratio        = bridge_min_area_ratio
        self.bridge_min_skip_dist         = bridge_min_skip_dist

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _enhance_contrast(self, img_bgr):
        if not self.use_clahe:
            return img_bgr
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    def _filter_components(self, binary_mask, min_area, max_area_ratio):
        h, w = binary_mask.shape
        max_area = int(max_area_ratio * h * w)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        cleaned = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                cleaned[labels == i] = 255
        return cleaned

    def _build_tissue_roi(self, lab_l, hsv_v, hsv_h, hsv_s):
        bright = (hsv_v >= self.tissue_v_min) & (lab_l >= self.tissue_l_min)
        blue_cloth = (
            (hsv_h >= self.cloth_h_min) &
            (hsv_h <= self.cloth_h_max) &
            (hsv_s >= self.cloth_s_min)
        )
        tissue = bright & ~blue_cloth
        tissue_u8 = tissue.astype(np.uint8) * 255
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_OPEN,  k_open)
        tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, k_close, iterations=2)
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.tissue_dilate_kernel, self.tissue_dilate_kernel))
        tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, k)
        tissue_u8 = cv2.dilate(tissue_u8, k, iterations=1)
        return tissue_u8

    def _segment_dark_marker(self, img_bgr):
        img = self._enhance_contrast(img_bgr)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l, h, s, v = lab[:,:,0], hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        tissue_roi = self._build_tissue_roi(l, v, h, s)

        mask_color = (
            (h >= self.marker_h_min) & (h <= self.marker_h_max) &
            (s >= self.marker_s_min) & (v >= self.marker_v_min)
        ).astype(np.uint8) * 255
        mask_color = cv2.bitwise_and(mask_color, tissue_roi)

        mask_dark = (
            (v <= 100) & (s >= 190) & (h >= 150) & (l <= 60)
        ).astype(np.uint8) * 255
        mask_dark = cv2.bitwise_and(mask_dark, tissue_roi)

        mask_light = (
            (v >= self.light_v_min) & (s <= self.light_s_max)
        ).astype(np.uint8) * 255
        mask_light = cv2.bitwise_and(mask_light, tissue_roi)

        mode = self.marker_mode
        if mode == "color":
            mask = mask_color
        elif mode == "dark":
            mask = mask_dark
        elif mode == "light":
            mask = mask_light
        else:  # "auto": color if strong; else MV-D decides dark vs light
            n_color = int(np.count_nonzero(mask_color))
            if n_color >= self.min_color_pixels:
                mask = mask_color
            elif self.mv_votes_needed is not None:
                # -- MV-D: 4 independent votes, majority decides dark vs light ----
                n_dark = int(np.count_nonzero(mask_dark))

                # narrow dark ink: H in [150,165], V<=80, S>=190, L<=60
                # (excludes dark red meat at H=165-180 that inflates n_dark)
                mask_narrow = (
                    (v <= 80) & (s >= 190) & (h >= 150) & (h <= 165) & (l <= 60)
                ).astype(np.uint8) * 255
                n_narrow = int(np.count_nonzero(cv2.bitwise_and(mask_narrow, tissue_roi)))

                # white tissue: V>=180, S<=60 (pale fascia surface)
                mask_white = (
                    (v >= self.light_v_min) & (s <= self.light_s_max)
                ).astype(np.uint8) * 255
                n_white = int(np.count_nonzero(cv2.bitwise_and(mask_white, tissue_roi)))

                ratio        = n_color / (n_dark   + 1)
                narrow_ratio = n_color / (n_narrow + 1)

                votes = 0
                if ratio        >= self.mv_ratio_threshold:        votes += 1
                if n_narrow      < self.mv_narrow_threshold:       votes += 1
                if narrow_ratio >= self.mv_narrow_ratio_threshold: votes += 1
                if n_white      >= self.mv_white_threshold:        votes += 1

                mask = mask_light if votes >= self.mv_votes_needed else mask_dark
            else:
                mask = mask_dark

        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_kernel,  self.open_kernel))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=self.close_iters)
        return self._filter_components(mask, self.min_component_area, self.max_component_area_ratio)

    def _skeleton_to_graph(self, skel_bool):
        G = nx.Graph()
        ys, xs = np.where(skel_bool)
        points = set(zip(ys, xs))
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for y, x in points:
            G.add_node((y, x))
            for dy, dx in nbrs:
                ny, nx_ = y+dy, x+dx
                if (ny, nx_) in points:
                    w = np.sqrt(2.0) if (dy != 0 and dx != 0) else 1.0
                    G.add_edge((y,x),(ny,nx_), weight=w)
        return G

    def _prune_short_branches(self, G, min_len=10):
        changed = True
        while changed:
            changed = False
            deg = dict(G.degree())
            for ep in [n for n, d in deg.items() if d == 1]:
                if ep not in G:
                    continue
                path, prev, cur = [ep], None, ep
                while True:
                    nbrs = [n for n in G.neighbors(cur) if n != prev]
                    if not nbrs:
                        break
                    nxt = nbrs[0]
                    path.append(nxt)
                    prev, cur = cur, nxt
                    if G.degree(cur) != 2:
                        break
                if len(path) <= min_len and G.degree(path[-1]) >= 3:
                    for node in path[:-1]:
                        if node in G:
                            G.remove_node(node)
                    changed = True
                    break
        return G

    def _extract_longest_path(self, skel_bool):
        h, w = skel_bool.shape
        G = self._skeleton_to_graph(skel_bool)
        if G.number_of_nodes() == 0:
            return np.zeros((h, w), dtype=np.uint8)
        G = self._prune_short_branches(G, min_len=self.prune_branch_length)
        if G.number_of_nodes() == 0:
            return np.zeros((h, w), dtype=np.uint8)
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()
        deg = dict(H.degree())
        endpoints = [n for n, d in deg.items() if d == 1]
        if len(endpoints) >= 2:
            best_path, best_dist = None, -1.0
            for i in range(len(endpoints)):
                dists, paths = nx.single_source_dijkstra(H, source=endpoints[i], weight="weight")
                for j in range(i+1, len(endpoints)):
                    t = endpoints[j]
                    if t in dists and dists[t] > best_dist:
                        best_dist, best_path = dists[t], paths[t]
        else:
            seed = next(iter(H.nodes()))
            d1, _ = nx.single_source_dijkstra(H, source=seed, weight="weight")
            u = max(d1, key=d1.get)
            d2, p2 = nx.single_source_dijkstra(H, source=u, weight="weight")
            best_path = p2[max(d2, key=d2.get)]
        out = np.zeros((h, w), dtype=np.uint8)
        if best_path is None or len(best_path) < 2:
            return out
        for (y1,x1),(y2,x2) in zip(best_path[:-1], best_path[1:]):
            cv2.line(out, (x1,y1), (x2,y2), 255, thickness=self.line_thickness)
        return out

    def _extract_anchors(self, binary_mask):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        anchors = []
        for lid in range(1, num_labels):
            area = stats[lid, cv2.CC_STAT_AREA]
            if area < self.min_component_area:
                continue
            cx, cy = centroids[lid]
            ys, xs = np.where(labels == lid)
            if len(xs) == 0:
                continue
            d2 = (xs - cx)**2 + (ys - cy)**2
            k = int(np.argmin(d2))
            ax, ay = float(xs[k]), float(ys[k])
            comp_u8 = (labels == lid).astype(np.uint8)
            M = cv2.moments(comp_u8)
            if M["m00"] > 0:
                a_m, b_m, c_m = M["mu20"]/M["m00"], M["mu11"]/M["m00"], M["mu02"]/M["m00"]
                disc = np.sqrt(max(((a_m-c_m)/2)**2 + b_m**2, 0))
                eig1 = (a_m+c_m)/2 + disc
                eig2 = (a_m+c_m)/2 - disc
                elongation = float(np.sqrt(eig1 / max(eig2, 1e-4)))
            else:
                elongation = 1.0
            contours_c, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_c:
                hull_area = float(cv2.contourArea(cv2.convexHull(contours_c[0])))
                solidity = float(area) / hull_area if hull_area > 0 else 1.0
            else:
                solidity = 1.0
            anchors.append({
                "label":     lid,
                "point":     np.array([ax, ay], dtype=np.float32),
                "area":      int(area),
                "elongation": elongation,
                "solidity":  solidity,
            })
        return anchors, labels

    def _suppress_shadows(self, anchors, total_px=None, img_shape=None, debug_label=""):
        if len(anchors) <= 1:
            return anchors
        pts        = np.array([a["point"]              for a in anchors], dtype=np.float32)
        areas      = np.array([a["area"]               for a in anchors], dtype=np.float32)
        solidities = np.array([a.get("solidity", 1.0)  for a in anchors], dtype=np.float32)
        N          = len(pts)
        is_shadow  = np.zeros(N, dtype=bool)
        shadow_reason = [""] * N
        area_median  = float(np.median(areas))
        large_thresh = area_median * self.shadow_large_ratio

        border_margin    = self.border_margin_px
        border_max_area  = self.border_margin_max_area
        cluster_area_thr = (total_px * self.shadow_cluster_area_ratio
                            if total_px is not None else 0.0)

        for i in range(N):
            # Border exclusion: ONLY for small blobs (instrument edge noise).
            # Valid ink marks near the image boundary are larger and must not be suppressed.
            if img_shape is not None and border_margin > 0 and areas[i] <= border_max_area:
                ih, iw = img_shape
                px, py = float(pts[i][0]), float(pts[i][1])
                if min(px, iw - px, py, ih - py) < border_margin:
                    is_shadow[i] = True; shadow_reason[i] = "border"; continue

            # Right-edge exclusion: suppresses any blob (any size) whose centroid is
            # within right_border_margin_px of the right image edge.  Used for
            # images where the surgical instrument sits at the far right.
            right_margin = self.right_border_margin_px
            if img_shape is not None and right_margin > 0:
                ih, iw = img_shape
                px = float(pts[i][0])
                if iw - px < right_margin:
                    is_shadow[i] = True; shadow_reason[i] = "right-border"; continue

            # Bottom-edge exclusion: suppresses any blob (any size) whose centroid is
            # within bottom_border_margin_px of the bottom image edge.  Used for
            # images where instrument artifacts pull the chain downward.
            bottom_margin = self.bottom_border_margin_px
            if img_shape is not None and bottom_margin > 0:
                ih, iw = img_shape
                py = float(pts[i][1])
                if ih - py < bottom_margin:
                    is_shadow[i] = True; shadow_reason[i] = "bottom-border"; continue

            # Tiny-blob anchor check: a tiny blob with no large neighbor is isolated
            # noise.  Valid tiny ink dots along a trajectory always sit near larger
            # blobs; instrument artifacts or sparkle far from the main path do not.
            tiny_req = self.tiny_blob_req_anchor_ratio
            if tiny_req > 0.0 and areas[i] < area_median * tiny_req:
                large_thr = area_median * self.tiny_blob_anchor_large_ratio
                dists_ta  = np.linalg.norm(pts - pts[i], axis=1)
                has_anchor = np.any(
                    (dists_ta < self.max_link_dist) &
                    (areas    >= large_thr) &
                    (np.arange(N) != i)
                )
                if not has_anchor:
                    is_shadow[i] = True; shadow_reason[i] = "tiny-no-anchor"; continue

            # Type B0: per-blob absolute area
            if total_px is not None and areas[i] > total_px * self.shadow_abs_area_ratio:
                is_shadow[i] = True; shadow_reason[i] = "B0-abs"; continue

            # Cluster-area check: if the total area of all nearby blobs collectively
            # exceeds shadow_cluster_area_ratio * total_px, treat as shadow.
            # Catches fragmented shadow regions whose individual components are each
            # below the per-blob B0 threshold but form a large spatial cluster.
            if cluster_area_thr > 0:
                dists_cl = np.linalg.norm(pts - pts[i], axis=1)
                cluster_mask = dists_cl < self.shadow_density_radius
                if cluster_mask.sum() >= self.shadow_density_count:
                    cluster_total = float(np.sum(areas[cluster_mask]))
                    if cluster_total > cluster_area_thr:
                        is_shadow[i] = True; shadow_reason[i] = "cluster-area"; continue

            # Type B1: very large + very irregular
            if areas[i] > area_median * self.shadow_strict_large_ratio and solidities[i] < self.shadow_strict_solidity:
                # Linearity escape: if the neighbourhood within density_radius is linear,
                # this large blob sits on a real ink trajectory — skip B1 suppression.
                dists_b1   = np.linalg.norm(pts - pts[i], axis=1)
                nearby_b1  = np.where(
                    (dists_b1 < self.shadow_density_radius) &
                    (np.arange(N) != i) &
                    (areas >= area_median * 0.25)
                )[0]
                if len(nearby_b1) >= 2:
                    g_pts  = pts[np.append(nearby_b1, i)]
                    c      = g_pts - g_pts.mean(axis=0)
                    vals_b1 = np.sort(np.linalg.eigvalsh(c.T @ c))[::-1]
                    lin_b1  = vals_b1[0] / (vals_b1[1] + 1e-6)
                    b1_lin_thr = self.shadow_linearity_threshold
                    if lin_b1 >= b1_lin_thr:
                        # Extra guard: at least one non-shadow neighbour must have
                        # area >= b1_large_self_ratio × this blob's own area.
                        # Real ink large blobs travel with other large blobs; a
                        # large shadow blob whose non-shadow neighbours are all much
                        # smaller is still suppressed despite the linearity escape.
                        # Shadow blobs already marked in this pass are excluded so
                        # that a large shadow blob cannot vouch for another.
                        self_ratio = self.b1_large_self_ratio
                        if self_ratio > 0:
                            nearby_ns = [j for j in nearby_b1 if not is_shadow[j]]
                            if not any(areas[j] >= self_ratio * areas[i]
                                       for j in nearby_ns):
                                is_shadow[i] = True; shadow_reason[i] = "B1"; continue
                        pass  # linear AND large-neighbour condition met → real ink
                    else:
                        is_shadow[i] = True; shadow_reason[i] = "B1"; continue
                else:
                    is_shadow[i] = True; shadow_reason[i] = "B1"; continue
            # Type B2: large + moderately irregular + many tiny neighbours
            # Extra check: if those tiny neighbours are themselves linearly arranged,
            # the big blob is likely a real ink mark sitting along a trajectory
            # (not a shadow mass), so skip B2 suppression in that case.
            if areas[i] > large_thresh and solidities[i] < self.shadow_solidity_min:
                dists_i  = np.linalg.norm(pts - pts[i], axis=1)
                tiny_mask = (
                    (dists_i < self.shadow_density_radius) &
                    (areas < area_median) &
                    (np.arange(N) != i)
                )
                n_tiny = int(np.sum(tiny_mask))
                if n_tiny >= self.shadow_density_count - 1:
                    # Check linearity of the tiny-neighbour distribution.
                    # Shadow clusters are 2D; trajectory neighbours are roughly 1D.
                    tiny_pts = pts[tiny_mask]
                    if len(tiny_pts) >= 3:
                        centered_t = tiny_pts - tiny_pts.mean(axis=0)
                        cov_t = centered_t.T @ centered_t
                        vals_t = np.sort(np.linalg.eigvalsh(cov_t))[::-1]
                        lin_tiny = vals_t[0] / (vals_t[1] + 1e-6)
                        if lin_tiny >= self.shadow_linearity_threshold:
                            pass  # linear tiny neighbours → real ink, fall through
                        else:
                            is_shadow[i] = True; shadow_reason[i] = "B2"; continue
                    else:
                        is_shadow[i] = True; shadow_reason[i] = "B2"; continue
            # Linearity test (Type A)
            # Filter neighbourhood to significant blobs only (area >= area_median * 0.25).
            # Tiny noise dots near a trajectory bend drag the PCA cluster toward 2-D,
            # which would wrongly suppress a valid ink blob sitting at that bend.
            # Use shadow_linearity_radius (smaller than shadow_density_radius) so that
            # distant large blobs on a curve don't falsely appear as a 2D cluster.
            lin_area_thresh = area_median * 0.25
            lin_radius = self.shadow_linearity_radius
            dists      = np.linalg.norm(pts - pts[i], axis=1)
            nearby_idx = np.where(
                (dists < lin_radius) &
                (np.arange(N) != i) &
                (areas >= lin_area_thresh)
            )[0]
            if len(nearby_idx) < self.shadow_density_count - 1:
                continue
            group_pts = pts[np.append(nearby_idx, i)]
            centered  = group_pts - group_pts.mean(axis=0)
            cov       = centered.T @ centered
            vals      = np.sort(np.linalg.eigvalsh(cov))[::-1]
            linearity = vals[0] / (vals[1] + 1e-6)
            if linearity < self.shadow_linearity_threshold:
                is_shadow[i] = True; shadow_reason[i] = f"linearity={linearity:.1f}"

        # ── Pass 2: re-test linearity-suppressed blobs ───────────────────────────
        # A blob may be falsely suppressed in pass 1 because a large shadow blob
        # was in its neighbourhood making it look 2D.  After pass 1, that shadow
        # is gone; re-run the linearity test excluding now-suppressed neighbours.
        # Applied to ALL sizes (not just small), since large legitimate ink marks
        # at trajectory bends can also fail the linearity test.
        for i in range(N):
            if not is_shadow[i]:
                continue
            if "linearity" not in shadow_reason[i]:
                continue
            lin_area_thresh = area_median * 0.25
            lin_radius = self.shadow_linearity_radius
            dists = np.linalg.norm(pts - pts[i], axis=1)
            # Only un-suppress if a large non-suppressed "anchor" blob is nearby.
            # This prevents floating shadow clusters (far from any real trajectory blob)
            # from being incorrectly freed in images like 8.47 and 7.56.
            anchor_thresh = area_median * 2.0
            anchor_mask = (~is_shadow) & (areas >= anchor_thresh) & (np.arange(N) != i)
            if not np.any((dists < self.max_link_dist) & anchor_mask):
                continue  # no large keep blob nearby → leave as shadow
            # Very large solid blobs (> 2× large_thresh) with a nearby anchor are almost
            # certainly real ink whose neighbourhood happens to look 2-D (e.g. one off-
            # trajectory fragment skews the PCA).  Un-suppress without re-running PCA.
            if areas[i] > large_thresh * 2 and solidities[i] >= self.shadow_solidity_min:
                is_shadow[i] = False
                shadow_reason[i] = ""
                continue
            nearby_idx = np.where(
                (dists < lin_radius) &
                (~is_shadow) &                           # exclude pass-1 shadows
                (np.arange(N) != i) &
                (areas >= lin_area_thresh)
            )[0]
            if len(nearby_idx) < self.shadow_density_count - 1:
                # Large irregular blobs stay shadow even with few remaining neighbours —
                # their shadow cluster was just removed in pass 1, leaving them sparse.
                # Valid ink blobs at bends are compact; these are not.
                if areas[i] > large_thresh and solidities[i] < self.shadow_solidity_min:
                    continue  # keep as shadow
                # Too few non-shadow neighbours → un-suppress
                is_shadow[i] = False
                shadow_reason[i] = ""
                continue
            group_pts = pts[np.append(nearby_idx, i)]
            centered  = group_pts - group_pts.mean(axis=0)
            cov       = centered.T @ centered
            vals      = np.sort(np.linalg.eigvalsh(cov))[::-1]
            linearity = vals[0] / (vals[1] + 1e-6)
            if linearity >= self.shadow_linearity_threshold * 0.875:
                is_shadow[i] = False
                shadow_reason[i] = ""
        # ─────────────────────────────────────────────────────────────────────────

        if debug_label:
            print(f"  [{debug_label}] shadow suppression (median_area={area_median:.0f}, "
                  f"cluster_thr={cluster_area_thr:.0f} px):")
            for i, a in enumerate(anchors):
                tag = f"SHADOW({shadow_reason[i]})" if is_shadow[i] else "keep"
                print(f"    blob#{i:2d}  area={a['area']:5d}  sol={a.get('solidity',1):.2f}  "
                      f"pt=({a['point'][0]:.0f},{a['point'][1]:.0f})  {tag}")

        return [a for i, a in enumerate(anchors) if not is_shadow[i]]

    def _trim_outlier_links(self, chain, outlier_ratio):
        """
        Remove endpoint blobs whose connecting link is an outlier vs. the reference distance.

        FIX: Reference median is computed only from links where BOTH endpoints have area >=
        25% of the 5th-largest blob in the chain (filters tiny noise blobs that create
        artifically short median distances).  Also applies a floor of
        max_link_dist * trim_ref_floor_ratio so a short-median chain never over-trims.
        """
        if len(chain) < 3:
            return chain

        pts   = np.array([c["point"] for c in chain], dtype=np.float64)
        areas = np.array([c["area"]  for c in chain], dtype=np.float64)

        while len(chain) >= 3:
            dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)

            # ── FIX: significant-blob reference ──────────────────────────────────
            sorted_areas = np.sort(areas)[::-1]
            ref_area     = sorted_areas[min(4, len(sorted_areas) - 1)]   # 5th largest
            sig_thresh   = ref_area * 0.25                                # 25 % of 5th-largest
            sig_mask     = (areas[:-1] >= sig_thresh) & (areas[1:] >= sig_thresh)
            ref_dists    = dists[sig_mask] if sig_mask.sum() >= 2 else dists
            med          = float(np.percentile(ref_dists, 75))           # P75 of sig links

            # Floor: never trim something that is merely moderately longer than
            # the typical link when all links happen to be short.
            floor = self.max_link_dist * self.trim_ref_floor_ratio
            med   = max(med, floor)
            # ─────────────────────────────────────────────────────────────────────

            bad = np.where(dists > outlier_ratio * med)[0]
            if len(bad) == 0:
                break
            split  = int(bad[0]) + 1
            piece1, piece2 = chain[:split], chain[split:]
            chain  = piece1 if len(piece1) >= len(piece2) else piece2
            pts    = np.array([c["point"] for c in chain], dtype=np.float64)
            areas  = np.array([c["area"]  for c in chain], dtype=np.float64)
        return chain

    def _build_chain(self, anchors, max_link_dist, max_angle_deg, dp_link_scale=0):
        """
        Longest chain with direction-continuity constraint.

        FIX: Per-node area contribution is capped at NODE_BONUS (the median area).
        This prevents a single large/shadow blob from outscoring a longer correct path
        through multiple smaller ink blobs.  Two small blobs always beat one large blob.
        """
        if len(anchors) < 2:
            return anchors

        pts        = np.array([a["point"] for a in anchors], dtype=np.float64)
        areas_arr  = np.array([a["area"]  for a in anchors], dtype=np.float64)
        N          = len(pts)
        cos_thresh = np.cos(np.radians(max_angle_deg))

        vec      = pts[None,:,:] - pts[:,None,:]
        dist_mat = np.linalg.norm(vec, axis=2)
        with np.errstate(invalid='ignore', divide='ignore'):
            uv = vec / (dist_mat[:,:,None] + 1e-8)

        if dp_link_scale > 0:
            # Proportional distance limit: allowed link distance grows with the
            # smaller of the two blob areas.  Prevents small blobs from being
            # chained over large gaps that only large blobs can legitimately span.
            min_areas      = np.minimum(areas_arr[:, None], areas_arr[None, :])
            effective_max  = max_link_dist + min_areas / dp_link_scale
            link_ok = (dist_mat > 1) & (dist_mat <= effective_max)
        else:
            link_ok = (dist_mat > 1) & (dist_mat <= max_link_dist)

        # Area-dominated scoring: blob area is the primary score driver so that
        # a chain of 3 big ink blobs beats a chain of 20 tiny sparkle dots.
        # NODE_BONUS is a tiny tie-breaker (0.1*median) so longer chains still beat
        # shorter chains when blob sizes are equal, without letting count dominate.
        # No upper cap on area — big legitimate ink blobs should be fully rewarded.
        NODE_BONUS = float(np.median(areas_arr)) * 0.1

        dp = np.where(link_ok.T, dist_mat.T * 0.001 + NODE_BONUS + areas_arr[:, np.newaxis], -1.0)
        back = np.full((N, N), -1, dtype=np.int32)

        for _ in range(N - 2):
            improved = False
            new_dp   = dp.copy()
            new_back = back.copy()
            for j in range(N):
                for i in range(N):
                    if dp[i, j] < 0:
                        continue
                    dir_in = uv[j, i]
                    dots   = uv[i] @ dir_in
                    ok     = link_ok[i] & (dots >= cos_thresh)
                    ok[i]  = ok[j] = False
                    cands  = dp[i, j] + dist_mat[i] * 0.001 + NODE_BONUS + areas_arr
                    better = ok & (cands > new_dp[:, i])
                    new_dp[:,   i] = np.where(better, cands,  new_dp[:,   i])
                    new_back[:, i] = np.where(better, j,      new_back[:, i])
                    improved |= bool(np.any(better))
            dp, back = new_dp, new_back
            if not improved:
                break

        if dp.max() < 0:
            return []

        end_i, end_j = map(int, np.unravel_index(np.argmax(dp), dp.shape))
        path = [end_i, end_j]
        ci, cj = end_i, end_j
        while back[ci, cj] >= 0:
            pk = int(back[ci, cj])
            path.append(pk)
            ci, cj = cj, pk
        path.reverse()
        seen, unique = set(), []
        for idx in path:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        if len(unique) < self.min_chain_nodes:
            return []
        return [anchors[i] for i in unique]

    def _extend_endpoints(self, chain, keep_anchors, max_dist, max_angle_deg, dp_link_scale=0):
        """
        Greedily extend each endpoint of the chain after trimming.
        Adds non-chain keep blobs within max_dist that continue the trajectory
        direction (angle ≤ ext_max_angle_deg), preferring larger blobs.

        dp_link_scale: when > 0, apply proportional distance limit using the smaller
        of the two blobs' areas (mirrors chain DP building constraint).  Only pass
        this after sparkle retry to avoid constraining normal long-gap extensions.
        """
        if len(chain) < 2:
            return chain
        chain_labels = {c["label"] for c in chain}
        ext_angle  = max_angle_deg   # caller controls the angle limit
        cos_thresh = np.cos(np.radians(ext_angle))
        chain_med  = float(np.median([c["area"] for c in chain]))
        min_area   = chain_med * self.ext_min_area_ratio

        changed = True
        while changed:
            changed  = False
            available = [a for a in keep_anchors
                         if a["label"] not in chain_labels and a["area"] >= min_area]
            if not available:
                break

            # Try head then tail; restart outer loop after each successful extension
            for is_head in (True, False):
                endpoint = chain[0]  if is_head else chain[-1]
                ref_blob = chain[1]  if is_head else chain[-2]
                dir_vec  = endpoint["point"] - ref_blob["point"]
                norm     = float(np.linalg.norm(dir_vec))
                if norm < 1e-6:
                    continue
                dir_vec = dir_vec / norm

                best_blob, best_score = None, -1.0
                normal_link = self.max_link_dist
                for blob in available:
                    vec = blob["point"] - endpoint["point"]
                    d   = float(np.linalg.norm(vec))
                    if d < 1 or d > max_dist:
                        continue
                    cos = float(np.dot(vec / d, dir_vec))
                    if cos < cos_thresh:
                        continue
                    # Endpoint-proportional distance limit: beyond max_link_dist,
                    # the endpoint blob's area determines how far the chain can
                    # reach.  max_allowed = max_link_dist + ep.area / ep_scale.
                    # A large endpoint can bridge larger gaps; a small one cannot.
                    ep_scale = self.ext_ep_scale
                    if ep_scale > 0 and d > normal_link:
                        max_allowed = normal_link + endpoint["area"] / ep_scale
                        if d > max_allowed:
                            continue
                    # Proportional distance limit (sparkle-only): smaller of the two
                    # blob areas caps the link distance.  A small candidate cannot be
                    # reached over a large gap even if the endpoint blob is large.
                    if dp_link_scale > 0:
                        max_prop = normal_link + min(endpoint["area"], blob["area"]) / dp_link_scale
                        if d > max_prop:
                            continue
                    if blob["area"] > best_score:   # prefer larger blobs
                        best_score = blob["area"]
                        best_blob  = blob

                if best_blob is not None:
                    chain = [best_blob] + chain if is_head else chain + [best_blob]
                    chain_labels.add(best_blob["label"])
                    changed = True
                    break   # restart outer loop with updated chain/endpoint

        return chain

    def _extract_single_path(self, binary_mask, debug_label=""):
        h, w = binary_mask.shape

        anchors, labels = self._extract_anchors(binary_mask)
        anchors = self._suppress_shadows(anchors, total_px=h * w, img_shape=(h, w),
                                         debug_label=debug_label)
        keep_anchors = anchors   # non-shadow blobs; used for endpoint extension

        chain   = self._build_chain(
                      anchors,
                      max_link_dist=self.max_link_dist,
                      max_angle_deg=self.chain_max_angle_deg)
        chain   = self._trim_outlier_links(chain, self.chain_outlier_ratio)

        # ── Bridge-node pruning ────────────────────────────────────────────────────
        # Remove interior chain nodes that are too small to justify bridging a large
        # gap.  For each interior node B between neighbours A and C:
        #   if skip_dist(A,C) > bridge_min_skip_dist
        #   AND B.area < bridge_min_area_ratio * skip_dist  →  remove B.
        # Repeated until no more removals (handles cascades).
        bridge_ratio    = self.bridge_min_area_ratio
        bridge_skip_min = self.bridge_min_skip_dist
        if bridge_ratio > 0 and len(chain) >= 3:
            changed = True
            while changed and len(chain) >= 3:
                changed = False
                for i in range(1, len(chain) - 1):
                    skip = float(np.linalg.norm(
                        chain[i-1]["point"] - chain[i+1]["point"]))
                    if skip > bridge_skip_min and chain[i]["area"] < bridge_ratio * skip:
                        if debug_label:
                            print(f"  [{debug_label}] bridge prune: removed area={chain[i]['area']} "
                                  f"(skip={skip:.0f}px, min={bridge_ratio*skip:.1f})")
                        chain = chain[:i] + chain[i+1:]
                        changed = True
                        break

        # ── Sparkle filter (BEFORE extension) ─────────────────────────────────────
        # Evaluate chain quality on the trimmed DP chain BEFORE endpoint extension.
        # Extension can add small endpoint blobs that would lower chain_med and
        # incorrectly trigger the sparkle filter on a valid chain.
        # If chain is dominated by tiny blobs while larger blobs exist elsewhere,
        # the DP latched onto a noise/sparkle region — retry with significant blobs.
        ext_dist        = self.chain_endpoint_ext_dist
        sparkle_ratio   = self.sparkle_filter_ratio
        ext_anchors     = keep_anchors   # default: extend with all keep blobs
        sparkle_accepted = False          # set True if sig retry replaces the chain
        is_loop          = False          # set True if loop detection triggered sparkle

        if sparkle_ratio > 0 and len(chain) >= 2 and len(keep_anchors) >= 3:
            chain_med    = float(np.median([c["area"] for c in chain]))
            sorted_areas = sorted([a["area"] for a in keep_anchors], reverse=True)
            ref_area     = sorted_areas[min(2, len(sorted_areas) - 1)]   # 3rd-largest blob
            ratio_val    = chain_med / ref_area if ref_area > 0 else 1.0

            # Loop detection: if the chain start and end are close relative to its
            # bounding-box diagonal, the DP traced a circular arc rather than a
            # linear trajectory.  Apply a higher sparkle ratio to force the retry
            # to use only the big blobs and break the loop.
            pts_chain  = np.array([c["point"] for c in chain])
            ep_dist    = float(np.linalg.norm(pts_chain[0] - pts_chain[-1]))
            bbox_diag  = float(np.linalg.norm(pts_chain.max(axis=0) - pts_chain.min(axis=0)))
            is_loop    = ep_dist < max(200.0, bbox_diag * 0.15)
            eff_ratio  = self.sparkle_loop_ratio if is_loop else sparkle_ratio

            if debug_label and (ratio_val < eff_ratio or is_loop):
                print(f"  [{debug_label}] sparkle: ratio={ratio_val:.3f} thr={eff_ratio:.3f}"
                      f"{'(loop,ep_dist={:.0f})'.format(ep_dist) if is_loop else ''}"
                      f" {'TRIGGERS' if ratio_val < eff_ratio else 'ok'}")

            if chain_med < eff_ratio * ref_area:
                sig = [a for a in keep_anchors if a["area"] >= eff_ratio * ref_area]
                if len(sig) >= 2:
                    sig_max_link = self.sparkle_sig_max_link_dist
                    c2 = self._build_chain(sig, sig_max_link,
                                           self.chain_max_angle_deg)
                    c2 = self._trim_outlier_links(c2, self.chain_outlier_ratio)
                    c2_med = float(np.median([c["area"] for c in c2])) if len(c2) >= 2 else 0.0
                    # Only replace chain if sig retry is significantly better (2x median improvement)
                    if len(c2) >= 2 and c2_med > chain_med * 2.0:
                        chain            = c2
                        ext_anchors      = sig   # extend only with sig blobs to prevent sparkle re-entry
                        sparkle_accepted = True
                        if debug_label:
                            print(f"  [{debug_label}]   >> sparkle retry accepted: "
                                  f"{len(sig)} sig blobs -> chain ({len(c2)} blobs, med={c2_med:.0f})")
                    else:
                        # Retry didn't improve enough — keep original chain but widen extension pool
                        ext_anchors = keep_anchors
                        if debug_label:
                            print(f"  [{debug_label}]   >> sparkle retry rejected "
                                  f"(c2={len(c2)} blobs, c2_med={c2_med:.0f} <= 2x chain_med={chain_med:.0f})")
        # ─────────────────────────────────────────────────────────────────────────

        # ── Endpoint extension ────────────────────────────────────────────────────
        if ext_dist > 0 and len(chain) >= 2:
            # Pass dp_link_scale only after sparkle retry so that normal long-gap
            # extensions (e.g. 7.87's 621px gap) are not constrained.
            _ext_dp_scale = self.dp_link_scale if sparkle_accepted else 0
            chain = self._extend_endpoints(chain, ext_anchors,
                                           max_dist=ext_dist,
                                           max_angle_deg=self.ext_max_angle_deg,
                                           dp_link_scale=_ext_dp_scale)
        # ─────────────────────────────────────────────────────────────────────────

        # ── Post-extension segment pruning ────────────────────────────────────────
        # After extension, a long link (> ext_seg_prune_dist) whose SMALLER side has
        # fewer than ext_seg_min_blobs blobs is an isolated noise cluster attached
        # via a gap jump — remove the smaller side.
        # For sparkle-retry chains, use sparkle_sig_max_link_dist as the prune threshold
        # so the sig DP's larger intra-chain links are not mistaken for noise clusters.
        if sparkle_accepted:
            prune_dist = self.sparkle_sig_max_link_dist
        else:
            prune_dist = self.ext_seg_prune_dist
        prune_min  = self.ext_seg_min_blobs
        if prune_dist < 9999 and len(chain) >= 3:
            changed = True
            while changed and len(chain) >= 3:
                changed = False
                pts  = np.array([c["point"] for c in chain], dtype=np.float64)
                lnks = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                for idx in range(len(lnks)):
                    if lnks[idx] <= prune_dist:
                        continue
                    p1, p2 = chain[:idx + 1], chain[idx + 1:]
                    sm, lg = (p1, p2) if len(p1) <= len(p2) else (p2, p1)
                    protect_area = self.ext_seg_protect_large_area
                    if len(sm) < prune_min and len(lg) >= 3 and not (
                            protect_area > 0 and len(sm) >= 1
                            and any(b["area"] >= protect_area for b in sm)):
                        # Prune if sm blobs are isolated (no non-chain keep neighbor within
                        # max_link_dist) OR the long link is a large outlier vs the other
                        # chain links (> 2x their median).  Non-isolated, non-outlier sm
                        # blobs have local support and may be valid endpoint continuations.
                        chain_labels_now = {c["label"] for c in chain}
                        sm_isolated = not any(
                            any(np.linalg.norm(other["point"] - b["point"]) <= self.max_link_dist
                                for other in keep_anchors
                                if other["label"] != b["label"]
                                and other["label"] not in chain_labels_now)
                            for b in sm)
                        other_lnks = np.concatenate([lnks[:idx], lnks[idx+1:]])
                        sm_outlier = (lnks[idx] > 2.0 * float(np.median(other_lnks))
                                      if len(other_lnks) >= 1 else True)
                        if sm_isolated or sm_outlier:
                            if debug_label:
                                reason = "isolated" if sm_isolated else "outlier"
                                print(f"  [{debug_label}] prune seg: removed {len(sm)}-blob side "
                                      f"(link={lnks[idx]:.0f}px, {reason})")
                            chain   = lg
                            changed = True
                            break
        # ─────────────────────────────────────────────────────────────────────────

        # ── Endpoint trimming ─────────────────────────────────────────────────────
        # Trim endpoint blobs that make a large turn AND are small (below 5th-largest
        # in the chain).  Catches spurious extensions to distant noise blobs where
        # the trajectory direction changes sharply.
        trim_angle = self.endpoint_trim_angle_deg
        if trim_angle > 0 and len(chain) >= 3:
            cos_trim = np.cos(np.radians(trim_angle))
            changed  = True
            while changed and len(chain) >= 3:
                changed = False
                chain_areas_sorted = sorted([c["area"] for c in chain], reverse=True)
                size_thresh   = chain_areas_sorted[min(4, len(chain_areas_sorted) - 1)]
                chain_median  = float(np.median([c["area"] for c in chain]))
                median_ratio  = self.endpoint_trim_median_ratio
                for is_head in (True, False):
                    ep = chain[0]  if is_head else chain[-1]
                    n1 = chain[1]  if is_head else chain[-2]
                    n2 = chain[2]  if is_head else chain[-3]
                    dir_main = n1["point"] - n2["point"]   # trajectory direction toward ep
                    dir_ep   = ep["point"] - n1["point"]   # direction from chain to endpoint
                    norm_m = float(np.linalg.norm(dir_main))
                    norm_e = float(np.linalg.norm(dir_ep))
                    if norm_m < 1 or norm_e < 1:
                        continue
                    cos_a = float(np.dot(dir_main / norm_m, dir_ep / norm_e))
                    ep_n1_dist_val = float(np.linalg.norm(ep["point"] - n1["point"]))
                    # Long-link + angle trim: bypasses size_thresh so that isolated outlier
                    # dots connected via a long extension jump can be removed even when they
                    # are larger than the 5th-largest chain blob.  Requires a bad angle so
                    # that legitimate endpoints with a long gap (e.g. 7.87, 43.186) are safe.
                    long_link_ratio = self.endpoint_trim_long_link_ratio
                    long_link_trim = (
                        long_link_ratio > 0
                        and ep_n1_dist_val > self.max_link_dist
                        and ep["area"] < long_link_ratio * chain_median
                        and cos_a < cos_trim  # also requires a bad angle
                    )
                    if long_link_trim:
                        if debug_label:
                            angle_deg = np.degrees(np.arccos(max(-1.0, min(1.0, cos_a))))
                            print(f"  [{debug_label}] trim {'head' if is_head else 'tail'} "
                                  f"ep: area={ep['area']} "
                                  f"long_link({ep_n1_dist_val:.0f}px,angle={angle_deg:.0f}°)")
                        chain = chain[1:] if is_head else chain[:-1]
                        changed = True
                        break
                    # Proportional-link trim (sparkle chains only): after sparkle
                    # retry, if the endpoint link exceeds the proportional distance
                    # limit (max_link_dist + min(ep,n1)/dp_link_scale), trim the
                    # endpoint regardless of size guard.  Catches blobs whose link
                    # distance is disproportionate to their size — the same constraint
                    # applied during the sig DP build step.
                    _dp_trim_scale = self.dp_link_scale
                    prop_link_trim = (
                        sparkle_accepted and _dp_trim_scale > 0
                        and ep_n1_dist_val > self.max_link_dist
                                              + min(ep["area"], n1["area"]) / _dp_trim_scale
                    )
                    if prop_link_trim:
                        if debug_label:
                            prop_max = self.max_link_dist + min(ep["area"], n1["area"]) / _dp_trim_scale
                            print(f"  [{debug_label}] trim {'head' if is_head else 'tail'} "
                                  f"ep: area={ep['area']} "
                                  f"prop_link({ep_n1_dist_val:.0f}px>{prop_max:.0f}px)")
                        chain = chain[1:] if is_head else chain[:-1]
                        changed = True
                        break
                    # Size-thresh guard: protect large endpoints from angle/adj/median trim.
                    if ep["area"] > size_thresh:
                        continue
                    # Trim if: angle too sharp OR endpoint is a tiny satellite fragment
                    # of a much larger blob.  Checks both n1 and n2: after one satellite
                    # is stripped, the next ep may now be adjacent to the large blob as n1.
                    adj_ratio    = self.endpoint_trim_adj_ratio
                    adj_min_area = self.endpoint_trim_adj_min_area
                    too_small  = False
                    if adj_ratio > 0 and not (adj_min_area > 0 and ep["area"] >= adj_min_area):
                        ep_n1_dist = ep_n1_dist_val
                        ep_n2_dist = float(np.linalg.norm(ep["point"] - n2["point"]))
                        too_small = (
                            (ep["area"] < adj_ratio * n1["area"]
                             and ep_n1_dist < self.max_link_dist)
                            or
                            (ep["area"] < adj_ratio * n2["area"]
                             and ep_n2_dist < self.max_link_dist)
                        )
                    # Also trim if endpoint is tiny relative to the chain median —
                    # catches small DP-chain dots that are not satellites of one large blob.
                    too_small_med = (median_ratio > 0
                                     and ep["area"] < median_ratio * chain_median)
                    # Skip angle check if ep is very close to n1 — a tightly-adjacent
                    # endpoint is part of the same structure, not a deviant outlier.
                    min_dist_angle = 50.0
                    angle_trim = (cos_a < cos_trim) and (ep_n1_dist_val >= min_dist_angle)
                    if angle_trim or too_small or too_small_med:
                        if debug_label:
                            angle_deg = np.degrees(np.arccos(max(-1.0, min(1.0, cos_a))))
                            if angle_trim:
                                reason = f"angle={angle_deg:.0f}°"
                            elif too_small:
                                reason = f"tiny({ep['area']}/{n1['area']})"
                            else:
                                reason = f"tiny_med({ep['area']}/{chain_median:.0f})"
                            print(f"  [{debug_label}] trim {'head' if is_head else 'tail'} "
                                  f"ep: area={ep['area']} {reason}")
                        chain = chain[1:] if is_head else chain[:-1]
                        changed = True
                        break
        # ── Both-small endpoint distance trim ─────────────────────────────────────
        # After angle/size trim: if both the endpoint and its neighbour are small
        # (area < ep_large_area_thr) AND the link exceeds ep_small_max_dist, trim.
        ep_large_thr = self.ep_large_area_thr
        ep_small_max = self.ep_small_max_dist
        if ep_large_thr > 0 and ep_small_max > 0 and len(chain) >= 2:
            ep_changed = True
            while ep_changed and len(chain) >= 2:
                ep_changed = False
                for is_head in (True, False):
                    ep = chain[0]  if is_head else chain[-1]
                    n1 = chain[1]  if is_head else chain[-2]
                    dist_ep = float(np.linalg.norm(ep["point"] - n1["point"]))
                    if max(ep["area"], n1["area"]) < ep_large_thr and dist_ep > ep_small_max:
                        if debug_label:
                            print(f"  [{debug_label}] trim {'head' if is_head else 'tail'} "
                                  f"ep: area={ep['area']} both_small_far"
                                  f"(dist={dist_ep:.0f}px, n1_area={n1['area']})")
                        chain = chain[1:] if is_head else chain[:-1]
                        ep_changed = True
                        break
        # ── Post-both-small tiny-adjacent endpoint trim ────────────────────────────
        # Second pass of adj_ratio check: catches tiny endpoints that were shielded
        # by a slightly-larger blob during the first trim_angle pass (e.g. blob#4
        # protected blob#3 in 12.49).  After both-small removes the shielding blob,
        # the new endpoint may be tiny relative to its now-exposed large neighbour.
        tiny_adj_ratio    = self.endpoint_trim_adj_ratio
        tiny_adj_min_area = self.endpoint_trim_adj_min_area
        if tiny_adj_ratio > 0 and len(chain) >= 2:
            ta_changed = True
            while ta_changed and len(chain) >= 2:
                ta_changed = False
                for is_head in (True, False):
                    ep = chain[0]  if is_head else chain[-1]
                    n1 = chain[1]  if is_head else chain[-2]
                    dist_ep = float(np.linalg.norm(ep["point"] - n1["point"]))
                    if (ep["area"] < tiny_adj_ratio * n1["area"] and dist_ep < self.max_link_dist
                            and not (tiny_adj_min_area > 0 and ep["area"] >= tiny_adj_min_area)):
                        if debug_label:
                            print(f"  [{debug_label}] trim {'head' if is_head else 'tail'} "
                                  f"ep: area={ep['area']} tiny_adj2"
                                  f"(n1_area={n1['area']}, dist={dist_ep:.0f}px)")
                        chain = chain[1:] if is_head else chain[:-1]
                        ta_changed = True
                        break
        # ── Extension pass 2: loop-triggered sparkle ──────────────────────────────
        # After all endpoint trimming, add non-sig blobs that continue the endpoint
        # direction using a wider angle.  Only fires when sparkle was accepted AND
        # the chain was a loop — these chains have a "turning" endpoint whose
        # continuation blobs fall outside the normal ext_max_angle_deg.
        if ext_dist > 0 and sparkle_accepted and is_loop and len(chain) >= 2:
            ext_loop_angle = self.ext_loop_angle_deg
            chain = self._extend_endpoints(chain, keep_anchors,
                                           max_dist=ext_dist,
                                           max_angle_deg=ext_loop_angle)
            if debug_label:
                print(f"  [{debug_label}] ext pass2 (loop, angle={ext_loop_angle}°): "
                      f"chain now {len(chain)} blobs")
        # ─────────────────────────────────────────────────────────────────────────

        if debug_label:
            print(f"  [{debug_label}] chain ({len(chain)} blobs):")
            for c in chain:
                print(f"    area={c['area']:5d}  pt=({c['point'][0]:.0f},{c['point'][1]:.0f})")

        if len(chain) < 2:
            return self._extract_longest_path(skeletonize(binary_mask > 0))

        out = np.zeros((h, w), dtype=np.uint8)
        for a, b in zip(chain[:-1], chain[1:]):
            x1, y1 = a["point"]; x2, y2 = b["point"]
            cv2.line(out,
                     (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))),
                     255, thickness=self.link_thickness)
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
        return self._extract_longest_path(skeletonize(out > 0))

    # ── Public API ──────────────────────────────────────────────────────────────

    def __call__(self, img_bgr):
        """Return single-path trajectory mask (uint8 H×W, trajectory=255)."""
        raw_mask                       = self._segment_dark_marker(img_bgr)
        return self._extract_single_path(raw_mask)

# ── CLI entry point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Extract dissection trajectory from a single image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",  help="Input BGR image (PNG / JPG)")
    p.add_argument("output", help="Output trajectory mask (PNG); trajectory pixels = 255")

    # Most useful CLI flags
    p.add_argument("--marker_mode",       default="auto",  help="auto|color|dark|light")
    p.add_argument("--max_link_dist",     type=float, default=165.0)
    p.add_argument("--chain_max_angle_deg", type=int, default=55)
    p.add_argument("--ext_max_angle_deg",   type=int, default=65)
    p.add_argument("--shadow_abs_area_ratio", type=float, default=0.008)
    p.add_argument("--shadow_large_ratio",    type=float, default=8.0)
    p.add_argument("--shadow_solidity_min",   type=float, default=0.65)
    p.add_argument("--border_margin_px",      type=int,   default=30)
    p.add_argument("--border_margin_max_area",type=int,   default=300)
    p.add_argument("--right_border_margin_px",type=int,   default=0)
    p.add_argument("--bottom_border_margin_px",type=int,  default=0)
    p.add_argument("--chain_endpoint_ext_dist",type=int,  default=380)
    p.add_argument("--sparkle_filter_ratio",  type=float, default=0.110)
    p.add_argument("--ext_seg_prune_dist",    type=int,   default=218)
    p.add_argument("--ext_seg_min_blobs",     type=int,   default=3)
    p.add_argument("--endpoint_trim_angle_deg",type=int,  default=50)
    p.add_argument("--use_clahe",             type=int,   default=1, help="1=on, 0=off")

    args = p.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: cannot read {args.input}", file=sys.stderr)
        sys.exit(1)

    extractor = DissectionTrajectoryExtractor(
        marker_mode            = args.marker_mode,
        max_link_dist          = args.max_link_dist,
        chain_max_angle_deg    = args.chain_max_angle_deg,
        ext_max_angle_deg      = args.ext_max_angle_deg,
        shadow_abs_area_ratio  = args.shadow_abs_area_ratio,
        shadow_large_ratio     = args.shadow_large_ratio,
        shadow_solidity_min    = args.shadow_solidity_min,
        border_margin_px       = args.border_margin_px,
        border_margin_max_area = args.border_margin_max_area,
        right_border_margin_px = args.right_border_margin_px,
        bottom_border_margin_px= args.bottom_border_margin_px,
        chain_endpoint_ext_dist= args.chain_endpoint_ext_dist,
        sparkle_filter_ratio   = args.sparkle_filter_ratio,
        ext_seg_prune_dist     = args.ext_seg_prune_dist,
        ext_seg_min_blobs      = args.ext_seg_min_blobs,
        endpoint_trim_angle_deg= args.endpoint_trim_angle_deg,
        use_clahe              = bool(args.use_clahe),
    )
    mask = extractor(img)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)
    print(f"Saved: {out_path}  ({mask.shape[1]}x{mask.shape[0]}, "
          f"{int((mask > 0).sum())} trajectory pixels)")


def run_trajectory_pipeline(
    input_path,
    output_path                 = None,
    # --- marker mode ---
    marker_mode                 = "auto",
    marker_h_min                = 112,
    marker_h_max                = 160,
    marker_s_min                = 90,
    marker_v_min                = 40,
    min_color_pixels            = 10000,
    light_v_min                 = 200,
    light_s_max                 = 60,
    # --- MV-D dark/light toggle ---
    mv_votes_needed             = 2,
    mv_ratio_threshold          = 0.5748,
    mv_narrow_threshold         = 147,
    mv_narrow_ratio_threshold   = 2.402,
    mv_white_threshold          = 188112,
    # --- tissue / contrast ---
    use_clahe                   = True,
    tissue_v_min                = 40,
    tissue_l_min                = 40,
    cloth_h_min                 = 85,
    cloth_h_max                 = 135,
    cloth_s_min                 = 25,
    tissue_dilate_kernel        = 40,
    # --- morphology ---
    open_kernel                 = 2,
    close_kernel                = 11,
    close_iters                 = 1,
    min_component_area          = 25,
    max_component_area_ratio    = 0.03,
    # --- shadow suppression ---
    shadow_abs_area_ratio       = 0.008,
    shadow_large_ratio          = 8.0,
    shadow_solidity_min         = 0.65,
    shadow_strict_large_ratio   = 6.0,
    shadow_strict_solidity      = 0.55,
    shadow_cluster_area_ratio   = 0.020,
    shadow_linearity_threshold  = 4.0,
    shadow_linearity_radius     = 200,
    shadow_density_radius       = 250,
    shadow_density_count        = 4,
    b1_large_self_ratio         = 0.12,
    # --- border exclusion ---
    border_margin_px            = 30,
    border_margin_max_area      = 300,
    right_border_margin_px      = 0,
    bottom_border_margin_px     = 0,
    # --- tiny blob suppression ---
    tiny_blob_req_anchor_ratio  = 0.0,
    tiny_blob_anchor_large_ratio= 2.0,
    # --- chain / DP ---
    max_link_dist               = 165.0,
    chain_max_angle_deg         = 55,
    chain_outlier_ratio         = 3.0,
    trim_ref_floor_ratio        = 0.30,
    dp_link_scale               = 8,
    min_chain_nodes             = 2,
    # --- endpoint extension ---
    chain_endpoint_ext_dist     = 380,
    ext_max_angle_deg           = 65,
    ext_min_area_ratio          = 0.05,
    ext_ep_scale                = 1.0,
    ext_loop_angle_deg          = 90,
    # --- post-extension pruning ---
    ext_seg_prune_dist          = 218,
    ext_seg_min_blobs           = 3,
    ext_seg_protect_large_area  = 500,
    # --- sparkle filter ---
    sparkle_filter_ratio        = 0.110,
    sparkle_loop_ratio          = 0.35,
    sparkle_sig_max_link_dist   = 320,
    # --- endpoint trimming ---
    endpoint_trim_angle_deg     = 50,
    endpoint_trim_adj_ratio     = 0.04,
    endpoint_trim_adj_min_area  = 100,
    endpoint_trim_median_ratio  = 0.03,
    endpoint_trim_long_link_ratio = 1.5,
    ep_large_area_thr           = 150,
    ep_small_max_dist           = 130,
    # --- bridge pruning ---
    bridge_min_area_ratio       = 0.15,
    bridge_min_skip_dist        = 200.0,
    # --- visualisation ---
    line_thickness              = 3,
    link_thickness              = 4,
    prune_branch_length         = 10,
):
    """Extract dissection trajectory from an image file.

    Parameters
    ----------
    input_path : str
        Path to input image (BGR, any OpenCV-readable format).
    output_path : str or None
        If given, saves the trajectory mask as a PNG to this path.
        If None, returns (raw_mask, skeleton, trajectory) instead.

    All remaining parameters tune the pipeline and have pre-trained defaults.

    Returns
    -------
    If output_path is None:
        (raw_mask, skeleton, trajectory) — three uint8 H×W numpy arrays.
    If output_path is given:
        None  (result is written to disk).
    """
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    extractor = DissectionTrajectoryExtractor(
        marker_mode=marker_mode, marker_h_min=marker_h_min,
        marker_h_max=marker_h_max, marker_s_min=marker_s_min,
        marker_v_min=marker_v_min, min_color_pixels=min_color_pixels,
        light_v_min=light_v_min, light_s_max=light_s_max,
        mv_votes_needed=mv_votes_needed, mv_ratio_threshold=mv_ratio_threshold,
        mv_narrow_threshold=mv_narrow_threshold,
        mv_narrow_ratio_threshold=mv_narrow_ratio_threshold,
        mv_white_threshold=mv_white_threshold,
        use_clahe=use_clahe, tissue_v_min=tissue_v_min, tissue_l_min=tissue_l_min,
        cloth_h_min=cloth_h_min, cloth_h_max=cloth_h_max, cloth_s_min=cloth_s_min,
        tissue_dilate_kernel=tissue_dilate_kernel,
        open_kernel=open_kernel, close_kernel=close_kernel, close_iters=close_iters,
        min_component_area=min_component_area,
        max_component_area_ratio=max_component_area_ratio,
        shadow_abs_area_ratio=shadow_abs_area_ratio,
        shadow_large_ratio=shadow_large_ratio, shadow_solidity_min=shadow_solidity_min,
        shadow_strict_large_ratio=shadow_strict_large_ratio,
        shadow_strict_solidity=shadow_strict_solidity,
        shadow_cluster_area_ratio=shadow_cluster_area_ratio,
        shadow_linearity_threshold=shadow_linearity_threshold,
        shadow_linearity_radius=shadow_linearity_radius,
        shadow_density_radius=shadow_density_radius,
        shadow_density_count=shadow_density_count,
        b1_large_self_ratio=b1_large_self_ratio,
        border_margin_px=border_margin_px,
        border_margin_max_area=border_margin_max_area,
        right_border_margin_px=right_border_margin_px,
        bottom_border_margin_px=bottom_border_margin_px,
        tiny_blob_req_anchor_ratio=tiny_blob_req_anchor_ratio,
        tiny_blob_anchor_large_ratio=tiny_blob_anchor_large_ratio,
        max_link_dist=max_link_dist, chain_max_angle_deg=chain_max_angle_deg,
        chain_outlier_ratio=chain_outlier_ratio,
        trim_ref_floor_ratio=trim_ref_floor_ratio, dp_link_scale=dp_link_scale,
        min_chain_nodes=min_chain_nodes,
        chain_endpoint_ext_dist=chain_endpoint_ext_dist,
        ext_max_angle_deg=ext_max_angle_deg, ext_min_area_ratio=ext_min_area_ratio,
        ext_ep_scale=ext_ep_scale, ext_loop_angle_deg=ext_loop_angle_deg,
        ext_seg_prune_dist=ext_seg_prune_dist, ext_seg_min_blobs=ext_seg_min_blobs,
        ext_seg_protect_large_area=ext_seg_protect_large_area,
        sparkle_filter_ratio=sparkle_filter_ratio, sparkle_loop_ratio=sparkle_loop_ratio,
        sparkle_sig_max_link_dist=sparkle_sig_max_link_dist,
        endpoint_trim_angle_deg=endpoint_trim_angle_deg,
        endpoint_trim_adj_ratio=endpoint_trim_adj_ratio,
        endpoint_trim_adj_min_area=endpoint_trim_adj_min_area,
        endpoint_trim_median_ratio=endpoint_trim_median_ratio,
        endpoint_trim_long_link_ratio=endpoint_trim_long_link_ratio,
        ep_large_area_thr=ep_large_area_thr, ep_small_max_dist=ep_small_max_dist,
        bridge_min_area_ratio=bridge_min_area_ratio,
        bridge_min_skip_dist=bridge_min_skip_dist,
        line_thickness=line_thickness, link_thickness=link_thickness,
        prune_branch_length=prune_branch_length,
    )

    raw_mask   = extractor._segment_dark_marker(img)
    trajectory = extractor._extract_single_path(raw_mask)

    if output_path:
        import pathlib
        # Draw trajectory as a bright green overlay on the original image
        overlay = img.copy()
        overlay[trajectory > 0] = (0, 255, 0)   # green in BGR
        out = pathlib.Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), overlay)
        print(f"Saved: {out}  ({int((trajectory > 0).sum())} trajectory pixels)")
        return None

    return trajectory


if __name__ == "__main__":
    main()


# =============================================================================
# PARAMETER REFERENCE
# =============================================================================
#
# run_trajectory_pipeline(input_path, output_path=None, **params)
# DissectionTrajectoryExtractor(**params)(img_bgr)
#
# USE CASES
# ---------
#   raw, skel, traj = run_trajectory_pipeline("img.png")
#       No output_path → returns three uint8 H×W arrays:
#         raw  = binary marker mask before trajectory extraction
#         skel = skeleton visualisation
#         traj = final single-path trajectory mask (trajectory pixels = 255)
#
#   run_pipeline("img.png", "out.png")
#       output_path given → saves trajectory mask to disk, returns None.
#
#   run_pipeline("img.png", "out.png", marker_mode="dark", max_link_dist=200)
#       Override any subset of parameters; all others use trained defaults.
#
#   extractor = DissectionTrajectoryExtractor(max_link_dist=200)
#   for img in images:
#       mask = extractor(img)   # reuse same extractor across many images
#
# =============================================================================
# MARKER / MODE SELECTION
# =============================================================================
#   marker_mode          "auto"|"color"|"dark"|"light"
#                        auto: uses color mask if strong, else MV-D decides dark vs light.
#                        color: always use purple/blue HSV mask.
#                        dark:  always use dark (low-V, high-S) mask.
#                        light: always use light (high-V, low-S) mask.
#   marker_h_min/max     HSV hue range for color marker (purple ≈ 112–160).
#   marker_s_min         Min HSV saturation for color marker.
#   marker_v_min         Min HSV value for color marker.
#   min_color_pixels     If color mask has >= this many pixels, use color mask directly.
#                        Skip MV-D entirely. Lower = more sensitive to faint color markers.
#
# LIGHT MASK (white/pale marker on tissue)
# -----------------------------------------
#   light_v_min          Min HSV-V for light marker pixels (default 200 = very bright).
#   light_s_max          Max HSV-S for light marker pixels (default 60 = near-white).
#                        Together these detect bright, low-saturation regions (white fascia).
#
# MV-D DARK vs LIGHT TOGGLE (auto mode only, when color mask is weak)
# --------------------------------------------------------------------
#   Four independent classifiers each cast one vote; >= mv_votes_needed → use light mask.
#   All thresholds learned from 108 labeled images (37 dissection folders × 3 frames).
#
#   mv_votes_needed          Votes needed to switch to light mask (default 2).
#                            Set to None to disable MV-D and always use dark mask.
#   mv_ratio_threshold       col_px / (drk_px+1) >= T → color vote.
#                            High ratio = many color pixels relative to dark pixels.
#   mv_narrow_threshold      narrow_dark_px < T → color vote.
#                            narrow_dark counts only true dark-ink pixels (H 150–165, V≤80).
#                            Excludes dark red meat at H 165–180 that inflates drk_px.
#   mv_narrow_ratio_threshold col_px / (narrow_dark+1) >= T → color vote.
#                            Robust version of ratio; not thrown off by red meat background.
#   mv_white_threshold       white_tissue_px >= T → color vote.
#                            white_tissue = tissue pixels with V≥light_v_min & S≤light_s_max.
#                            High count signals white/pale fascia surface (color-mode images).
#
# =============================================================================
# TISSUE ROI
# =============================================================================
#   use_clahe            Apply CLAHE contrast enhancement before any processing.
#   tissue_v_min         Min HSV-V to be considered tissue (excludes very dark background).
#   tissue_l_min         Min LAB-L to be considered tissue (excludes very dark background).
#   cloth_h_min/max      HSV hue range of surgical cloth/drape to exclude from tissue ROI.
#   cloth_s_min          Min saturation of cloth pixels to exclude.
#   tissue_dilate_kernel Kernel size for dilating tissue mask (fills small gaps).
#
# =============================================================================
# MORPHOLOGICAL CLEANUP (applied to marker mask)
# =============================================================================
#   open_kernel          Erosion then dilation — removes small noise specks.
#   close_kernel         Dilation then erosion — fills small holes inside marker blobs.
#   close_iters          Number of closing iterations.
#   min_component_area   Blobs smaller than this (px²) are discarded entirely.
#   max_component_area_ratio  Blobs larger than this fraction of total image pixels
#                            are discarded (likely background leakage).
#   prune_branch_length  Skeleton branches shorter than this (px) are pruned.
#   line_thickness       Thickness of drawn trajectory line (px).
#   link_thickness       Thickness of link lines drawn between chain blobs (px).
#
# =============================================================================
# SHADOW / ARTIFACT SUPPRESSION
# =============================================================================
#   shadow_abs_area_ratio      Blobs with area > ratio × total_px are "large" candidates
#                              for B0 absolute suppression. Raise if valid ink blobs are
#                              incorrectly suppressed.
#   shadow_large_ratio         Blobs with area > ratio × median_area trigger B1 (large +
#                              low solidity) check. Default 8: blobs 8× median are suspect.
#   shadow_solidity_min        B1 solidity threshold. Blobs below this AND large are shadows.
#   shadow_strict_large_ratio  Stricter large-ratio for B2 (fewer neighbours).
#   shadow_strict_solidity     Stricter solidity for B2.
#   shadow_cluster_area_ratio  If total area of all blobs within shadow_density_radius
#                              exceeds ratio × total_px → cluster shadow suppression.
#   shadow_linearity_threshold PCA linearity ratio threshold. Blobs whose neighbourhood
#                              is NOT linear (ratio < threshold) are shadows. Lower = more
#                              aggressive suppression.
#   shadow_linearity_radius    Radius (px) for PCA neighbourhood in linearity test.
#                              Smaller than density_radius to avoid pulling in far blobs.
#   shadow_density_radius      Radius (px) for cluster-area and B2 density checks.
#   shadow_density_count       Min neighbour count to consider a blob "dense" for B2.
#   b1_large_self_ratio        B1 linearity escape: only escape if largest neighbour has
#                              area >= this fraction of candidate's area. Prevents isolated
#                              large shadows from escaping via tiny collinear neighbours.
#
# =============================================================================
# BORDER EXCLUSION
# =============================================================================
#   border_margin_px         Distance (px) from image edge to flag a blob as "border".
#   border_margin_max_area   Only suppress border blobs SMALLER than this (px²).
#                            Larger blobs near the border are real ink, not hardware noise.
#   right_border_margin_px   Suppress ALL blobs within this distance of the right edge.
#                            Default 0 = disabled. Use when instrument clamp appears at right.
#   bottom_border_margin_px  Same for the bottom edge. Default 0 = disabled.
#
# =============================================================================
# TINY BLOB SUPPRESSION
# =============================================================================
#   tiny_blob_req_anchor_ratio   Blobs with area < median × ratio AND no large neighbour
#                                within max_link_dist are suppressed as isolated noise.
#                                Default 0.0 = disabled.
#   tiny_blob_anchor_large_ratio A "large" neighbour has area >= median × this ratio.
#
# =============================================================================
# CHAIN / DP TRAJECTORY BUILDING
# =============================================================================
#   max_link_dist        Max distance (px) between consecutive chain blobs. Gaps larger
#                        than this break the chain. Raise to bridge wider gaps.
#   chain_max_angle_deg  Max turn angle (deg) between consecutive links in the DP chain.
#                        Lower = straighter paths; raise if trajectory curves sharply.
#   chain_outlier_ratio  A chain link is an outlier if its length > ratio × P75 of other
#                        links. Outlier links are trimmed.
#   trim_ref_floor_ratio Floor for the trim reference distance = ratio × max_link_dist.
#                        Prevents over-trimming when the chain has very short median links.
#   dp_link_scale        Proportional distance bonus in DP: max_link = max_link_dist +
#                        min(area_i, area_j) / dp_link_scale. Larger blobs can bridge
#                        bigger gaps. Set 0 to disable.
#   min_chain_nodes      Minimum number of blobs for a valid chain.
#
# =============================================================================
# ENDPOINT EXTENSION
# =============================================================================
#   chain_endpoint_ext_dist  After trimming, greedily extend chain endpoints by adding
#                            keep blobs within this distance that continue the direction.
#                            Raise to reach farther blobs at trajectory ends.
#   ext_max_angle_deg        Max angle (deg) for extension candidates. Wider than chain
#                            angle to handle abrupt curve endpoints.
#   ext_min_area_ratio       Extension candidates must have area >= ratio × chain median.
#                            Filters tiny noise fragments from being added.
#   ext_ep_scale             Endpoint-proportional distance: max_ext = chain_endpoint_ext_dist
#                            + ep.area / ext_ep_scale. Larger endpoints can bridge wider gaps.
#   ext_loop_angle_deg       Wider angle used in extension pass 2 (loop-triggered sparkle only).
#
# POST-EXTENSION SEGMENT PRUNING
# --------------------------------
#   ext_seg_prune_dist       After extension, check links longer than this (px). If the
#                            smaller side of the split has < ext_seg_min_blobs blobs AND
#                            those blobs are isolated or the link is an outlier → prune.
#   ext_seg_min_blobs        Min blobs on smaller side to keep a long extension link.
#   ext_seg_protect_large_area  Never prune a small side containing a blob >= this area (px²).
#
# =============================================================================
# SPARKLE FILTER (noise-dominated chain detection)
# =============================================================================
#   sparkle_filter_ratio     If chain median area < ratio × 3rd-largest keep blob area,
#                            the chain is dominated by tiny noise dots. Re-run DP using
#                            only significant (large) blobs. Lower = triggers more often.
#   sparkle_loop_ratio       Higher sparkle threshold used when the chain forms a near-loop
#                            (ep_dist < max(200, bbox_diag × 0.15)). Handles arc trajectories.
#   sparkle_sig_max_link_dist  Max link distance for the sig-only retry DP. Larger than
#                              max_link_dist because large blobs can justify wider gaps.
#
# =============================================================================
# ENDPOINT TRIMMING
# =============================================================================
#   endpoint_trim_angle_deg     Trim endpoint if it makes a turn > this angle AND is small.
#   endpoint_trim_adj_ratio     Trim endpoint if area < ratio × adjacent blob area (tiny
#                               fragment attached via a large bridge blob).
#   endpoint_trim_adj_min_area  Don't apply adj_ratio trim if endpoint area >= this (px²).
#                               Protects medium endpoints dwarfed by a very large neighbour.
#   endpoint_trim_median_ratio  Trim endpoint if area < ratio × chain median (isolated dot).
#   endpoint_trim_long_link_ratio  Trim endpoint if link > max_link_dist AND angle > trim
#                                  angle AND area < ratio × chain median. Removes outlier
#                                  extension blobs even if larger than the 5th-biggest.
#   ep_large_area_thr           Area threshold for "large" endpoint (no distance restriction).
#   ep_small_max_dist           If BOTH endpoint and neighbour are small, max allowed link
#                               distance. Trims small blobs connected via a long jump.
#
# =============================================================================
# BRIDGE NODE PRUNING
# =============================================================================
#   bridge_min_area_ratio    Interior node B between A and C is pruned if skip_dist(A,C)
#                            > bridge_min_skip_dist AND B.area < ratio × skip_dist.
#                            Removes tiny "stepping-stone" blobs that lie on the path by
#                            chance and add no real ink evidence.
#   bridge_min_skip_dist     Only apply bridge pruning when A→C skip exceeds this (px).
#
# =============================================================================
# WAYPOINT INSERTION
# =============================================================================
#   waypoint_max_lateral     After extension, insert keep blobs between consecutive chain
#                            nodes if lateral deviation from chord < this (px). Ensures
#                            nearby off-chain blobs are captured as skeleton anchors.
#   waypoint_max_angle_deg   Max turn angle at inserted waypoint (deg). Prevents concave
#                            V-shapes that would create loops in the skeleton.
# =============================================================================
