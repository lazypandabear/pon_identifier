import numpy as np
import cv2

def line_intersection(p1, p2, p3, p4):
    # Returns intersection point of lines p1-p2 and p3-p4, or None if parallel
    s = np.vstack([p1, p2, p3, p4])        # s for stacked
    h = np.hstack((s, np.ones((4, 1))))    # homogeneous
    l1 = np.cross(h[0], h[1])              # line 1
    l2 = np.cross(h[2], h[3])              # line 2
    x, y, z = np.cross(l1, l2)
    if z == 0:
        return None  # Parallel
    return np.array([x/z, y/z], dtype=np.float32)

def order_corners(quad, keypoints, mask=None):
    kp0, kp6, kp5 = keypoints
    tl = kp0
    tr = kp6
    # Default: use kp5 as BL, but override if mask is provided
    bl = kp5
    bl_contour = None
    br_parallelogram = tr + (bl - tl)
    br = br_parallelogram.copy()
    br_found = False
    last_valid_intersection = None  # ensure defined for debug print
    if mask is not None:
        h, w = mask.shape
        def clamp(pt):
            return np.array([np.clip(pt[0], 0, w-1), np.clip(pt[1], 0, h-1)], dtype=np.float32)
        br = clamp(br)
        tr = clamp(tr)
        # Fill in gaps in the mask contours using morphological operations
        kernel = np.ones((5, 100), np.uint8)  # Wider horizontally
        filled_mask = cv2.dilate(mask, kernel, iterations=1)
        filled_mask = cv2.erode(filled_mask, kernel, iterations=1)
        # Combine all contours into one mass
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        combined_contour = np.vstack(contours)
        # Find the bottom-left-most point of the combined contour
        min_y = np.max(combined_contour[:, 0, 1])  # largest y (lowest)
        bottom_points = combined_contour[combined_contour[:, 0, 1] == min_y]
        bl_contour = bottom_points[np.argmin(bottom_points[:, 0, 0]), 0]  # ensure shape (2,)
        bl = bl_contour.astype(np.float32)
        # Find the bottom edge of the combined contour (for reference)
        bottom_edge = combined_contour[combined_contour[:, 0, 1] == min_y]
        bottom_left = bottom_edge[np.argmin(bottom_edge[:, 0, 0])]
        bottom_right = bottom_edge[np.argmax(bottom_edge[:, 0, 0])]
        # Sweep angles from BL-TL direction clockwise
        line_bl_tl = tl - bl
        base_angle = np.arctan2(line_bl_tl[1], line_bl_tl[0])
        img_rotation = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.drawContours(img_rotation, [combined_contour], -1, (0, 255, 0), 2)
        # Draw the new BL point for debug
        cv2.circle(img_rotation, tuple(map(int, bl)), 10, (0, 255, 255), -1)
        intersection_pt = None
        last_valid_intersection = None
        best_intersection = None
        max_y = -np.inf
        max_shift = int(0.05 * w)  # 5% of image width, adaptive
        shift_step = 1
        angle_step = 5  # Increase angle step for clarity
        valid_candidates = []  # (x, y, slope)
        img_candidates = np.zeros_like(img_rotation)
        for deg in range(0, 91, angle_step):  # up to 90 degrees clockwise
            angle = base_angle + np.deg2rad(deg)
            rot_vec = np.array([np.cos(angle), np.sin(angle)]) * np.linalg.norm(line_bl_tl)
            p1 = bl
            p2 = bl + rot_vec
            # Draw fewer rotation lines for clarity
            cv2.line(img_rotation, tuple(map(int, bl)), tuple(map(int, p2)), (0, 0, 255), 1)
            found = False
            # Shift right
            for shift in range(0, max_shift+1, shift_step):
                tr_shifted = tr + np.array([shift, 0])
                br_shifted = br + np.array([shift, 0])
                inter = line_intersection(p1, p2, br_shifted, tr_shifted)
                if inter is not None:
                    x, y = int(round(inter[0])), int(round(inter[1]))
                    if 0 <= x < w and 0 <= y < h:
                        # Draw candidate point large and bright
                        cv2.circle(img_rotation, (x, y), 6, (0, 255, 255), -1)
                        cv2.circle(img_candidates, (x, y), 10, (0, 140, 255), -1)  # orange
                        if mask[y, x] > 0:  # Use the union mask (which includes cropped occupied class)
                            slope = (y - bl[1]) / (x - bl[0] + 1e-6)
                            valid_candidates.append((x, y, slope))
                            found = True
            # If not found, shift left
            if not found:
                for shift in range(-1, -max_shift-1, -shift_step):
                    tr_shifted = tr + np.array([shift, 0])
                    br_shifted = br + np.array([shift, 0])
                    inter = line_intersection(p1, p2, br_shifted, tr_shifted)
                    if inter is not None:
                        x, y = int(round(inter[0])), int(round(inter[1]))
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(img_rotation, (x, y), 6, (0, 255, 255), -1)
                            cv2.circle(img_candidates, (x, y), 10, (0, 140, 255), -1)  # orange
                            if mask[y, x] > 0:  # Use the union mask (which includes cropped occupied class)
                                slope = (y - bl[1]) / (x - bl[0] + 1e-6)
                                valid_candidates.append((x, y, slope))
        # Draw the mask contour thick and green for clarity
        cv2.drawContours(img_rotation, [combined_contour], -1, (0, 255, 0), 3)
        cv2.drawContours(img_candidates, [combined_contour], -1, (0, 255, 0), 3)
        # Slope filtering
        br = br  # fallback
        if valid_candidates:
            print("[DEBUG] All BR candidates (x, y, slope):")
            for cand in valid_candidates:
                print(f"    {cand}")
            # Find the minimum absolute slope among all candidates
            min_abs_slope = min(abs(s) for (_, _, s) in valid_candidates)
            epsilon = 0.01
            flattest_candidates = [(x, y, s) for (x, y, s) in valid_candidates if abs(abs(s) - min_abs_slope) < epsilon]
            selected_slope = None
            if flattest_candidates:
                # Among these, pick the one with the largest x (rightmost)
                # But ensure it's not too close to BL (at least 20 pixels right)
                valid_right_candidates = [(x, y, s) for (x, y, s) in flattest_candidates if x > bl[0] + 20]
                if valid_right_candidates:
                    br_candidate = max(valid_right_candidates, key=lambda pt: pt[0])
                else:
                    # Fallback to original parallelogram BR if no valid right candidates
                    br_candidate = (int(br_parallelogram[0]), int(br_parallelogram[1]), 0)
                br = np.array(br_candidate[:2], dtype=np.float32)
                intersection_pt = tuple(map(int, br_candidate[:2]))
                selected_slope = br_candidate[2]
                # Draw the flattest and lowest BR point very large and bold (yellow)
                cv2.circle(img_rotation, intersection_pt, 16, (0, 255, 255), -1)
                cv2.circle(img_candidates, intersection_pt, 20, (0, 255, 255), -1)
                # Draw a line from BL to the final BR point
                cv2.line(img_rotation, tuple(map(int, bl)), intersection_pt, (0, 255, 255), 2)
                cv2.line(img_candidates, tuple(map(int, bl)), intersection_pt, (0, 255, 255), 2)
                # Draw a line from TR to the final BR point
                cv2.line(img_rotation, tuple(map(int, tr)), intersection_pt, (0, 255, 255), 2)
                cv2.line(img_candidates, tuple(map(int, tr)), intersection_pt, (0, 255, 255), 2)
            # Extend the flattest slope to the rightmost mask point, up to tr[0] + margin
            if selected_slope is not None:
                x0, y0 = int(bl[0]), int(bl[1])
                margin = 10  # allow a small margin past TR
                x_end = min(int(tr[0]) + margin, w)
                rightmost_br = None
                for x in range(x0, x_end):
                    y = int(round(selected_slope * (x - x0) + y0))
                    if 0 <= y < h and mask[y, x] > 0:  # Use the union mask (which includes cropped occupied class)
                        rightmost_br = (x, y)
                # Post-process: allow a vertical tolerance to find a more rightward candidate
                tol = 10  # vertical tolerance in pixels
                if rightmost_br is not None:
                    x_br, y_br = rightmost_br
                    for x in range(x_br + 1, x_end):
                        for dy in range(-tol, tol + 1):
                            y = y_br + dy
                            if 0 <= y < h and mask[y, x] > 0:  # Use the union mask (which includes cropped occupied class)
                                rightmost_br = (x, y)
                if rightmost_br is not None:
                    br = np.array(rightmost_br, dtype=np.float32)
                    # Draw the extended rightmost BR point (cyan)
                    cv2.circle(img_rotation, rightmost_br, 18, (255, 255, 0), 3)
                    cv2.circle(img_candidates, rightmost_br, 24, (255, 255, 0), 3)
                    print(f"[DEBUG] Extended rightmost BR: {rightmost_br}")
            # Rightmost candidate (for comparison)
            rightmost_cand = max(valid_candidates, key=lambda pt: pt[0])
            rightmost_pt = tuple(map(int, rightmost_cand[:2]))
            cv2.circle(img_rotation, rightmost_pt, 12, (255, 0, 255), 2)  # magenta ring
            cv2.circle(img_candidates, rightmost_pt, 16, (255, 0, 255), 2)
            print(f"[DEBUG] Flattest and lowest candidate: {br_candidate if flattest_candidates else None}")
            print(f"[DEBUG] Rightmost candidate: {rightmost_cand}")
        cv2.imwrite('step_rotation.png', img_rotation)
        cv2.imwrite('step_candidates.png', img_candidates)
    br_mask_search = br.copy()  # This is now the correct BR
    # --- Proportional shift so BL is at kp5, and BR moves by the same vector ---
    shift = kp5 - bl
    bl_new = bl + shift  # This will be exactly kp5
    br_new = br + shift
    print(f"[DEBUG] br_parallelogram: {br_parallelogram}")
    print(f"[DEBUG] last_valid_intersection: {last_valid_intersection}")
    print(f"[DEBUG] final br used (before shift): {br}")
    print(f"[DEBUG] final bl used (before shift): {bl}")
    print(f"[DEBUG] shift vector: {shift}")
    print(f"[DEBUG] final br used (after shift): {br_new}")
    print(f"[DEBUG] final bl used (after shift): {bl_new}")
    ordered = np.array([tl, tr, br_new, bl_new], dtype=np.float32)
    # Check orientation (should be clockwise)
    def quad_area_signed(pts):
        return 0.5 * (
            (pts[0][0]*pts[1][1] + pts[1][0]*pts[2][1] + pts[2][0]*pts[3][1] + pts[3][0]*pts[0][1]) -
            (pts[1][0]*pts[0][1] + pts[2][0]*pts[1][1] + pts[3][0]*pts[2][1] + pts[0][0]*pts[3][1])
        )
    area_signed = quad_area_signed(ordered)
    if area_signed < 0:
        ordered = ordered[[0, 3, 2, 1]]
        orientation = 'counterclockwise (flipped to clockwise)'
    else:
        orientation = 'clockwise'
    return ordered, orientation, br_parallelogram, br_mask_search

def compute_homography(quad, grid_cfg):
    cols, rows = grid_cfg['cols'], grid_cfg['rows']
    src = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    H, _ = cv2.findHomography(src, quad)
    return H

def overlay_grid(img, H, grid_cfg, quad=None, keypoints=None, mask=None):
    cols, rows = grid_cfg['cols'], grid_cfg['rows']
    u, v = np.meshgrid(np.arange(cols+1), np.arange(rows+1))
    pts = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float32)
    pts = cv2.perspectiveTransform(pts.reshape(-1,1,2), H).reshape(-1,2)
    for p in pts:
        cv2.circle(img, tuple(map(int, p)), 2, (0,0,255), -1)
    # Draw quad corners with big text
    if quad is not None:
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, (pt, label) in enumerate(zip(quad, labels)):
            pt_int = tuple(map(int, pt))
            cv2.circle(img, pt_int, 10, (0,255,0), -1)
            cv2.putText(img, label, pt_int, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 4, cv2.LINE_AA)
    # Draw keypoints with big text
    if keypoints is not None:
        kp_labels = ['kp0', 'kp6', 'kp5']
        for pt, label in zip(keypoints, kp_labels):
            pt_int = tuple(map(int, pt))
            cv2.circle(img, pt_int, 10, (255,0,0), -1)
            cv2.putText(img, label, pt_int, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,0,0), 4, cv2.LINE_AA)
    # Overlay mask if provided
    if mask is not None:
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_col = cv2.applyColorMap(mask_resized.astype(np.uint8), cv2.COLORMAP_JET)
        if mask_col.shape[:2] != img.shape[:2]:
            mask_col = cv2.resize(mask_col, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.addWeighted(mask_col, 0.3, img, 0.7, 0, img)
    return img 

def visualize_steps(img, keypoints, mask, br_parallelogram, br_mask_search, final_quad, grid_cfg):
    # Step 1: Overlay keypoints and mask (no grid overlay)
    img1 = img.copy()
    # Draw all 7 keypoints with unique colors
    kp_colors = [
        (255, 0, 0),    # kp0 - blue
        (0, 255, 0),    # kp1 - green
        (0, 0, 255),    # kp2 - red
        (255, 255, 0),  # kp3 - cyan
        (255, 0, 255),  # kp4 - magenta
        (0, 255, 255),  # kp5 - yellow
        (128, 128, 255) # kp6 - light pink
    ]
    kp_labels = [f'kp{i}' for i in range(len(keypoints))]
    for i, (pt, label) in enumerate(zip(keypoints, kp_labels)):
        pt_int = tuple(map(int, pt))
        color = kp_colors[i % len(kp_colors)]
        cv2.circle(img1, pt_int, 10, color, -1)
        cv2.putText(img1, label, pt_int, cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4, cv2.LINE_AA)
    # Draw shifted BL and BR from the final quad
    bl_shifted = final_quad[3]
    br_shifted = final_quad[2]
    cv2.circle(img1, tuple(map(int, bl_shifted)), 10, (255, 255, 255), -1)
    cv2.putText(img1, 'BL_shifted', tuple(map(int, bl_shifted)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(img1, tuple(map(int, br_shifted)), 10, (255, 165, 0), -1)
    cv2.putText(img1, 'BR_shifted', tuple(map(int, br_shifted)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 3, cv2.LINE_AA)
    # Overlay mask
    if mask is not None:
        mask_resized = cv2.resize(mask, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_col = cv2.applyColorMap(mask_resized.astype(np.uint8), cv2.COLORMAP_JET)
        if mask_col.shape[:2] != img1.shape[:2]:
            mask_col = cv2.resize(mask_col, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.addWeighted(mask_col, 0.3, img1, 0.7, 0, img1)
    cv2.imwrite('step1_keypoints_mask.png', img1)

    # For all overlays, use the BL from the final quad
    bl = final_quad[3]

    # Step 2: Overlay parallelogram BR
    quad_parallelogram = np.array([keypoints[0], keypoints[1], br_parallelogram, bl], dtype=np.float32)
    H2 = compute_homography(quad_parallelogram, grid_cfg)
    img2 = overlay_grid(img.copy(), H2, grid_cfg, quad=quad_parallelogram, keypoints=keypoints, mask=mask)
    cv2.circle(img2, tuple(map(int, br_parallelogram)), 10, (0, 255, 255), -1)
    cv2.putText(img2, "BR_parallelogram", tuple(map(int, br_parallelogram)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    # Draw BL point at the new location
    cv2.circle(img2, tuple(map(int, bl)), 10, (0, 255, 0), -1)
    cv2.putText(img2, "BL", tuple(map(int, bl)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imwrite('step2_br_parallelogram.png', img2)

    # Step 3: Overlay mask search BR
    quad_mask_search = np.array([keypoints[0], keypoints[1], br_mask_search, bl], dtype=np.float32)
    H3 = compute_homography(quad_mask_search, grid_cfg)
    img3 = overlay_grid(img.copy(), H3, grid_cfg, quad=quad_mask_search, keypoints=keypoints, mask=mask)
    cv2.line(img3, tuple(map(int, br_parallelogram)), tuple(map(int, keypoints[1])), (255, 0, 255), 2)
    cv2.circle(img3, tuple(map(int, br_mask_search)), 10, (255, 0, 255), -1)
    cv2.putText(img3, "BR_mask_search", tuple(map(int, br_mask_search)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    # Draw BL point at the new location
    cv2.circle(img3, tuple(map(int, bl)), 10, (0, 255, 0), -1)
    cv2.putText(img3, "BL", tuple(map(int, bl)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imwrite('step3_br_mask_search.png', img3)

    # Step 4: Overlay final quad
    H4 = compute_homography(final_quad, grid_cfg)
    img4 = overlay_grid(img.copy(), H4, grid_cfg, quad=final_quad, keypoints=keypoints, mask=mask)
    # Draw BL point at the new location
    cv2.circle(img4, tuple(map(int, bl)), 10, (0, 255, 0), -1)
    cv2.putText(img4, "BL", tuple(map(int, bl)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imwrite('step4_final_quad.png', img4) 