import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

def detect_port_face(img, cfg):
    # Run pose model to get all keypoints
    pose_model = YOLO(str(Path(cfg['pose_model_path'])))
    pose_res = pose_model(img)[0]
    if pose_res.keypoints is None:
        raise RuntimeError("Pose model returned no keypoints.")
    kp = pose_res.keypoints.xy.cpu().numpy()[0]
    kp0, kp1, kp2, kp4, kp5, kp6 = kp[0], kp[1], kp[2], kp[4], kp[5], kp[6]
    v = kp1 - kp0
    kp6_new = kp6 + v
    keypoints = (kp0, kp6_new, kp5)

    # Run segmentation model
    seg_model = YOLO(str(Path(cfg['seg_model_path'])))
    seg_res = seg_model(img)[0]
    masks = seg_res.masks.data.cpu().numpy()
    classes = seg_res.boxes.cls.cpu().numpy().astype(int)

    # Process masks
    h, w = img.shape[:2]
    occupied_mask = np.zeros((h, w), dtype=np.uint8)
    vacant_mask = np.zeros((h, w), dtype=np.uint8)
    noport_mask = np.zeros((h, w), dtype=np.uint8)
    class4_mask = np.zeros((h, w), dtype=np.uint8)

    # Process each mask
    for i, (mask, cls) in enumerate(zip(masks, classes)):
        # Resize mask to match image dimensions
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8) * 255
        if cls == 0:  # occupied
            occupied_mask = cv2.bitwise_or(occupied_mask, mask)
        elif cls == 1:  # vacant
            vacant_mask = cv2.bitwise_or(vacant_mask, mask)
        elif cls == 2:  # noport
            noport_mask = cv2.bitwise_or(noport_mask, mask)
        elif cls == 4:  # class 4
            class4_mask = cv2.bitwise_or(class4_mask, mask)

    # Create union mask
    union = cv2.bitwise_or(occupied_mask, vacant_mask)
    union = cv2.bitwise_or(union, noport_mask)
    union = cv2.bitwise_or(union, class4_mask)

    # Save visualization of raw occupied masks
    cv2.imwrite('viz_1_occupied_raw.png', occupied_mask)
    contours, _ = cv2.findContours(occupied_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    viz_contour = np.zeros_like(occupied_mask)
    cv2.drawContours(viz_contour, contours, -1, 255, 2)
    cv2.imwrite('viz_1a_occupied_contour.png', viz_contour)

    # Crop occupied masks using parallelogram defined by keypoints
    kp0, kp6, kp5 = keypoints
    tl = kp0
    tr = kp6
    bl = kp5
    br = tr + (bl - tl)  # Calculate BR to maintain parallelogram shape

    # Create parallelogram mask for cropping
    para_mask = np.zeros_like(occupied_mask)
    para_pts = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.fillPoly(para_mask, [para_pts], 255)

    # Crop occupied masks
    occupied_cropped = cv2.bitwise_and(occupied_mask, para_mask)
    cv2.imwrite('viz_2_occupied_cropped.png', occupied_cropped)
    contours, _ = cv2.findContours(occupied_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    viz_contour = np.zeros_like(occupied_cropped)
    cv2.drawContours(viz_contour, contours, -1, 255, 2)
    cv2.imwrite('viz_2a_occupied_cropped_contour.png', viz_contour)

    # Create final union mask
    union = cv2.bitwise_or(occupied_cropped, vacant_mask)
    union = cv2.bitwise_or(union, noport_mask)
    union = cv2.bitwise_or(union, class4_mask)
    cv2.imwrite('viz_3_union_merged.png', union)
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    viz_contour = np.zeros_like(union)
    cv2.drawContours(viz_contour, contours, -1, 255, 2)
    cv2.imwrite('viz_3a_union_merged_contour.png', viz_contour)

    return keypoints, union 