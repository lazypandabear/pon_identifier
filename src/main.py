import cv2
import numpy as np
import yaml
import argparse
from detection import detect_port_face
from geometry import order_corners, compute_homography, overlay_grid, visualize_steps

def main():
    parser = argparse.ArgumentParser(description='Process an image to detect port face and overlay grid.')
    parser.add_argument('--config', type=str, default='/config/default.yaml', help='Path to config file')
    parser.add_argument('--image', type=str, help='Path to input image (overrides config)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load image
    img_path = args.image if args.image else config['image_path']
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image from {img_path}")

    # Detect port face
    keypoints, mask = detect_port_face(img, config)

    # Order corners and get orientation
    quad, orientation, br_parallelogram, br_mask_search = order_corners(None, keypoints, mask)
    print(f"Quad orientation: {orientation}")

    # Compute homography
    H = compute_homography(quad, config['grid'])

    # Overlay grid
    img_overlay = overlay_grid(img.copy(), H, config['grid'], quad=quad, keypoints=keypoints, mask=mask)

    # Visualize steps
    visualize_steps(img, keypoints, mask, br_parallelogram, br_mask_search, quad, config['grid'])

    # Save result
    cv2.imwrite('result.png', img_overlay)

if __name__ == '__main__':
    main() 