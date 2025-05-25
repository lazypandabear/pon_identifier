import numpy as np

def compute_vanishing_points(quad_corners):
    """
    Compute vanishing points from quadrilateral corners.
    Args:
        quad_corners: numpy array of shape (4, 2) with corners ordered as [TL, TR, BR, BL]
    Returns:
        vp1, vp2: Two vanishing points as numpy arrays
    """
    # Compute lines from the quad corners
    lines = []
    # Horizontal lines (top and bottom)
    lines.append((quad_corners[0], quad_corners[1]))  # Top line
    lines.append((quad_corners[3], quad_corners[2]))  # Bottom line
    # Vertical lines (left and right)
    lines.append((quad_corners[0], quad_corners[3]))  # Left line
    lines.append((quad_corners[1], quad_corners[2]))  # Right line

    # Compute intersection of horizontal lines (vanishing point 1)
    h1 = np.cross(np.append(lines[0][0], 1), np.append(lines[0][1], 1))
    h2 = np.cross(np.append(lines[1][0], 1), np.append(lines[1][1], 1))
    vp1 = np.cross(h1, h2)
    vp1 = vp1[:2] / vp1[2]  # Convert from homogeneous to Cartesian coordinates

    # Compute intersection of vertical lines (vanishing point 2)
    v1 = np.cross(np.append(lines[2][0], 1), np.append(lines[2][1], 1))
    v2 = np.cross(np.append(lines[3][0], 1), np.append(lines[3][1], 1))
    vp2 = np.cross(v1, v2)
    vp2 = vp2[:2] / vp2[2]  # Convert from homogeneous to Cartesian coordinates

    return vp1, vp2
