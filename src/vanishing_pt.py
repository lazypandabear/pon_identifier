import cv2
import numpy as np
from sklearn.cluster import KMeans


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
    return lines

def get_line_angles(lines):
    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi
            angles.append([angle])
    return np.array(angles)

def cluster_lines_by_angle(lines, n_clusters=3):
    angles = get_line_angles(lines)
    kmeans = KMeans(n_clusters=n_clusters).fit(angles)
    clusters = [[] for _ in range(n_clusters)]
    for idx, line in enumerate(lines):
        cluster_id = kmeans.labels_[idx]
        clusters[cluster_id].append(line)
    return clusters

def compute_intersections(cluster):
    intersections = []
    for i, line1 in enumerate(cluster):
        for line2 in cluster[i+1:]:
            for x1, y1, x2, y2 in line1:
                for x3, y3, x4, y4 in line2:
                    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    if denom == 0:
                        continue
                    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                    intersections.append([px, py])
    return np.array(intersections)

def estimate_vanishing_point(intersections):
    if len(intersections) == 0:
        return None
    return np.median(intersections, axis=0)

# Example usage
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red for 3 clusters


image = cv2.imread("C:/Users/Dennis/VSCode/pon_identifier/data/13.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define green color range in HSV
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# Optional: Morphological operations to clean up the mask
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Apply the mask to original image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Use only the masked_image for edge and line detection

image = masked_image

# Define padding (e.g., 1000 pixels in each direction)
padding = 1000
h, w = image.shape[:2]
canvas = np.full((h + 2 * padding, w + 2 * padding, 3), 255, dtype=np.uint8)  # white background

# Place the image at the center of the canvas
canvas[padding:padding+h, padding:padding+w] = image.copy()



if image is None:
    raise ValueError("Could not load image")

lines = detect_lines(image)
clusters = cluster_lines_by_angle(lines, 3)
vanishing_points = []

for cluster in clusters:
    intersections = compute_intersections(cluster)
    vp = estimate_vanishing_point(intersections)
    vanishing_points.append(vp)

print("Vanishing Points:", vanishing_points)


# Draw vanishing points and lines
output = image.copy()
'''
# Draw detected lines for each cluster
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB for 3 clusters
for cluster_id, cluster in enumerate(clusters):
    for line in cluster:
        for x1, y1, x2, y2 in line:
            cv2.line(output, (x1, y1), (x2, y2), colors[cluster_id], 1)

# Draw vanishing points
for vp in vanishing_points:
    if vp is not None:
        x, y = map(int, vp)
        cv2.circle(output, (x, y), 8, (0, 255, 255), -1)  # yellow dot
'''

# Shift and draw lines
for cluster_id, cluster in enumerate(clusters):
    for line in cluster:
        for x1, y1, x2, y2 in line:
            cv2.line(canvas,
                     (x1 + padding, y1 + padding),
                     (x2 + padding, y2 + padding),
                     colors[cluster_id], 1)

# Draw vanishing points (if they exist)
for vp in vanishing_points:
    if vp is not None:
        x, y = map(int, vp)
        cv2.circle(canvas, (x + padding, y + padding), 8, (0, 0, 255), -1)  # red dots

cv2.imwrite("vanishing_points_full_view.jpg", canvas)
# Display the result
cv2.imshow("Vanishing Points", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
