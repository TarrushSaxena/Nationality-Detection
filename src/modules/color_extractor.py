import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# Simple color name mapping (no webcolors dependency)
COLOR_NAMES = {
    (255, 0, 0): "Red",
    (0, 255, 0): "Green",
    (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow",
    (255, 165, 0): "Orange",
    (128, 0, 128): "Purple",
    (255, 192, 203): "Pink",
    (0, 255, 255): "Cyan",
    (255, 255, 255): "White",
    (0, 0, 0): "Black",
    (128, 128, 128): "Gray",
    (165, 42, 42): "Brown",
    (0, 128, 0): "Dark Green",
    (0, 0, 128): "Navy",
    (128, 0, 0): "Maroon",
    (255, 215, 0): "Gold",
    (192, 192, 192): "Silver",
    (245, 245, 220): "Beige",
    (75, 0, 130): "Indigo",
    (64, 224, 208): "Turquoise"
}

def get_closest_color_name(rgb):
    """Find the closest named color to the given RGB value."""
    r, g, b = rgb
    min_dist = float('inf')
    closest = "Unknown"
    
    for color_rgb, name in COLOR_NAMES.items():
        cr, cg, cb = color_rgb
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < min_dist:
            min_dist = dist
            closest = name
    
    return closest

class ColorExtractor:
    def __init__(self, k=3):
        self.k = k

    def extract_dress_color(self, full_image, face_bbox):
        """
        Extracts dominant color from the torso region (below face).
        face_bbox: (x, y, w, h)
        """
        x, y, w, h = face_bbox
        img_h, img_w, _ = full_image.shape

        # Heuristic: Torso is below the face
        torso_y_start = y + h
        torso_y_end = min(img_h, int(torso_y_start + (1.5 * h)))
        torso_x_start = max(0, x - int(0.2 * w))
        torso_x_end = min(img_w, x + w + int(0.2 * w))

        if torso_y_start >= torso_y_end or torso_x_start >= torso_x_end:
            return "Unknown"

        torso_roi = full_image[torso_y_start:torso_y_end, torso_x_start:torso_x_end]
        
        if torso_roi.size == 0:
            return "Unknown"

        # Convert BGR to RGB
        torso_roi = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2RGB)
        
        # Flatten pixels
        pixels = torso_roi.reshape(-1, 3)
        
        if len(pixels) < self.k:
            return "Unknown"
        
        # K-Means clustering
        try:
            kmeans = KMeans(n_clusters=self.k, n_init=5, random_state=42)
            kmeans.fit(pixels)
            
            # Get dominant color
            counts = Counter(kmeans.labels_)
            dominant_cluster = counts.most_common(1)[0][0]
            dominant_color = kmeans.cluster_centers_[dominant_cluster]
            
            dominant_color_int = tuple(map(int, dominant_color))
            color_name = get_closest_color_name(dominant_color_int)
            
            return color_name
        except Exception as e:
            return "Unknown"
