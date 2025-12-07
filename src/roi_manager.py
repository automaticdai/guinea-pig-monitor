"""
ROI Manager - Manages Region of Interest (ROI) definitions
"""
import cv2
import numpy as np
import json
from pathlib import Path


class ROIManager:
    """Manages Region of Interest (ROI) definitions"""
    def __init__(self, zones_file='data/zones.json'):
        self.zones_file = zones_file
        self.zones = {
            'hay': None,
            'pellet': None,
            'sleeping': None,
            'drinking': None
        }
        self.current_roi_index = 0
        self.roi_names = ['hay', 'pellet', 'sleeping', 'drinking']
        self.drawing = False
        self.start_point = None
        self.current_roi = None
        self.load_zones()

    def load_zones(self):
        """Load zones from JSON file"""
        if Path(self.zones_file).exists():
            try:
                with open(self.zones_file, 'r') as f:
                    data = json.load(f)
                    for name in self.roi_names:
                        if name in data and data[name] is not None:
                            self.zones[name] = np.array(data[name], dtype=np.int32)
            except Exception as e:
                print(f"Error loading zones: {e}")

    def save_zones(self):
        """Save zones to JSON file"""
        # Ensure directory exists
        Path(self.zones_file).parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for name in self.roi_names:
            if self.zones[name] is not None:
                data[name] = self.zones[name].tolist()
            else:
                data[name] = None
        with open(self.zones_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Zones saved to {self.zones_file}")

    def reset(self):
        """Reset all zones"""
        for name in self.roi_names:
            self.zones[name] = None
        self.current_roi_index = 0
        print("All zones reset")

    def point_in_roi(self, point, roi_name):
        """Check if a point is inside a ROI"""
        if self.zones[roi_name] is None:
            return False
        return cv2.pointPolygonTest(self.zones[roi_name], point, False) >= 0

    def get_roi_for_point(self, point):
        """Get ROI name for a given point"""
        for name in self.roi_names:
            if self.point_in_roi(point, name):
                return name
        return None

    def start_drawing(self, x, y):
        """Start drawing a new ROI"""
        self.drawing = True
        self.start_point = (x, y)
        self.current_roi = None

    def update_drawing(self, x, y):
        """Update current ROI being drawn"""
        if self.drawing and self.start_point:
            x1, y1 = self.start_point
            self.current_roi = np.array([[x1, y1], [x, y1], [x, y], [x1, y]], dtype=np.int32)

    def finish_drawing(self, x, y):
        """Finish drawing and save ROI"""
        if self.drawing and self.start_point:
            x1, y1 = self.start_point
            roi = np.array([[x1, y1], [x, y1], [x, y], [x1, y]], dtype=np.int32)
            roi_name = self.roi_names[self.current_roi_index]
            self.zones[roi_name] = roi
            print(f"ROI '{roi_name}' defined: {roi.tolist()}")
            self.current_roi_index = (self.current_roi_index + 1) % len(self.roi_names)
            self.drawing = False
            self.start_point = None
            self.current_roi = None

    def draw_zones(self, frame):
        """Draw all zones on frame"""
        colors = {
            'hay': (0, 255, 0),      # Green
            'pellet': (255, 0, 0),    # Blue
            'sleeping': (0, 0, 255),  # Red
            'drinking': (255, 255, 0) # Cyan
        }
        
        for name, roi in self.zones.items():
            if roi is not None:
                cv2.polylines(frame, [roi], True, colors[name], 2)
                # Draw label
                if len(roi) > 0:
                    center = tuple(roi.mean(axis=0).astype(int))
                    cv2.putText(frame, name, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)
        
        # Draw current ROI being drawn
        if self.current_roi is not None:
            cv2.polylines(frame, [self.current_roi], True, (255, 255, 255), 2)

