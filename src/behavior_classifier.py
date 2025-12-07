"""
Behavior Classifier - Classifies behavior based on motion and location
"""
from collections import defaultdict


class BehaviorClassifier:
    """Classifies behavior based on motion and location"""
    def __init__(self, roi_manager, flow_analyzer):
        self.roi_manager = roi_manager
        self.flow_analyzer = flow_analyzer
        self.still_frames = defaultdict(int)  # Track ID -> consecutive still frames
        self.SLEEP_THRESHOLD = 0.02
        self.MOTION_THRESHOLD = 0.15
        self.SLEEP_FRAMES_REQUIRED = 300
        
        # Priority order for overlapping zones (higher index = higher priority)
        # When zones overlap, the behavior with highest priority is selected
        self.zone_priority = {
            'sleeping': 4,   # Highest priority
            'drinking': 3,
            'pellet': 2,
            'hay': 1,        # Lowest priority among zones
        }

    def classify(self, track_id, bbox):
        """Classify behavior for a tracked object
        
        Returns:
            tuple: (activity_behavior, motion_behavior) where:
                - activity_behavior: 'eat_hay', 'eat_pellet', 'drinking', 'sleeping', or '' (empty if no activity)
                - motion_behavior: 'idle' or 'activate'
        """
        # Get center point of bounding box
        x1, y1, x2, y2 = bbox
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # Get flow in region
        mean_flow = self.flow_analyzer.get_flow_in_region(bbox)
        
        # Check which ROI the center is in
        in_hay = self.roi_manager.point_in_roi(center, 'hay')
        in_pellet = self.roi_manager.point_in_roi(center, 'pellet')
        in_drinking = self.roi_manager.point_in_roi(center, 'drinking')
        in_sleeping = self.roi_manager.point_in_roi(center, 'sleeping')
        
        # Update still frames counter
        if mean_flow < self.SLEEP_THRESHOLD:
            self.still_frames[track_id] += 1
        else:
            self.still_frames[track_id] = 0
        
        # Determine motion state (idle or activate)
        if mean_flow > self.MOTION_THRESHOLD:
            motion_behavior = 'activate'
        else:
            motion_behavior = 'idle'
        
        # Behavior classification logic with priority handling for overlapping zones
        
        # Sleeping: very low motion for extended period (highest priority)
        # Sleeping is both an activity and idle motion
        if mean_flow < self.SLEEP_THRESHOLD and self.still_frames[track_id] >= self.SLEEP_FRAMES_REQUIRED:
            return ('sleeping', 'idle')
        
        # Collect all valid activity behaviors based on zones and motion
        # This handles overlapping zones by selecting the highest priority one
        valid_behaviors = []
        
        if in_drinking and mean_flow > self.MOTION_THRESHOLD:
            valid_behaviors.append(('drinking', self.zone_priority['drinking']))
        if in_pellet and mean_flow > self.MOTION_THRESHOLD:
            valid_behaviors.append(('eat_pellet', self.zone_priority['pellet']))
        if in_hay and mean_flow > self.MOTION_THRESHOLD:
            valid_behaviors.append(('eat_hay', self.zone_priority['hay']))
        
        # If multiple zones overlap, select the one with highest priority
        if valid_behaviors:
            # Sort by priority (descending) and return the highest
            valid_behaviors.sort(key=lambda x: x[1], reverse=True)
            activity_behavior = valid_behaviors[0][0]
            return (activity_behavior, motion_behavior)
        
        # No activity behavior (not in any zone), just return motion state
        return ('', motion_behavior)

