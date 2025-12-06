"""
Guinea Pig Behavior Monitoring System
Main program implementing real-time behavior classification
"""
import cv2
import numpy as np
import json
import time
import threading
from queue import Queue, Empty
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO
from bytetrack import ByteTracker


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


class OpticalFlowAnalyzer:
    """Analyzes optical flow for motion detection with GPU acceleration support"""
    def __init__(self, scale_factor=0.25, use_gpu=True):
        self.prev_gray = None
        self.flow = None
        self.flow_small = None  # Initialize flow_small
        self.flow_scale = (1.0, 1.0)  # Initialize flow_scale
        self.scale_factor = scale_factor  # More aggressive downscaling (0.25 = 1/16 pixels)
        self.use_gpu = use_gpu
        self.gpu_available = False
        self.prev_gpu = None
        self.curr_gpu = None
        
        # Try to initialize GPU if available
        if self.use_gpu:
            try:
                # Check if CUDA is available (requires opencv-contrib-python with CUDA)
                if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        self.gpu_available = True
                        print("GPU acceleration enabled for optical flow")
                    else:
                        print("CUDA not available, using CPU for optical flow")
                else:
                    print("OpenCV CUDA module not available, using CPU for optical flow")
            except Exception as e:
                print(f"GPU check failed ({e}), using CPU for optical flow")
                self.gpu_available = False

    def calculate_flow(self, frame):
        """Calculate optical flow with GPU acceleration if available"""
        # Aggressively downscale frame for faster computation
        h, w = frame.shape[:2]
        small_w = int(w * self.scale_factor)
        small_h = int(h * self.scale_factor)
        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.original_shape = frame.shape[:2]
            if self.gpu_available:
                try:
                    if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'GpuMat'):
                        self.prev_gpu = cv2.cuda.GpuMat()
                        self.curr_gpu = cv2.cuda.GpuMat()
                        self.prev_gpu.upload(self.prev_gray)
                    else:
                        self.gpu_available = False
                except Exception as e:
                    self.gpu_available = False
            return 0.0
        
        # Use GPU-accelerated optical flow if available
        if self.gpu_available:
            try:
                # Upload to GPU
                self.curr_gpu.upload(gray)
                
                # Check if cv2.cuda module is available
                if hasattr(cv2, 'cuda') and hasattr(cv2.cuda, 'FarnebackOpticalFlow'):
                    # Create GPU optical flow calculator
                    flow_calc = cv2.cuda.FarnebackOpticalFlow.create(
                        numLevels=3, pyrScale=0.5, fastPyramids=False,
                        winSize=15, numIterations=3, polyN=5, polySigma=1.2, flags=0
                    )
                    
                    # Calculate flow on GPU
                    flow_gpu = flow_calc.calc(self.prev_gpu, self.curr_gpu, None)
                    
                    # Download from GPU
                    flow = flow_gpu.download()
                    
                    # Update GPU buffer
                    self.prev_gpu, self.curr_gpu = self.curr_gpu, self.prev_gpu
                else:
                    # GPU module not available, fallback to CPU
                    self.gpu_available = False
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
            except Exception as e:
                # Fallback to CPU if GPU fails
                self.gpu_available = False
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        else:
            # CPU fallback with highly optimized parameters for speed
            # Reduced pyramid levels (2 instead of 3), smaller window (10 instead of 15)
            # Fewer iterations (2 instead of 3) for much faster computation
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 2, 10, 2, 5, 1.2, 0)
        
        # Calculate magnitude of flow vectors (faster with numpy)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mean_flow = np.mean(magnitude)
        
        # Scale flow back to original size if needed (for get_flow_in_region)
        if self.scale_factor != 1.0:
            h, w = self.original_shape
            # Only scale if we need it for region queries
            # For now, store at small size and scale on demand
            self.flow_small = flow
            self.flow_scale = (w / small_w, h / small_h)
            # Create placeholder at original size (will scale on demand)
            flow = None
        else:
            self.flow = flow
            self.flow_small = None
        
        self.prev_gray = gray
        
        return mean_flow

    def get_flow_in_region(self, bbox):
        """Get mean flow in a bounding box region"""
        # Check if flow has been calculated
        if not hasattr(self, 'flow_small') or (self.flow_small is None and self.flow is None):
            return 0.0
        
        # Use small flow if available, otherwise use full flow
        if hasattr(self, 'flow_small') and self.flow_small is not None:
            flow = self.flow_small
            # Scale bbox coordinates to small size
            x1, y1, x2, y2 = map(int, bbox)
            if hasattr(self, 'flow_scale') and self.flow_scale[0] > 0:
                x1 = int(x1 / self.flow_scale[0])
                y1 = int(y1 / self.flow_scale[1])
                x2 = int(x2 / self.flow_scale[0])
                y2 = int(y2 / self.flow_scale[1])
            else:
                return 0.0
        elif hasattr(self, 'flow') and self.flow is not None:
            flow = self.flow
            x1, y1, x2, y2 = map(int, bbox)
        else:
            return 0.0
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(flow.shape[1], x2)
        y2 = min(flow.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        region_flow = flow[y1:y2, x1:x2]
        magnitude = np.sqrt(region_flow[..., 0]**2 + region_flow[..., 1]**2)
        return np.mean(magnitude)


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
        """Classify behavior for a tracked object"""
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
        
        # Behavior classification logic with priority handling for overlapping zones
        
        # Sleeping: very low motion for extended period (highest priority)
        if mean_flow < self.SLEEP_THRESHOLD and self.still_frames[track_id] >= self.SLEEP_FRAMES_REQUIRED:
            return 'sleeping'
        
        # Collect all valid behaviors based on zones and motion
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
            return valid_behaviors[0][0]
        
        # Moving vs Idle (no zone match)
        if mean_flow > self.MOTION_THRESHOLD:
            return 'moving'
        else:
            return 'idle'


class StatisticsAggregator:
    """Aggregates behavior statistics per track ID"""
    def __init__(self, stats_file='data/statistics.json', initial_track_id=1):
        self.stats_file = stats_file
        self.stats = defaultdict(lambda: defaultdict(float))  # track_id -> behavior -> accumulated seconds
        self.current_behavior = defaultdict(str)  # track_id -> current behavior
        self.behavior_start_time = defaultdict(float)  # track_id -> start time in seconds (using time.time())
        self.track_start_time = defaultdict(float)  # track_id -> when tracking started for this track
        self.last_seen_time = defaultdict(float)  # track_id -> last time track was seen
        self.session_start_time = time.time()  # When this session started
        self.initial_track_id = initial_track_id
        self.load_stats()
        
        # Initialize track as lost - start counting immediately
        if initial_track_id not in self.track_start_time or self.track_start_time[initial_track_id] == 0:
            self.track_start_time[initial_track_id] = 0.0  # Start from session start
            self.last_seen_time[initial_track_id] = 0.0  # Never seen yet
            self.current_behavior[initial_track_id] = 'idle'  # Start as idle
            self.behavior_start_time[initial_track_id] = 0.0  # Start counting from beginning

    def update(self, track_id, behavior, current_time):
        """Update statistics for a track using actual elapsed time"""
        # Check if this is the first time we see this track (transition from lost to found)
        was_lost = (track_id in self.last_seen_time and self.last_seen_time[track_id] == 0.0)
        
        # Track when this track ID first appeared in this session
        if track_id not in self.track_start_time or self.track_start_time[track_id] == 0:
            self.track_start_time[track_id] = 0.0  # Start from session beginning
            was_lost = True
        
        # If track was lost and now found, finalize the lost time
        if was_lost and self.last_seen_time.get(track_id, 0.0) == 0.0:
            # Count time from start until now
            lost_duration = current_time
            if lost_duration > 20.0:
                # After 20 seconds, count as sleeping
                self.stats[track_id]['sleeping'] = self.stats[track_id].get('sleeping', 0) + (lost_duration - 20.0)
                self.stats[track_id]['idle'] = self.stats[track_id].get('idle', 0) + 20.0  # First 20 seconds as idle
            else:
                self.stats[track_id]['idle'] = self.stats[track_id].get('idle', 0) + lost_duration
        
        # Update last seen time
        self.last_seen_time[track_id] = current_time
        
        # If behavior changed, finalize previous behavior duration
        if self.current_behavior[track_id] != behavior:
            if self.current_behavior[track_id] and self.behavior_start_time[track_id] >= 0:
                # Calculate actual elapsed time
                prev_duration = current_time - self.behavior_start_time[track_id]
                if prev_duration > 0:
                    self.stats[track_id][self.current_behavior[track_id]] += prev_duration
            
            # Start new behavior
            self.current_behavior[track_id] = behavior
            self.behavior_start_time[track_id] = current_time
    
    def get_stats_with_lost_track_sleep(self, track_id, current_time, hay_zone_has_motion=False):
        """Get stats including sleep/eat time for lost tracks (>20 seconds)
        
        Args:
            track_id: Track ID
            current_time: Current elapsed time
            hay_zone_has_motion: True if hay zone has motion (indicates eating)
        """
        stats = self.stats[track_id].copy()
        
        # Check if track has never been seen (initial lost state)
        if track_id in self.last_seen_time and self.last_seen_time[track_id] == 0.0:
            # Track is still lost from the beginning
            if current_time > 20.0:
                # After 20 seconds, count as sleeping
                stats['sleeping'] = stats.get('sleeping', 0) + (current_time - 20.0)
                stats['idle'] = stats.get('idle', 0) + 20.0  # First 20 seconds as idle
            else:
                stats['idle'] = stats.get('idle', 0) + current_time
            return stats
        
        # Add current behavior duration if track is active
        if track_id in self.current_behavior and self.current_behavior[track_id]:
            if self.behavior_start_time[track_id] >= 0:
                current_behavior = self.current_behavior[track_id]
                current_duration = current_time - self.behavior_start_time[track_id]
                if current_duration > 0:
                    stats[current_behavior] = stats.get(current_behavior, 0) + current_duration
        
        # Check if track is lost (not seen for > 20 seconds)
        if track_id in self.last_seen_time and self.last_seen_time[track_id] > 0:
            time_since_last_seen = current_time - self.last_seen_time[track_id]
            if time_since_last_seen > 20.0:  # 20 seconds threshold
                # Only count time beyond 20 seconds
                lost_time = time_since_last_seen - 20.0
                
                # If hay zone has motion, assume eating; otherwise assume sleeping
                if hay_zone_has_motion:
                    # Count as eating (add to eat_hay)
                    stats['eat_hay'] = stats.get('eat_hay', 0) + lost_time
                else:
                    # Count as sleeping
                    stats['sleeping'] = stats.get('sleeping', 0) + lost_time
        
        return stats

    def get_stats_for_display(self, track_id, current_time, hay_zone_has_motion=False):
        """Get statistics formatted for bar chart display"""
        # Always show stats for initial track ID, even if never seen
        if track_id not in self.stats and track_id not in self.current_behavior:
            if track_id not in self.track_start_time and track_id != self.initial_track_id:
                return None
        
        # Get stats including lost track sleep/eat time
        stats = self.get_stats_with_lost_track_sleep(track_id, current_time, hay_zone_has_motion)
        
        # Calculate totals
        eat_total = stats.get('eat_hay', 0) + stats.get('eat_pellet', 0)
        sleep_total = stats.get('sleeping', 0)
        drink_total = stats.get('drinking', 0)
        idle_total = stats.get('idle', 0)
        moving_total = stats.get('moving', 0)
        
        # Calculate total tracked time
        # Total time is the sum of all behaviors (they are mutually exclusive)
        total_time = eat_total + sleep_total + drink_total + idle_total + moving_total
        
        # If no stats yet, use current session time
        if total_time == 0:
            if track_id in self.track_start_time:
                if self.track_start_time[track_id] > 0:
                    total_time = current_time - self.track_start_time[track_id]
                else:
                    total_time = current_time
            else:
                total_time = current_time
        
        # Calculate percentages in two groups:
        # Group 1: sleeping + eating + drinking = 100%
        # Group 2: idle + moving = 100%
        group1_total = sleep_total + eat_total + drink_total
        group2_total = idle_total + moving_total
        
        if group1_total > 0:
            eat_pct = (eat_total / group1_total) * 100
            sleep_pct = (sleep_total / group1_total) * 100
            drink_pct = (drink_total / group1_total) * 100
        else:
            eat_pct = sleep_pct = drink_pct = 0.0
        
        if group2_total > 0:
            idle_pct = (idle_total / group2_total) * 100
            moving_pct = (moving_total / group2_total) * 100
        else:
            idle_pct = moving_pct = 0.0
        
        # Format as minutes
        eat_min = eat_total / 60.0
        sleep_min = sleep_total / 60.0
        drink_min = drink_total / 60.0
        idle_min = idle_total / 60.0
        moving_min = moving_total / 60.0
        
        return {
            'eat': {'min': eat_min, 'pct': eat_pct},
            'sleep': {'min': sleep_min, 'pct': sleep_pct},
            'drink': {'min': drink_min, 'pct': drink_pct},
            'idle': {'min': idle_min, 'pct': idle_pct},
            'moving': {'min': moving_min, 'pct': moving_pct},
            'total_time': total_time
        }

    def get_final_report(self, current_time, hay_zone_has_motion=False):
        """Get final statistics report including lost track sleep/eat time"""
        report = {}
        all_track_ids = set(self.stats.keys()) | set(self.current_behavior.keys())
        
        for track_id in all_track_ids:
            # Use the method that includes lost track sleep/eat time
            stats = self.get_stats_with_lost_track_sleep(track_id, current_time, hay_zone_has_motion)
            report[track_id] = stats
        
        return report
    
    def save_stats(self, current_time):
        """Save statistics to JSON file"""
        try:
            # Finalize current behaviors before saving
            temp_stats = {}
            for track_id in set(self.stats.keys()) | set(self.current_behavior.keys()):
                if track_id in self.current_behavior and self.current_behavior[track_id]:
                    if self.behavior_start_time[track_id] >= 0:
                        current_behavior = self.current_behavior[track_id]
                        current_duration = current_time - self.behavior_start_time[track_id]
                        if current_duration > 0:
                            temp_stats[track_id] = self.stats[track_id].copy()
                            temp_stats[track_id][current_behavior] = temp_stats[track_id].get(current_behavior, 0) + current_duration
                        else:
                            temp_stats[track_id] = self.stats[track_id].copy()
                    else:
                        temp_stats[track_id] = self.stats[track_id].copy()
                else:
                    temp_stats[track_id] = self.stats[track_id].copy()
            
            # Note: We don't need to track total_accumulated_time separately
            # because total_time is calculated as sum of all behaviors
            # The stats themselves already contain accumulated time from all sessions
            
            # Get final stats for all tracks
            data = {
                'session_start': self.session_start_time,
                'last_save_time': time.time(),
                'save_elapsed_time': current_time,
                'tracks': {}
            }
            
            all_track_ids = set(self.stats.keys()) | set(self.current_behavior.keys())
            for track_id in all_track_ids:
                # Save accumulated stats including finalized current behavior
                track_data = {
                    'stats': dict(temp_stats.get(track_id, self.stats[track_id])),
                    'track_start_time': 0.0,  # Always reset to 0 for next session
                    'last_seen_time': 0.0,  # Reset for next session
                    'current_behavior': '',  # Session-specific, don't save
                    'behavior_start_time': 0.0  # Session-specific, don't save
                }
                data['tracks'][str(track_id)] = track_data
            
            with open(self.stats_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving statistics: {e}")
            return False
    
    def load_stats(self):
        """Load statistics from JSON file"""
        if not Path(self.stats_file).exists():
            return
        
        try:
            with open(self.stats_file, 'r') as f:
                data = json.load(f)
            
            # Load track statistics
            if 'tracks' in data:
                for track_id_str, track_data in data['tracks'].items():
                    track_id = int(track_id_str)
                    if 'stats' in track_data:
                        self.stats[track_id] = defaultdict(float, track_data['stats'])
                    if 'track_start_time' in track_data:
                        # Reset track_start_time to 0 for current session
                        self.track_start_time[track_id] = 0.0
                    if 'last_seen_time' in track_data:
                        # Reset last_seen_time - will be updated when track is seen
                        self.last_seen_time[track_id] = 0.0
                    # Note: current_behavior and behavior_start_time are session-specific,
                    # so we don't restore them - they'll start fresh
                    # Note: total_accumulated_time is not needed - total is sum of behaviors
            
            print(f"Loaded statistics from {self.stats_file}")
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    def clear_stats(self):
        """Clear all statistics"""
        self.stats.clear()
        self.current_behavior.clear()
        self.behavior_start_time.clear()
        self.track_start_time.clear()
        self.last_seen_time.clear()
        self.session_start_time = time.time()
        
        # Also delete the stats file
        try:
            if Path(self.stats_file).exists():
                Path(self.stats_file).unlink()
                print("Statistics cleared and file deleted")
            else:
                print("Statistics cleared")
        except Exception as e:
            print(f"Error deleting statistics file: {e}")


class GuineaPigMonitor:
    """Main monitoring system"""
    def __init__(self, model_path='model/yolov11.pt', rtsp_url=None, single_object=True):
        self.model = YOLO(model_path)
        # Adjust tracker parameters for single object tracking
        # Higher match_thresh = stricter matching, prevents track fragmentation
        # Longer track_buffer = keeps tracks alive longer when temporarily lost
        self.tracker = ByteTracker(
            frame_rate=30,
            track_thresh=0.3,      # Lower threshold to keep tracks
            high_thresh=0.5,        # Lower high threshold
            match_thresh=0.7,      # Stricter matching to prevent duplicates
            track_buffer=60        # Keep tracks alive longer (2 seconds at 30fps)
        )
        self.single_object = single_object
        self.FORCED_TRACK_ID = 1  # Always use track ID 1 for single object
        self.roi_manager = ROIManager()
        self.flow_analyzer = OpticalFlowAnalyzer()
        self.behavior_classifier = BehaviorClassifier(self.roi_manager, self.flow_analyzer)
        self.stats_aggregator = StatisticsAggregator()
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_id = 0
        self.start_time = time.time()  # Track start time for accurate statistics
        self.window_name = "Guinea Pig Monitor"
        self.show_stats = True  # Toggle for showing statistics HUD
        
        # Threaded frame reading
        self.frame_queue = Queue(maxsize=2)  # Small buffer to prevent lag
        self.frame_reader_thread = None
        self.stop_frame_reader = False
        
        # FPS tracking
        self.fps = 0.0
        self.fps_frame_count = 0
        self.fps_start_time = time.time()
        self.fps_update_interval = 1.0  # Update FPS every second
        
        # Timing debug
        self.timing_enabled = True
        self.timing_stats = {
            'frame_read': [],
            'yolo_inference': [],
            'detection_processing': [],
            'single_object_cleanup': [],
            'tracking': [],
            'optical_flow': [],
            'frame_copy': [],
            'roi_drawing': [],
            'bbox_drawing': [],
            'behavior_classification': [],
            'statistics_update': [],
            'drawing': [],
            'display': [],
            'total': []
        }
        self.timing_frame_count = 0
        self.timing_report_interval = 60  # Report every 60 frames
        
        # Mouse callback setup
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def _frame_reader_worker(self):
        """Worker thread that continuously reads frames from the stream"""
        consecutive_failures = 0
        while not self.stop_frame_reader:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue
            
            read_start = time.time()
            
            # Try grab() + retrieve() which can be faster than read()
            # First grab the frame metadata (non-blocking)
            grabbed = self.cap.grab()
            
            if not grabbed:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            
            # Now retrieve the actual frame
            ret, frame = self.cap.retrieve()
            read_time = time.time() - read_start
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # Store timing if enabled
            if self.timing_enabled:
                self.timing_stats['frame_read'].append(read_time)
            
            # Put frame in queue (drop old frames if queue is full to prevent lag)
            try:
                self.frame_queue.put_nowait((frame, read_start))
            except:
                # Queue full, drop oldest frame to always use latest
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame, read_start))
                except:
                    pass
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI editing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_manager.start_drawing(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.roi_manager.drawing:
                self.roi_manager.update_drawing(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_manager.finish_drawing(x, y)

    def process_frame(self, frame, frame_start=None):
        """Process a single frame
        Args:
            frame: Input frame
            frame_start: Start time for total timing (if None, uses current time)
        """
        if frame_start is None:
            frame_start = time.time()
        
        # Run YOLO detection
        yolo_start = time.time()
        results = self.model(frame, verbose=False)
        yolo_time = time.time() - yolo_start
        if self.timing_enabled:
            self.timing_stats['yolo_inference'].append(yolo_time)
        
        # Process detections
        det_start = time.time()
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])
        
        # Single object mode: use only the highest confidence detection
        if self.single_object and len(detections) > 1:
            detections.sort(key=lambda d: d[4], reverse=True)  # Sort by confidence
            detections = [detections[0]]  # Keep only the best detection
        det_time = time.time() - det_start
        if self.timing_enabled:
            self.timing_stats['detection_processing'].append(det_time)
        
        # Single object mode: Prevent new track IDs by intercepting STrack counter
        if self.single_object:
            # Save current counter and force it to stay at 1
            from bytetrack import STrack
            original_count = STrack._count
            STrack._count = 0  # Reset to 0 so next_id() returns 1
        
        # Update tracker
        tracking_start = time.time()
        tracks = self.tracker.update(detections, self.frame_id)
        tracking_time = time.time() - tracking_start
        if self.timing_enabled:
            self.timing_stats['tracking'].append(tracking_time)
        
        # Single object mode: FORCE all tracks to use ID 1
        cleanup_start = time.time()
        if self.single_object:
            # Force ALL tracks to use ID 1 immediately
            for track in tracks:
                track.track_id = self.FORCED_TRACK_ID
            
            # If multiple tracks exist, keep only the longest one
            if len(tracks) > 1:
                tracks.sort(key=lambda t: (-t.tracklet_len, t.start_frame))
                tracks = [tracks[0]]  # Keep only the best track
            
            # Aggressively clean up ALL tracker internal state - only keep ID 1
            self.tracker.tracked_tracks = [t for t in self.tracker.tracked_tracks if t.track_id == self.FORCED_TRACK_ID]
            self.tracker.lost_tracks = [t for t in self.tracker.lost_tracks if t.track_id == self.FORCED_TRACK_ID]
            self.tracker.removed_tracks = []
            
            # Force all remaining tracks to use ID 1 (in case any slipped through)
            for track in self.tracker.tracked_tracks:
                track.track_id = self.FORCED_TRACK_ID
            for track in self.tracker.lost_tracks:
                track.track_id = self.FORCED_TRACK_ID
            
            # Ensure STrack counter doesn't increment beyond 1
            STrack._count = 0  # Keep at 0 so next_id() always returns 1
        cleanup_time = time.time() - cleanup_start
        if self.timing_enabled and cleanup_time > 0:
            self.timing_stats['single_object_cleanup'].append(cleanup_time)
        
        # Calculate optical flow
        flow_start = time.time()
        mean_flow = self.flow_analyzer.calculate_flow(frame)
        flow_time = time.time() - flow_start
        if self.timing_enabled:
            self.timing_stats['optical_flow'].append(flow_time)
        
        # Draw detections and tracks
        copy_start = time.time()
        annotated_frame = frame.copy()
        copy_time = time.time() - copy_start
        if self.timing_enabled:
            self.timing_stats['frame_copy'].append(copy_time)
        
        # Draw ROIs
        roi_start = time.time()
        self.roi_manager.draw_zones(annotated_frame)
        roi_time = time.time() - roi_start
        if self.timing_enabled:
            self.timing_stats['roi_drawing'].append(roi_time)
        
        # Draw tracks and behaviors
        behavior_total_time = 0.0
        stats_total_time = 0.0
        bbox_drawing_time = 0.0
        for track in tracks:
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track.track_id
            
            # Classify behavior
            behavior_start = time.time()
            behavior = self.behavior_classifier.classify(track_id, bbox)
            behavior_time = time.time() - behavior_start
            behavior_total_time += behavior_time
            
            # Update statistics using actual elapsed time
            stats_start = time.time()
            current_time = time.time() - self.start_time
            self.stats_aggregator.update(track_id, behavior, current_time)
            stats_time = time.time() - stats_start
            stats_total_time += stats_time
            
            # Draw bounding box
            bbox_start = time.time()
            color = self.get_color_for_track(track_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and behavior
            label = f"ID {track_id}: {behavior}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            bbox_drawing_time += time.time() - bbox_start
        
        if self.timing_enabled:
            if behavior_total_time > 0:
                self.timing_stats['behavior_classification'].append(behavior_total_time)
            if stats_total_time > 0:
                self.timing_stats['statistics_update'].append(stats_total_time)
            if bbox_drawing_time > 0:
                self.timing_stats['bbox_drawing'].append(bbox_drawing_time)
        
        # Update FPS
        self.fps_frame_count += 1
        current_fps_time = time.time()
        elapsed_fps_time = current_fps_time - self.fps_start_time
        if elapsed_fps_time >= self.fps_update_interval:
            self.fps = self.fps_frame_count / elapsed_fps_time
            self.fps_frame_count = 0
            self.fps_start_time = current_fps_time
        
        # Draw FPS
        drawing_start = time.time()
        self.draw_fps(annotated_frame)
        
        # Draw statistics HUD (if enabled)
        if self.show_stats:
            self.draw_hud(annotated_frame)
        drawing_time = time.time() - drawing_start
        if self.timing_enabled:
            self.timing_stats['drawing'].append(drawing_time)
        
        total_time = time.time() - frame_start
        if self.timing_enabled:
            self.timing_stats['total'].append(total_time)
            self.timing_frame_count += 1
            
            # Print timing report periodically
            if self.timing_frame_count % self.timing_report_interval == 0:
                self.print_timing_report()
        
        self.frame_id += 1
        return annotated_frame

    def get_color_for_track(self, track_id):
        """Get a consistent color for a track ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        return colors[track_id % len(colors)]
    
    def print_timing_report(self):
        """Print timing breakdown report"""
        print("\n" + "="*60)
        print("TIMING BREAKDOWN (last {} frames):".format(self.timing_report_interval))
        print("="*60)
        
        total_avg = np.mean(self.timing_stats['total']) if self.timing_stats['total'] else 0
        if total_avg == 0:
            print("No timing data available")
            print("="*60 + "\n")
            return
        
        print(f"{'Component':<25} {'Avg (ms)':<12} {'Max (ms)':<12} {'% of Total':<12}")
        print("-"*60)
        
        # Calculate sum of all component averages
        component_sum = 0.0
        component_data = []
        
        for component, times in self.timing_stats.items():
            if component == 'total' or not times:
                continue
            avg = np.mean(times) * 1000  # Convert to ms
            max_time = np.max(times) * 1000 if times else 0
            pct = (avg / 1000) / total_avg * 100 if total_avg > 0 else 0
            component_sum += avg / 1000  # Keep in seconds for sum
            component_data.append((component, avg, max_time, pct))
        
        # Sort by percentage (descending)
        component_data.sort(key=lambda x: x[3], reverse=True)
        
        # Print components
        for component, avg, max_time, pct in component_data:
            print(f"{component:<25} {avg:<12.2f} {max_time:<12.2f} {pct:<12.1f}")
        
        if self.timing_stats['total']:
            total_avg_ms = total_avg * 1000
            total_max_ms = np.max(self.timing_stats['total']) * 1000
            print("-"*60)
            print(f"{'TOTAL':<25} {total_avg_ms:<12.2f} {total_max_ms:<12.2f} {'100.0':<12}")
            print(f"Effective FPS: {1.0/total_avg:.1f}")
        print("="*60 + "\n")
        
        # Clear stats for next interval
        for key in self.timing_stats:
            self.timing_stats[key] = []
    
    def draw_fps(self, frame):
        """Draw FPS in top-left corner"""
        fps_text = f"FPS: {self.fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)  # Green color
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        padding = 5
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (padding, padding),
                     (padding + text_width + padding * 2, padding + text_height + padding * 2),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw FPS text
        cv2.putText(frame, fps_text,
                   (padding * 2, padding + text_height),
                   font, font_scale, color, thickness)

    def draw_hud(self, frame):
        """Draw statistics HUD overlay in top right corner with bars"""
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # Increased from 0.4 for better readability
        thickness = 2  # Increased from 1 for better visibility
        line_height = 25  # Increased from 18 to accommodate larger font
        padding = 12  # Increased from 10
        bar_width = 180  # Increased from 150 for better visibility
        bar_height = 16  # Increased from 12 for better visibility
        bar_spacing = 3  # Increased from 2
        
        # Get current elapsed time
        current_time = time.time() - self.start_time
        
        # Get all track IDs (active + lost + initial track)
        all_track_ids = set()
        # Active tracks
        for track in self.tracker.tracked_tracks:
            if track.is_activated:
                all_track_ids.add(track.track_id)
        # Lost tracks (from stats aggregator)
        all_track_ids.update(self.stats_aggregator.stats.keys())
        all_track_ids.update(self.stats_aggregator.current_behavior.keys())
        # Always include initial track ID (even if never seen)
        all_track_ids.add(self.stats_aggregator.initial_track_id)
        
        if not all_track_ids:
            return
        
        # Check if hay zone has motion (for lost track eating detection)
        hay_zone_has_motion = False
        if self.roi_manager.zones['hay'] is not None:
            # Get bounding box of hay zone
            hay_roi = self.roi_manager.zones['hay']
            x_coords = hay_roi[:, 0]
            y_coords = hay_roi[:, 1]
            hay_bbox = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
            # Check flow in hay zone
            hay_flow = self.flow_analyzer.get_flow_in_region(hay_bbox)
            hay_zone_has_motion = hay_flow > 0.15  # Same threshold as behavior classification
        
        # For each track, get stats and draw
        for track_id in sorted(all_track_ids):
            # Check if track is lost
            is_lost = track_id not in [t.track_id for t in self.tracker.tracked_tracks if t.is_activated]
            lost_indicator = " [LOST]" if is_lost else ""
            
            # Get stats with hay zone motion info (only relevant for lost tracks)
            stats_data = self.stats_aggregator.get_stats_for_display(
                track_id, current_time, 
                hay_zone_has_motion=hay_zone_has_motion if is_lost else False
            )
            if stats_data is None:
                continue
            
            # Calculate dimensions for this track's stats
            num_metrics = 5  # eat, sleep, drink, idle, moving
            gap_before_moving = line_height // 2  # Half line height gap
            track_header_height = line_height
            track_height = track_header_height + (num_metrics * (line_height + bar_spacing)) + gap_before_moving + padding
            
            # Layout constants
            text_width = 220  # Fixed width for text (increased for better spacing)
            bar_spacing_x = 15  # Spacing between text and bar
            
            # Calculate position (stack tracks vertically)
            y_start = padding
            for tid in sorted(all_track_ids):
                if tid == track_id:
                    break
                tid_is_lost = tid not in [t.track_id for t in self.tracker.tracked_tracks if t.is_activated]
                tid_stats = self.stats_aggregator.get_stats_for_display(
                    tid, current_time,
                    hay_zone_has_motion=hay_zone_has_motion if tid_is_lost else False
                )
                if tid_stats:
                    tid_height = track_header_height + (num_metrics * (line_height + bar_spacing)) + gap_before_moving + padding
                    y_start += tid_height + padding
            
            # Calculate width needed
            total_width = text_width + bar_spacing_x + bar_width + padding * 2
            x_start = frame_width - total_width - padding
            
            # Draw semi-transparent background for this track
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (x_start, y_start), 
                         (x_start + total_width, y_start + track_height),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw track header (larger font)
            header_text = f"ID {track_id}{lost_indicator}"
            cv2.putText(frame, header_text, (x_start + padding, y_start + line_height),
                       font, font_scale + 0.2, (255, 255, 255), thickness + 1)
            
            # Draw each metric with bar
            y_offset = y_start + track_header_height + padding + 5
            metrics = [
                ('eat', stats_data['eat'], (0, 255, 0)),      # Green
                ('sleep', stats_data['sleep'], (0, 0, 255)),  # Red
                ('drink', stats_data['drink'], (255, 255, 0)), # Cyan
            ]
            
            for metric_name, metric_data, bar_color in metrics:
                # Calculate text position (vertically centered with bar)
                text_y = y_offset + int(bar_height / 2) + int(line_height / 3)
                
                # Draw metric label and values
                label_text = f"{metric_name}: {metric_data['min']:.1f}m ({metric_data['pct']:.0f}%)"
                cv2.putText(frame, label_text, (x_start + padding, text_y),
                           font, font_scale, (255, 255, 255), thickness)
                
                # Draw bar (aligned with text)
                bar_x = x_start + text_width + bar_spacing_x
                bar_y = y_offset
                bar_fill_width = int(bar_width * (metric_data['pct'] / 100.0))
                
                # Bar background
                cv2.rectangle(frame, 
                             (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height),
                             (50, 50, 50), -1)
                
                # Bar fill
                if bar_fill_width > 0:
                    cv2.rectangle(frame, 
                                 (bar_x, bar_y), 
                                 (bar_x + bar_fill_width, bar_y + bar_height),
                                 bar_color, -1)
                
                # Bar border
                cv2.rectangle(frame, 
                             (bar_x, bar_y), 
                             (bar_x + bar_width, bar_y + bar_height),
                             (200, 200, 200), 1)
                
                y_offset += line_height + bar_spacing
            
            # Add gap before idle
            y_offset += gap_before_moving
            
            # Draw idle and moving metrics separately after the gap
            # First, draw idle
            metric_name = 'idle'
            metric_data = stats_data['idle']
            bar_color = (128, 128, 128)  # Gray
            
            # Calculate text position (vertically centered with bar)
            text_y = y_offset + int(bar_height / 2) + int(line_height / 3)
            
            # Draw metric label and values
            label_text = f"{metric_name}: {metric_data['min']:.1f}m ({metric_data['pct']:.0f}%)"
            cv2.putText(frame, label_text, (x_start + padding, text_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw bar (aligned with text)
            bar_x = x_start + text_width + bar_spacing_x
            bar_y = y_offset
            bar_fill_width = int(bar_width * (metric_data['pct'] / 100.0))
            
            # Bar background
            cv2.rectangle(frame, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Bar fill
            if bar_fill_width > 0:
                cv2.rectangle(frame, 
                             (bar_x, bar_y), 
                             (bar_x + bar_fill_width, bar_y + bar_height),
                             bar_color, -1)
            
            # Bar border
            cv2.rectangle(frame, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (200, 200, 200), 1)
            
            y_offset += line_height + bar_spacing
            
            # Then, draw moving
            metric_name = 'moving'
            metric_data = stats_data['moving']
            bar_color = (255, 165, 0)  # Orange
            
            # Calculate text position (vertically centered with bar)
            text_y = y_offset + int(bar_height / 2) + int(line_height / 3)
            
            # Draw metric label and values
            label_text = f"{metric_name}: {metric_data['min']:.1f}m ({metric_data['pct']:.0f}%)"
            cv2.putText(frame, label_text, (x_start + padding, text_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw bar (aligned with text)
            bar_x = x_start + text_width + bar_spacing_x
            bar_y = y_offset
            bar_fill_width = int(bar_width * (metric_data['pct'] / 100.0))
            
            # Bar background
            cv2.rectangle(frame, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Bar fill
            if bar_fill_width > 0:
                cv2.rectangle(frame, 
                             (bar_x, bar_y), 
                             (bar_x + bar_fill_width, bar_y + bar_height),
                             bar_color, -1)
            
            # Bar border
            cv2.rectangle(frame, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (200, 200, 200), 1)

    def run(self):
        """Main loop"""
        # Open video source
        if self.rtsp_url:
            self.cap = cv2.VideoCapture(self.rtsp_url)
        else:
            # Try to open default camera or ask for RTSP URL
            print("No RTSP URL provided. Please provide RTSP URL or use default camera.")
            rtsp_url = input("Enter RTSP URL (or press Enter for default camera): ").strip()
            if rtsp_url:
                self.cap = cv2.VideoCapture(rtsp_url)
            else:
                self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Set buffer size to 1 to minimize latency and frame accumulation
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Start threaded frame reader
        self.stop_frame_reader = False
        self.frame_reader_thread = threading.Thread(target=self._frame_reader_worker, daemon=True)
        self.frame_reader_thread.start()
        
        print("Starting monitoring...")
        print("Controls:")
        print("  - Left-click + drag: Define ROI")
        print("  - Press 'r': Reset all ROIs")
        print("  - Press 's': Save ROIs to zones.json")
        print("  - Press 'h': Toggle statistics HUD")
        print("  - Press 'c': Clear statistics")
        print("  - Press 'q': Quit")
        
        last_save_time = time.time()
        save_interval = 60.0  # Auto-save every 60 seconds
        
        try:
            while True:
                # Get frame from queue (non-blocking with timeout)
                try:
                    frame, read_start = self.frame_queue.get(timeout=1.0)
                except Empty:
                    print("Timeout waiting for frame")
                    continue
                
                # Process frame (pass frame_start for accurate total timing)
                frame_start = read_start  # Total time starts from frame read
                annotated_frame = self.process_frame(frame, frame_start)
                
                # Display frame
                display_start = time.time()
                cv2.imshow(self.window_name, annotated_frame)
                display_time = time.time() - display_start
                if self.timing_enabled:
                    self.timing_stats['display'].append(display_time)
                    
                    # Update total time to include frame_read and display
                    # Total should be from frame_start (which is read_start) to after display
                    total_with_overhead = time.time() - frame_start
                    # Replace the last total (which was process_frame only) with full total
                    if self.timing_stats['total']:
                        self.timing_stats['total'][-1] = total_with_overhead
                
                # Auto-save statistics periodically
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    current_elapsed = time.time() - self.start_time
                    self.stats_aggregator.save_stats(current_elapsed)
                    last_save_time = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.roi_manager.reset()
                elif key == ord('s'):
                    self.roi_manager.save_zones()
                elif key == ord('h'):
                    self.show_stats = not self.show_stats
                    print(f"Statistics HUD: {'ON' if self.show_stats else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Save statistics before exit
            current_time = time.time() - self.start_time
            print("\nSaving statistics...")
            self.stats_aggregator.save_stats(current_time)
            
            # Stop frame reader thread
            self.stop_frame_reader = True
            if hasattr(self, 'frame_reader_thread') and self.frame_reader_thread and self.frame_reader_thread.is_alive():
                self.frame_reader_thread.join(timeout=1.0)
            
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Print final report
            print("\n=== Final Statistics Report ===")
            # Check hay zone motion for lost tracks (if we have a recent frame)
            hay_zone_has_motion = False
            if hasattr(self, 'flow_analyzer') and self.roi_manager.zones['hay'] is not None:
                hay_roi = self.roi_manager.zones['hay']
                x_coords = hay_roi[:, 0]
                y_coords = hay_roi[:, 1]
                hay_bbox = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
                hay_flow = self.flow_analyzer.get_flow_in_region(hay_bbox)
                hay_zone_has_motion = hay_flow > 0.15
            report = self.stats_aggregator.get_final_report(current_time, hay_zone_has_motion)
            for track_id, behaviors in report.items():
                print(f"\nTrack ID {track_id}:")
                for behavior, seconds in behaviors.items():
                    print(f"  {behavior}: {seconds:.2f} seconds ({seconds/60:.2f} minutes)")


def main():
    """Main entry point"""
    import sys
    
    # Get RTSP URL from command line if provided
    rtsp_url = None
    if len(sys.argv) > 1:
        rtsp_url = sys.argv[1]
    
    # Create and run monitor
    monitor = GuineaPigMonitor(model_path='model/yolov11.pt', rtsp_url=rtsp_url)
    monitor.run()


if __name__ == '__main__':
    main()

