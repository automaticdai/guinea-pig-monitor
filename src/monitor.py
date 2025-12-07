"""
Guinea Pig Monitor - Main monitoring system
"""
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from ultralytics import YOLO
from bytetrack import ByteTracker

from .roi_manager import ROIManager
from .optical_flow import OpticalFlowAnalyzer
from .behavior_classifier import BehaviorClassifier
from .statistics import StatisticsAggregator


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
        self.show_timing_reports = False  # Default: timing reports off
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
            
            # Classify behavior (returns tuple: (activity_behavior, motion_behavior))
            behavior_start = time.time()
            activity_behavior, motion_behavior = self.behavior_classifier.classify(track_id, bbox)
            behavior_time = time.time() - behavior_start
            behavior_total_time += behavior_time
            
            # Update statistics using actual elapsed time
            stats_start = time.time()
            current_time = time.time() - self.start_time
            self.stats_aggregator.update(track_id, activity_behavior, motion_behavior, current_time)
            stats_time = time.time() - stats_start
            stats_total_time += stats_time
            
            # Draw bounding box
            bbox_start = time.time()
            color = self.get_color_for_track(track_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and behaviors
            # Format label: show activity if present, otherwise show motion
            if activity_behavior:
                label = f"ID {track_id}: {activity_behavior} ({motion_behavior})"
            else:
                label = f"ID {track_id}: {motion_behavior}"
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
        
        # Draw statistics HUD (if enabled) - FPS is now drawn inside the HUD
        drawing_start = time.time()
        if self.show_stats:
            self.draw_hud(annotated_frame)
        drawing_time = time.time() - drawing_start
        if self.timing_enabled:
            self.timing_stats['drawing'].append(drawing_time)
        
        total_time = time.time() - frame_start
        if self.timing_enabled:
            self.timing_stats['total'].append(total_time)
            self.timing_frame_count += 1
            
            # Print timing report periodically (only if enabled)
            if self.timing_frame_count % self.timing_report_interval == 0 and self.show_timing_reports:
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
    
    def _check_hay_zone_motion(self):
        """Helper method to check if hay zone has motion (for lost track eating detection)"""
        if self.roi_manager.zones['hay'] is None:
            return False
        
        # Get bounding box of hay zone
        hay_roi = self.roi_manager.zones['hay']
        x_coords = hay_roi[:, 0]
        y_coords = hay_roi[:, 1]
        hay_bbox = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
        
        # Check flow in hay zone using MOTION_THRESHOLD from behavior classifier
        hay_flow = self.flow_analyzer.get_flow_in_region(hay_bbox)
        motion_threshold = self.behavior_classifier.MOTION_THRESHOLD
        return hay_flow > motion_threshold
    
    def _draw_metric_bar(self, frame, metric_name, metric_data, bar_color, x_start, y_offset, 
                         text_width, bar_spacing_x, bar_width, bar_height, line_height, 
                         padding, font, font_scale, thickness):
        """Helper method to draw a single metric bar (reduces code duplication)"""
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
        all_track_ids.update(self.stats_aggregator.current_activity.keys())
        all_track_ids.update(self.stats_aggregator.current_motion.keys())
        # Always include initial track ID (even if never seen)
        all_track_ids.add(self.stats_aggregator.initial_track_id)
        
        if not all_track_ids:
            return
        
        # Check if hay zone has motion (for lost track eating detection)
        hay_zone_has_motion = self._check_hay_zone_motion()
        
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
            num_metrics = 5  # eat, sleep, drink, idle, activate
            gap_before_activate = line_height // 2  # Half line height gap
            track_header_height = line_height * 2  # Header + FPS line
            track_height = track_header_height + (num_metrics * (line_height + bar_spacing)) + gap_before_activate + padding
            
            # Layout constants
            text_width = 250  # Fixed width for text (increased for better spacing)
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
                    tid_height = track_header_height + (num_metrics * (line_height + bar_spacing)) + gap_before_activate + padding
                    y_start += tid_height + padding
            
            # Ensure HUD doesn't go off bottom of screen
            if y_start + track_height > frame_height - padding:
                # If it would overflow, try to fit it by reducing spacing or moving up
                max_y = frame_height - track_height - padding
                if max_y > padding:
                    y_start = max_y
                else:
                    # Can't fit, skip drawing this track
                    continue
            
            # Calculate width needed
            total_width = text_width + bar_spacing_x + bar_width + padding * 2
            x_start = frame_width - total_width - padding
            
            # Ensure HUD doesn't go off-screen
            if x_start + total_width > frame_width - padding:
                x_start = frame_width - total_width - padding
            if x_start < padding:
                x_start = padding  # Fallback: ensure at least some padding from left edge
            
            # Draw semi-transparent background for this track
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (x_start, y_start), 
                         (x_start + total_width, y_start + track_height),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw track header (larger font)
            header_text = f"Monitor {lost_indicator}"
            cv2.putText(frame, header_text, (x_start + padding, y_start + line_height),
                       font, font_scale + 0.2, (255, 255, 255), thickness + 1)
            
            # Draw FPS inside the HUD box (below header)
            fps_text = f"FPS: {self.fps:.1f}"
            fps_y = y_start + line_height + int(line_height * 0.8)
            cv2.putText(frame, fps_text, (x_start + padding, fps_y),
                       font, font_scale, (0, 255, 0), thickness)  # Green color for FPS
            
            # Draw each metric with bar (start after header + FPS)
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
            y_offset += gap_before_activate
            
            # Draw idle and activate metrics separately after the gap
            # First, draw idle
            self._draw_metric_bar(frame, 'idle', stats_data['idle'], (128, 128, 128), 
                                 x_start, y_offset, text_width, bar_spacing_x, 
                                 bar_width, bar_height, line_height, padding, 
                                 font, font_scale, thickness)
            y_offset += line_height + bar_spacing
            
            # Then, draw activate
            self._draw_metric_bar(frame, 'activate', stats_data['activate'], (255, 165, 0), 
                                 x_start, y_offset, text_width, bar_spacing_x, 
                                 bar_width, bar_height, line_height, padding, 
                                 font, font_scale, thickness)

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
        print("  - Press 'd': Toggle timing breakdown reports")
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
                    # Check hay zone motion for lost tracks
                    hay_zone_has_motion = self._check_hay_zone_motion()
                    self.stats_aggregator.save_stats(current_elapsed, hay_zone_has_motion)
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
                elif key == ord('c'):
                    self.stats_aggregator.clear_stats()
                    # Reset monitor start time to match statistics session start time
                    self.start_time = time.time()
                    print("Statistics cleared")
                elif key == ord('d'):
                    self.show_timing_reports = not self.show_timing_reports
                    print(f"Timing breakdown reports: {'ON' if self.show_timing_reports else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Save statistics before exit
            current_time = time.time() - self.start_time
            print("\nSaving statistics...")
            # Check hay zone motion for lost tracks (if we have a recent frame)
            hay_zone_has_motion = self._check_hay_zone_motion()
            self.stats_aggregator.save_stats(current_time, hay_zone_has_motion)
            
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
            hay_zone_has_motion = self._check_hay_zone_motion()
            report = self.stats_aggregator.get_final_report(current_time, hay_zone_has_motion)
            for track_id, behaviors in report.items():
                print(f"\nTrack ID {track_id}:")
                for behavior, seconds in behaviors.items():
                    print(f"  {behavior}: {seconds:.2f} seconds ({seconds/60:.2f} minutes)")

