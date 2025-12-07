"""
Statistics Aggregator - Aggregates behavior statistics per track ID
"""
import json
import time
from collections import defaultdict
from pathlib import Path


class StatisticsAggregator:
    """Aggregates behavior statistics per track ID"""
    def __init__(self, stats_file='data/statistics.json', initial_track_id=1):
        self.stats_file = stats_file
        self.stats = defaultdict(lambda: defaultdict(float))  # track_id -> behavior -> accumulated seconds
        self.current_activity = defaultdict(str)  # track_id -> current activity behavior (eat/sleep/drink)
        self.current_motion = defaultdict(str)  # track_id -> current motion behavior (idle/activate)
        self.activity_start_time = defaultdict(float)  # track_id -> start time for current activity
        self.motion_start_time = defaultdict(float)  # track_id -> start time for current motion
        self.track_start_time = defaultdict(float)  # track_id -> when tracking started for this track
        self.last_seen_time = defaultdict(float)  # track_id -> last time track was seen
        self.session_start_time = time.time()  # When this session started
        self.initial_track_id = initial_track_id
        self.load_stats()
        
        # Initialize track as lost - start counting immediately
        if initial_track_id not in self.track_start_time or self.track_start_time[initial_track_id] == 0:
            self.track_start_time[initial_track_id] = 0.0  # Start from session start
            self.last_seen_time[initial_track_id] = 0.0  # Never seen yet
            self.current_activity[initial_track_id] = ''  # Start with no activity
            self.current_motion[initial_track_id] = 'idle'  # Start as idle
            self.activity_start_time[initial_track_id] = 0.0  # Start counting from beginning
            self.motion_start_time[initial_track_id] = 0.0  # Start counting from beginning

    def update(self, track_id, activity_behavior, motion_behavior, current_time):
        """Update statistics for a track using actual elapsed time
        
        Args:
            track_id: Track ID
            activity_behavior: Activity behavior ('eat_hay', 'eat_pellet', 'drinking', 'sleeping', or '')
            motion_behavior: Motion behavior ('idle' or 'activate')
            current_time: Current elapsed time
        """
        # Check if this is the first time we see this track (transition from lost to found)
        was_lost = (track_id in self.last_seen_time and self.last_seen_time[track_id] == 0.0)
        
        # Track when this track ID first appeared in this session
        if track_id not in self.track_start_time or self.track_start_time[track_id] == 0:
            self.track_start_time[track_id] = 0.0  # Start from session beginning
            was_lost = True
        
        # If track was lost and now found, finalize the lost time
        # Note: We don't count idle/activate for lost tracks - only sleeping/eating
        if was_lost and self.last_seen_time.get(track_id, 0.0) == 0.0:
            # Count time from start until now
            lost_duration = current_time
            if lost_duration > 20.0:
                # After 20 seconds, count as sleeping (idle/activate not counted for lost tracks)
                self.stats[track_id]['sleeping'] = self.stats[track_id].get('sleeping', 0) + (lost_duration - 20.0)
                # First 20 seconds are not counted (no idle/activate for lost tracks)
            # If less than 20 seconds, don't count anything (no idle/activate for lost tracks)
        
        # Update last seen time
        self.last_seen_time[track_id] = current_time
        
        # Update activity behavior if changed
        if self.current_activity[track_id] != activity_behavior:
            if self.current_activity[track_id] and self.activity_start_time[track_id] >= 0:
                # Calculate actual elapsed time
                prev_duration = current_time - self.activity_start_time[track_id]
                if prev_duration > 0:
                    self.stats[track_id][self.current_activity[track_id]] += prev_duration
            
            # Start new activity behavior
            self.current_activity[track_id] = activity_behavior
            self.activity_start_time[track_id] = current_time
        
        # Update motion behavior if changed
        if self.current_motion[track_id] != motion_behavior:
            if self.current_motion[track_id] and self.motion_start_time[track_id] >= 0:
                # Calculate actual elapsed time
                prev_duration = current_time - self.motion_start_time[track_id]
                if prev_duration > 0:
                    self.stats[track_id][self.current_motion[track_id]] += prev_duration
            
            # Start new motion behavior
            self.current_motion[track_id] = motion_behavior
            self.motion_start_time[track_id] = current_time
    
    def get_stats_with_lost_track_sleep(self, track_id, current_time, hay_zone_has_motion=False):
        """Get stats including sleep/eat time for lost tracks (>20 seconds)
        
        Args:
            track_id: Track ID
            current_time: Current elapsed time
            hay_zone_has_motion: True if hay zone has motion (indicates eating)
        """
        stats = self.stats[track_id].copy()
        
        # Check if track has never been seen (initial lost state)
        # Note: We don't count idle/activate for lost tracks - only sleeping/eating
        if track_id in self.last_seen_time and self.last_seen_time[track_id] == 0.0:
            # Track is still lost from the beginning
            if current_time > 20.0:
                # After 20 seconds, count as sleeping (idle/activate not counted for lost tracks)
                stats['sleeping'] = stats.get('sleeping', 0) + (current_time - 20.0)
                # First 20 seconds are not counted (no idle/activate for lost tracks)
            # If less than 20 seconds, don't count anything (no idle/activate for lost tracks)
            return stats
        
        # Add current activity behavior duration if track is active
        if track_id in self.current_activity and self.current_activity[track_id]:
            if self.activity_start_time[track_id] >= 0:
                current_activity = self.current_activity[track_id]
                current_duration = current_time - self.activity_start_time[track_id]
                if current_duration > 0:
                    stats[current_activity] = stats.get(current_activity, 0) + current_duration
        
        # Add current motion behavior duration if track is active
        if track_id in self.current_motion and self.current_motion[track_id]:
            if self.motion_start_time[track_id] >= 0:
                current_motion = self.current_motion[track_id]
                current_duration = current_time - self.motion_start_time[track_id]
                if current_duration > 0:
                    stats[current_motion] = stats.get(current_motion, 0) + current_duration
        
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
        if track_id not in self.stats and track_id not in self.current_activity and track_id not in self.current_motion:
            if track_id not in self.track_start_time and track_id != self.initial_track_id:
                return None
        
        # Get stats including lost track sleep/eat time
        stats = self.get_stats_with_lost_track_sleep(track_id, current_time, hay_zone_has_motion)
        
        # Calculate totals
        eat_total = stats.get('eat_hay', 0) + stats.get('eat_pellet', 0)
        sleep_total = stats.get('sleeping', 0)
        drink_total = stats.get('drinking', 0)
        idle_total = stats.get('idle', 0)
        activate_total = stats.get('activate', 0)
        
        # Calculate total tracked time
        # Total time is the sum of all behaviors (they are mutually exclusive)
        total_time = eat_total + sleep_total + drink_total + idle_total + activate_total
        
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
        # Group 2: idle + activate = 100%
        group1_total = sleep_total + eat_total + drink_total
        group2_total = idle_total + activate_total
        
        if group1_total > 0:
            eat_pct = (eat_total / group1_total) * 100
            sleep_pct = (sleep_total / group1_total) * 100
            drink_pct = (drink_total / group1_total) * 100
        else:
            eat_pct = sleep_pct = drink_pct = 0.0
        
        if group2_total > 0:
            idle_pct = (idle_total / group2_total) * 100
            activate_pct = (activate_total / group2_total) * 100
        else:
            idle_pct = activate_pct = 0.0
        
        # Format as minutes
        eat_min = eat_total / 60.0
        sleep_min = sleep_total / 60.0
        drink_min = drink_total / 60.0
        idle_min = idle_total / 60.0
        activate_min = activate_total / 60.0
        
        return {
            'eat': {'min': eat_min, 'pct': eat_pct},
            'sleep': {'min': sleep_min, 'pct': sleep_pct},
            'drink': {'min': drink_min, 'pct': drink_pct},
            'idle': {'min': idle_min, 'pct': idle_pct},
            'activate': {'min': activate_min, 'pct': activate_pct},
            'total_time': total_time
        }

    def get_final_report(self, current_time, hay_zone_has_motion=False):
        """Get final statistics report including lost track sleep/eat time"""
        report = {}
        all_track_ids = set(self.stats.keys()) | set(self.current_activity.keys()) | set(self.current_motion.keys())
        
        for track_id in all_track_ids:
            # Use the method that includes lost track sleep/eat time
            stats = self.get_stats_with_lost_track_sleep(track_id, current_time, hay_zone_has_motion)
            report[track_id] = stats
        
        return report
    
    def save_stats(self, current_time, hay_zone_has_motion=False):
        """Save statistics to JSON file"""
        try:
            # Ensure directory exists
            Path(self.stats_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Use get_stats_with_lost_track_sleep to include lost track sleeping time
            # This ensures sleeping time from lost tracks is included in saved stats
            all_track_ids = set(self.stats.keys()) | set(self.current_activity.keys()) | set(self.current_motion.keys())
            
            # Get final stats including lost track sleep/eat time for all tracks
            final_stats = {}
            for track_id in all_track_ids:
                final_stats[track_id] = self.get_stats_with_lost_track_sleep(track_id, current_time, hay_zone_has_motion)
            
            # Get final stats for all tracks
            data = {
                'session_start': self.session_start_time,
                'last_save_time': time.time(),
                'save_elapsed_time': current_time,
                'tracks': {}
            }
            
            for track_id in all_track_ids:
                # Save accumulated stats including lost track sleep/eat time
                track_data = {
                    'stats': dict(final_stats[track_id]),
                    'track_start_time': 0.0,  # Always reset to 0 for next session
                    'last_seen_time': 0.0,  # Reset for next session
                    'current_activity': '',  # Session-specific, don't save
                    'current_motion': '',  # Session-specific, don't save
                    'activity_start_time': 0.0,  # Session-specific, don't save
                    'motion_start_time': 0.0  # Session-specific, don't save
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
                    # Note: current_activity, current_motion, activity_start_time, and motion_start_time
                    # are session-specific, so we don't restore them - they'll start fresh
                    # Note: total_accumulated_time is not needed - total is sum of behaviors
            
            print(f"Loaded statistics from {self.stats_file}")
        except Exception as e:
            print(f"Error loading statistics: {e}")
    
    def clear_stats(self):
        """Clear all statistics"""
        self.stats.clear()
        self.current_activity.clear()
        self.current_motion.clear()
        self.activity_start_time.clear()
        self.motion_start_time.clear()
        self.track_start_time.clear()
        self.last_seen_time.clear()
        self.session_start_time = time.time()
        
        # Re-initialize initial track ID as lost (like in __init__)
        self.track_start_time[self.initial_track_id] = 0.0  # Start from session start
        self.last_seen_time[self.initial_track_id] = 0.0  # Never seen yet
        self.current_activity[self.initial_track_id] = ''  # Start with no activity
        self.current_motion[self.initial_track_id] = 'idle'  # Start as idle
        self.activity_start_time[self.initial_track_id] = 0.0  # Start counting from beginning
        self.motion_start_time[self.initial_track_id] = 0.0  # Start counting from beginning
        
        # Also delete the stats file
        try:
            if Path(self.stats_file).exists():
                Path(self.stats_file).unlink()
                print("Statistics cleared and file deleted")
            else:
                print("Statistics cleared")
        except Exception as e:
            print(f"Error deleting statistics file: {e}")

