"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Pure Python implementation
"""
import numpy as np
from .kalman_filter import KalmanFilter
from .matching import associate_detections_to_trackers, iou_batch


class STrack:
    """
    Single object track representation
    """
    _count = 0

    def __init__(self, tlwh, score, frame_id=0):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.track_id = 0
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0

        self.tlwh = tlwh
        self.tlbr = self.tlwh_to_tlbr(tlwh)

    def activate(self, frame_id, track_id):
        """Start a new tracklet"""
        self.track_id = track_id
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.is_activated = True
        self.tracklet_len = 0
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self.tlwh))

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a previously lost track"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tlwh = new_track.tlwh
        self.tlbr = self.tlwh_to_tlbr(self.tlwh)
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.score = new_track.score
        if new_id:
            self.track_id = self.next_id()
        self.is_activated = True

    def update(self, new_track, frame_id):
        """Update a matched track"""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.tlwh = new_tlwh
        self.tlbr = self.tlwh_to_tlbr(self.tlwh)
        self.score = new_track.score
        self.is_activated = True

    def predict(self):
        """Propagate the state distribution to the current time step"""
        if self.mean is None:
            self.mean = self.tlwh_to_xyah(self.tlwh)
        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
            # Extract only position part (first 4 dimensions: x, y, a, h)
            # The mean has 8 dimensions: [x, y, a, h, vx, vy, va, vh]
            mean_pos = self.mean[:4]
            self.tlwh = self.xyah_to_tlwh(mean_pos)

    def mark_lost(self):
        """Mark this track as lost"""
        self.is_activated = False

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert [t, l, w, h] to [x, y, a, h] where (x,y) is center, a is aspect ratio"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def xyah_to_tlwh(xyah):
        """Convert [x, y, a, h] to [t, l, w, h]"""
        ret = np.asarray(xyah).copy()
        ret[2] *= ret[3]
        ret[2:] = np.maximum(ret[2:], 1.0)  # Ensure width and height are at least 1
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert [t, l, w, h] to [t, l, b, r]"""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert [x1, y1, x2, y2] to [t, l, w, h]"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format [x1, y1, x2, y2]"""
        return self.tlbr.copy()

    def to_tlwh(self):
        """Get current position in [t, l, w, h] format"""
        return self.tlwh.copy()

    @staticmethod
    def next_id():
        """Get next track ID"""
        STrack._count += 1
        return STrack._count

    @staticmethod
    def multi_predict(stracks):
        """Predict multiple tracks"""
        if len(stracks) > 0:
            for track in stracks:
                track.predict()


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    """
    def __init__(self, frame_rate=30, track_thresh=0.5, high_thresh=0.6, match_thresh=0.8, track_buffer=30):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.frame_id = 0
        self.track_buffer = track_buffer
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.tracked_tracks = []  # type: list[STrack]
        self.lost_tracks = []  # type: list[STrack]
        self.removed_tracks = []  # type: list[STrack]

    def update(self, detections, frame_id):
        """
        Update tracker with new detections
        
        Args:
            detections: list of detections, each is [x1, y1, x2, y2, score]
            frame_id: current frame ID
        
        Returns:
            output_stracks: list of active tracks
        """
        self.frame_id = frame_id
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Convert detections to STrack format
        detections_strack = []
        for det in detections:
            tlwh = STrack.tlbr_to_tlwh(det[:4])
            det_strack = STrack(tlwh, det[4], frame_id)
            detections_strack.append(det_strack)
        
        # Separate high and low score detections
        detections_high = [d for d in detections_strack if d.score >= self.high_thresh]
        detections_low = [d for d in detections_strack if d.score < self.high_thresh and d.score >= self.track_thresh]

        # Separate confirmed and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Predict current location with Kalman filter
        strack_pool = joint_stracks(tracked_stracks, self.lost_tracks)
        STrack.multi_predict(strack_pool)

        # Associate high score detections with tracks
        if len(strack_pool) > 0 and len(detections_high) > 0:
            trackers = np.array([st.to_tlbr() for st in strack_pool])
            dets = np.array([d.to_tlbr() for d in detections_high])
            matches, u_track, u_detection_high = associate_detections_to_trackers(
                dets, trackers, iou_threshold=self.match_thresh)
            
            # Convert to integers for indexing and validate
            if len(matches) > 0:
                matches = np.array(matches, dtype=int)
                # Ensure matches is 2D
                if matches.ndim == 1 and len(matches) == 2:
                    matches = matches.reshape(1, -1)
                
                for itracked, idet in matches:
                    # Bounds checking
                    if itracked >= len(strack_pool) or idet >= len(detections_high) or itracked < 0 or idet < 0:
                        continue
                    track = strack_pool[itracked]
                    det = detections_high[idet]
                    if track.is_activated:
                        track.update(det, self.frame_id)
                        activated_stracks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
            
            u_track = np.array(u_track, dtype=int) if len(u_track) > 0 else np.array([], dtype=int)
            u_detection_high = np.array(u_detection_high, dtype=int) if len(u_detection_high) > 0 else np.array([], dtype=int)
        else:
            u_track = np.array(list(range(len(strack_pool))), dtype=int)
            u_detection_high = np.array(list(range(len(detections_high))), dtype=int)

        # Associate low score detections with lost tracks
        r_tracked_stracks = [strack_pool[int(i)] for i in u_track if strack_pool[int(i)].is_activated]
        if len(r_tracked_stracks) > 0 and len(detections_low) > 0:
            trackers = np.array([st.to_tlbr() for st in r_tracked_stracks])
            dets = np.array([d.to_tlbr() for d in detections_low])
            matches, u_track_low, u_detection_low = associate_detections_to_trackers(
                dets, trackers, iou_threshold=0.5)
            
            # Convert to integers for indexing and validate
            if len(matches) > 0:
                matches = np.array(matches, dtype=int)
                # Ensure matches is 2D
                if matches.ndim == 1 and len(matches) == 2:
                    matches = matches.reshape(1, -1)
                
                for itracked, idet in matches:
                    # Bounds checking
                    if itracked >= len(r_tracked_stracks) or idet >= len(detections_low) or itracked < 0 or idet < 0:
                        continue
                    track = r_tracked_stracks[itracked]
                    det = detections_low[idet]
                    if track.is_activated:
                        track.update(det, self.frame_id)
                        activated_stracks.append(track)
                    else:
                        track.re_activate(det, self.frame_id, new_id=False)
                        refind_stracks.append(track)
            
            u_track_low = np.array(u_track_low, dtype=int) if len(u_track_low) > 0 else np.array([], dtype=int)
            u_detection_low = np.array(u_detection_low, dtype=int) if len(u_detection_low) > 0 else np.array([], dtype=int)
        else:
            u_track_low = np.array(list(range(len(r_tracked_stracks))), dtype=int)
            u_detection_low = np.array(list(range(len(detections_low))), dtype=int)

        # Mark unmatched tracks as lost
        for it in u_track_low:
            track = r_tracked_stracks[int(it)]
            if track.is_activated:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks
        # Safely get remaining detections with bounds checking
        detections_remaining = []
        for i in u_detection_high:
            idx = int(i)
            if 0 <= idx < len(detections_high):
                detections_remaining.append(detections_high[idx])
        detections_remaining.extend(detections_low)
        if len(unconfirmed) > 0 and len(detections_remaining) > 0:
            trackers = np.array([st.to_tlbr() for st in unconfirmed])
            dets = np.array([d.to_tlbr() for d in detections_remaining])
            matches, u_unconfirmed, u_detection = associate_detections_to_trackers(
                dets, trackers, iou_threshold=0.7)
            
            # Convert to integers for indexing and validate
            if len(matches) > 0:
                matches = np.array(matches, dtype=int)
                # Ensure matches is 2D
                if matches.ndim == 1 and len(matches) == 2:
                    matches = matches.reshape(1, -1)
                
                for itracked, idet in matches:
                    # Bounds checking
                    if itracked >= len(unconfirmed) or idet >= len(detections_remaining) or itracked < 0 or idet < 0:
                        continue
                    unconfirmed[itracked].update(detections_remaining[idet], self.frame_id)
                    activated_stracks.append(unconfirmed[itracked])
            
            u_unconfirmed = np.array(u_unconfirmed, dtype=int) if len(u_unconfirmed) > 0 else np.array([], dtype=int)
            u_detection = np.array(u_detection, dtype=int) if len(u_detection) > 0 else np.array([], dtype=int)
        else:
            u_unconfirmed = np.array(list(range(len(unconfirmed))), dtype=int)
            u_detection = np.array(list(range(len(detections_remaining))), dtype=int)

        # Remove unconfirmed tracks that weren't matched
        for it in u_unconfirmed:
            track = unconfirmed[int(it)]
            track.mark_lost()
            removed_stracks.append(track)

        # Init new tracks from unmatched detections
        for inew in u_detection:
            track = detections_remaining[int(inew)]
            if track.score >= self.track_thresh:
                track.activate(self.frame_id, track_id=STrack.next_id())
                activated_stracks.append(track)

        # Update tracked and lost tracks
        self.tracked_tracks = [t for t in self.tracked_tracks if t.is_activated]
        self.tracked_tracks = joint_stracks(self.tracked_tracks, activated_stracks)
        self.tracked_tracks = joint_stracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_stracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_stracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_stracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_stracks(self.tracked_tracks, self.lost_tracks)

        # Remove tracks that have been lost for too long
        output_stracks = [track for track in self.tracked_tracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista, tlistb):
    """Join two track lists"""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Subtract track list b from track list a"""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks"""
    if len(stracksa) == 0 or len(stracksb) == 0:
        return stracksa, stracksb
    
    trackers_a = np.array([st.to_tlbr() for st in stracksa])
    trackers_b = np.array([st.to_tlbr() for st in stracksb])
    pdist = iou_batch(trackers_a, trackers_b)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
