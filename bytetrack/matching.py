"""
Matching algorithms for ByteTrack
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    
    Args:
        bb_test: ndarray, shape (N, 4)
        bb_gt: ndarray, shape (M, 4)
    
    Returns:
        iou: ndarray, shape (N, M)
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.5):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    
    Args:
        detections: ndarray, shape (N, 5) - [x1, y1, x2, y2, score]
        trackers: ndarray, shape (M, 4) - [x1, y1, x2, y2]
        iou_threshold: float, minimum IOU for match
    
    Returns:
        matches: ndarray, shape (K, 2) - matched pairs of indices
        unmatched_dets: ndarray, unmatched detection indices
        unmatched_trks: ndarray, unmatched tracker indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.asarray(matched_indices).T
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_dets = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_dets.append(d)
    unmatched_dets = np.array(unmatched_dets, dtype=int)

    unmatched_trks = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trks.append(t)
    unmatched_trks = np.array(unmatched_trks, dtype=int)

    # Filter out matches with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets = np.append(unmatched_dets, int(m[0]))
            unmatched_trks = np.append(unmatched_trks, int(m[1]))
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0).astype(int)
    
    # Ensure all arrays are integer type
    unmatched_dets = unmatched_dets.astype(int)
    unmatched_trks = unmatched_trks.astype(int)

    return matches, unmatched_dets, unmatched_trks

