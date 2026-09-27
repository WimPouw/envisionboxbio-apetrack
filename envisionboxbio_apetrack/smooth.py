"""Savitzky-Golay smoothing (order 2) per track, over runs of consecutive frames only (never across frames where
the ape was not tracked). Window = SMOOTHING[level] seconds, at least 5 frames (order 2 needs 5 to smooth)."""
import numpy as np
from scipy.signal import savgol_filter

from .config import SMOOTHING


def _window(level, fps):
    if SMOOTHING[level] == 0:
        return 0
    return max(int(round(SMOOTHING[level] * fps)) | 1, 5)


def _smooth_runs(df, cols, win):
    """Smooth columns `cols` of df (sorted by track, frame) within runs of consecutive frames."""
    sig = df[cols].to_numpy(float)
    run = ((df.frame.diff() != 1) | (df.track_id.diff() != 0)).cumsum().to_numpy()
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        w = min(win, len(idx) if len(idx) % 2 else len(idx) - 1)
        if w >= 5:
            sig[idx] = savgol_filter(sig[idx], w, 2, axis=0)
    return sig


def smooth_boxes(boxes, fps, level):
    """Smooths each box's centre, width and height. Keeps the unsmoothed box as raw_box_*."""
    out = boxes.sort_values(["track_id", "frame"]).reset_index(drop=True)
    for c in ["box_x", "box_y", "box_w", "box_h"]:
        out["raw_" + c] = out[c]
    win = _window(level, fps)
    if not win or not len(out):
        return out
    out["cx"], out["cy"] = out.box_x + out.box_w / 2, out.box_y + out.box_h / 2
    s = _smooth_runs(out, ["cx", "cy", "box_w", "box_h"], win)
    out["box_w"], out["box_h"] = np.maximum(s[:, 2], 1), np.maximum(s[:, 3], 1)
    out["box_x"], out["box_y"] = s[:, 0] - out.box_w / 2, s[:, 1] - out.box_h / 2
    return out.drop(columns=["cx", "cy"])


def smooth_keypoints(tracks, keypoints, fps, level):
    """Smooths every keypoint's x and y over time (likelihoods are left as they are)."""
    win = _window(level, fps)
    if not win or not len(tracks):
        return tracks
    out = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
    cols = [f"{k}_{a}" for k in keypoints for a in "xy"]
    out[cols] = _smooth_runs(out, cols, win)
    return out
