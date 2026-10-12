"""Savitzky-Golay smoothing (order 2) per track, over runs of consecutive frames only (never across frames where
the ape was not tracked). Window = SMOOTHING[level] seconds, at least 5 frames (order 2 needs 5 to smooth)."""
import numpy as np
from scipy.signal import savgol_filter

from .config import SMOOTHING


CHAINS = [["eye"], ["shoulder", "elbow", "wrist"], ["knee", "ankle"]]


# Not used
def fix_lr_swaps(tracks, prior=0.005):
    """Per track, per run of consecutive frames, per limb chain: Viterbi over {as predicted, swapped}.
    Cost = likelihood-weighted keypoint motion (in ape sizes) + `prior` per swapped frame, so short
    left/right flickers are undone and the raw labels are kept otherwise."""
    out = tracks.sort_values(["track_id", "frame"]).reset_index(drop=True)
    run = ((out.frame.diff() != 1) | (out.track_id.diff() != 0)).cumsum().to_numpy()
    size = np.sqrt(out.box_w * out.box_h).to_numpy()
    for parts in CHAINS:
        L, R = [f"left_{p}" for p in parts], [f"right_{p}" for p in parts]
        Lxy = np.stack([out[[f"{n}_x", f"{n}_y"]].to_numpy() for n in L], 1)      # (n, joints, 2)
        Rxy = np.stack([out[[f"{n}_x", f"{n}_y"]].to_numpy() for n in R], 1)
        w = (out[[f"{n}_likelihood" for n in L]].to_numpy() + out[[f"{n}_likelihood" for n in R]].to_numpy()) / 2
        swap = np.zeros(len(out), bool)
        for r in np.unique(run):
            idx = np.flatnonzero(run == r)
            if len(idx) < 3:
                continue
            # positions per frame under state 0 (as predicted) and 1 (swapped): (T, state, side, joint, 2)
            pos = np.stack([np.stack([Lxy[idx], Rxy[idx]], 1), np.stack([Rxy[idx], Lxy[idx]], 1)], 1)
            wi, si, T = w[idx], size[idx], len(idx)
            acc, back = np.array([0.0, prior]), np.zeros((T, 2), int)
            for t in range(1, T):
                d = np.linalg.norm(pos[t][None] - pos[t - 1][:, None], axis=-1)   # (prev, cur, side, joint)
                cost = (d * np.sqrt(wi[t] * wi[t - 1])).sum((-1, -2)) / si[t]
                tot = acc[:, None] + cost
                back[t], acc = tot.argmin(0), tot.min(0) + [0.0, prior]
            s = np.empty(T, int)
            s[-1] = acc.argmin()
            for t in range(T - 1, 0, -1):
                s[t - 1] = back[t, s[t]]
            swap[idx] = s == 1
        lc = [f"{n}_{a}" for n in L for a in ("x", "y", "likelihood")]
        rc = [f"{n}_{a}" for n in R for a in ("x", "y", "likelihood")]
        tmp = out.loc[swap, lc].to_numpy()
        out.loc[swap, lc] = out.loc[swap, rc].to_numpy()
        out.loc[swap, rc] = tmp
    return out


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
    
 
