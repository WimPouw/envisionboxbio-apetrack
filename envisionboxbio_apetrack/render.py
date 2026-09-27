"""Drawing tracks on frames: boxes with track IDs, and keypoints as points (red = low to green = high likelihood)
or as an OpenPose-style skeleton (one colour per limb)."""
import colorsys

import cv2
import numpy as np

from .pose import KEYPOINTS

_K = {k: i for i, k in enumerate(KEYPOINTS)}
SKELETON = [("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "head"), ("right_eye", "head"), ("nose", "neck"),
            ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("neck", "hip"), ("hip", "right_knee"), ("right_knee", "right_ankle"),
            ("hip", "left_knee"), ("left_knee", "left_ankle")]
EDGES = [(_K[a], _K[b]) for a, b in SKELETON]
LIMB_COLOURS = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / len(EDGES), 1, 1))[::-1] for i in range(len(EDGES))]
TRACK_COLOURS = [(255, 0, 255), (0, 200, 255), (255, 200, 0), (0, 255, 0), (255, 255, 255), (0, 0, 255), (255, 128, 0)]


def draw_frame(frame, rows, kp, style, scale, min_likelihood=0.0, frame_index=None):
    """rows: this frame's track rows (with box and raw_box columns); kp: (n, 16, 3) in input pixels."""
    view = np.ascontiguousarray(cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame.copy())
    t = max(1, int(round(2 * view.shape[1] / 960)))                 # line width relative to a 960-wide frame
    for j, r in enumerate(rows):
        col = TRACK_COLOURS[int(r["track_id"]) % len(TRACK_COLOURS)]
        if "raw_box_x" in r and (r["raw_box_x"], r["raw_box_w"]) != (r["box_x"], r["box_w"]):
            rx, ry, rw, rh = (np.array([r["raw_box_x"], r["raw_box_y"], r["raw_box_w"], r["raw_box_h"]]) * scale).astype(int)
            cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (170, 170, 170), 1)
        x, y, w, h = (np.array([r["box_x"], r["box_y"], r["box_w"], r["box_h"]]) * scale).astype(int)
        cv2.rectangle(view, (x, y), (x + w, y + h), col, 1 if r["interpolated"] else t)
        label = f"id {int(r['track_id'])}" + (" interp." if r["interpolated"] else "")
        cv2.putText(view, label, (x, max(15, y - 5)), 0, 0.3 * t, col, t)
        p = kp[j]
        if style == "skeleton":
            for (a, b), c in zip(EDGES, LIMB_COLOURS):
                if min(p[a, 2], p[b, 2]) >= min_likelihood:
                    cv2.line(view, tuple((p[a, :2] * scale).astype(int)), tuple((p[b, :2] * scale).astype(int)), c, t + 1, cv2.LINE_AA)
            for x_, y_, c in p:
                if c >= min_likelihood:
                    cv2.circle(view, (int(x_ * scale), int(y_ * scale)), t + 1, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            for x_, y_, c in p:
                if c >= min_likelihood:
                    c = min(max(c, 0), 1)
                    cv2.circle(view, (int(x_ * scale), int(y_ * scale)), 2 * t, (0, int(255 * c), int(255 * (1 - c))), -1, cv2.LINE_AA)
    if frame_index is not None:
        cv2.putText(view, f"frame {frame_index}", (10, 12 * t), 0, 0.3 * t, (255, 255, 255), t)
    return view
