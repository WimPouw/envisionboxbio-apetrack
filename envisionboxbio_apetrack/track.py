"""From raw detections to tracks: threshold -> merge overlapping boxes (NMS) -> ByteTrack -> stitch tracks
broken by occlusion -> fill short gaps. One row per ape per frame; x/y = top-left corner of the box."""
import numpy as np
import pandas as pd
import supervision as sv

BOX_COLS = ["box_x", "box_y", "box_w", "box_h"]
TRACK_COLS = ["frame", "time_s", "track_id", "det_conf", "interpolated"] + BOX_COLS


def bytetrack(dets, fps, threshold, nms_iou, activation, buffer_s, matching):
    every = int(dets.every.iloc[0]) if "every" in dets and len(dets) else 1
    n_frames = int(dets.n_frames.iloc[0]) if len(dets) else 0
    rate = fps / every
    tr = sv.ByteTrack(track_activation_threshold=activation, lost_track_buffer=max(1, int(round(buffer_s * rate))),
                      minimum_matching_threshold=matching, frame_rate=max(1, int(round(rate))))
    tr.reset()                                                       # track IDs start at 1 in every video
    by_frame = {f: g for f, g in dets[dets.conf >= threshold].groupby("frame")}
    rows = []
    for i in range(0, n_frames, every):
        g = by_frame.get(i)
        d = sv.Detections(xyxy=g[["x0", "y0", "x1", "y1"]].to_numpy(float), confidence=g.conf.to_numpy(float),
                          class_id=np.zeros(len(g), int)) if g is not None else sv.Detections.empty()
        if len(d):
            d = d.with_nms(threshold=nms_iou, class_agnostic=True)
        d = tr.update_with_detections(d)
        for (x0, y0, x1, y1), tid, c in zip(d.xyxy, d.tracker_id, d.confidence):
            rows.append({"frame": i, "time_s": i / fps, "track_id": int(tid), "det_conf": float(c), "interpolated": False,
                         "box_x": x0, "box_y": y0, "box_w": x1 - x0, "box_h": y1 - y0})
    return pd.DataFrame(rows, columns=TRACK_COLS)


def stitch_tracks(boxes, fps, max_gap_s, max_dist=1.0, max_size_ratio=2.0):
    """A track that starts 1 frame..max_gap_s after another ended, within max_dist x ape size (sqrt box area) of
    where that one ended and at a similar size, gets the older track's ID (an ape that was hidden and
    reappeared). Joins are one-to-one, closest first, so two apes cannot take over each other's ID."""
    if max_gap_s <= 0 or not len(boxes):
        return boxes
    b = boxes.sort_values("frame")
    ends = b.groupby("track_id").tail(1).set_index("track_id")
    starts = b.groupby("track_id").head(1).set_index("track_id")
    cand = []
    for old, e in ends.iterrows():
        es = np.sqrt(e.box_w * e.box_h)
        for new, st in starts.iterrows():
            gap = st.frame - e.frame
            if new == old or gap < 1 or gap > max_gap_s * fps:
                continue
            ss = np.sqrt(st.box_w * st.box_h)
            dist = np.hypot(st.box_x + st.box_w / 2 - e.box_x - e.box_w / 2, st.box_y + st.box_h / 2 - e.box_y - e.box_h / 2)
            if dist < max_dist * es and max(ss, es) / min(ss, es) < max_size_ratio:
                cand.append((dist / es, old, new))
    joined, used_old, used_new = {}, set(), set()
    for _, old, new in sorted(cand):
        if old not in used_old and new not in used_new:
            joined[new] = old
            used_old.add(old)
            used_new.add(new)

    def root(t):
        while t in joined:
            t = joined[t]
        return t
    return boxes.assign(track_id=boxes.track_id.map(root))


def fill_gaps(boxes, fps, max_gap_s):
    """Linearly interpolate a track's box over gaps up to max_gap_s (rows get interpolated=True)."""
    max_gap = int(round(max_gap_s * fps))
    if max_gap < 1 or not len(boxes):
        return boxes
    new = []
    for tid, g in boxes.sort_values("frame").groupby("track_id"):
        f = g.frame.to_numpy()
        for a in np.flatnonzero(np.diff(f) > 1):
            if f[a + 1] - f[a] - 1 > max_gap:
                continue
            ra, rb = g.iloc[a], g.iloc[a + 1]
            for fr in range(f[a] + 1, f[a + 1]):
                t = (fr - f[a]) / (f[a + 1] - f[a])
                row = {c: ra[c] + t * (rb[c] - ra[c]) for c in BOX_COLS}
                row.update(frame=fr, time_s=fr / fps, track_id=tid, det_conf=np.nan, interpolated=True)
                new.append(row)
    out = pd.concat([boxes, pd.DataFrame(new, columns=TRACK_COLS)], ignore_index=True) if new else boxes
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)
