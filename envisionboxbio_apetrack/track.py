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

def _iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)

def fix_takeovers(boxes, min_iou=0.5, margin=0.2):
    """Undo ID takeovers: when an ape disappears (behind another) and a track's box in the next frame continues
    that ape's box much better than its own previous box (IoU >= min_iou and >= own IoU + margin), the box
    belongs to the ape that disappeared. The two IDs are swapped from that frame on."""
    if not len(boxes):
        return boxes
    out = boxes.sort_values(["frame", "track_id"]).reset_index(drop=True)
    orig, xywh = out.track_id.to_numpy(), out[BOX_COLS].to_numpy(float)
    label = {t: t for t in np.unique(orig)}                         # original ID -> ID now
    new_ids, prev = orig.copy(), {}
    for f in sorted(out.frame.unique()):
        idx = np.flatnonzero(out.frame.to_numpy() == f)
        for x in [label[t] for t in orig[idx]]:
            cur = {label[t]: xywh[i] for t, i in zip(orig[idx], idx)}
            gone = [y for y in prev if y not in cur]
            if not gone or x not in cur:
                continue
            own = _iou(cur[x], prev[x]) if x in prev else 0.0
            y = max(gone, key=lambda g: _iou(cur[x], prev[g]))
            if _iou(cur[x], prev[y]) >= max(min_iou, own + margin):
                for t, l in label.items():                         # swap x <-> y from this frame on
                    label[t] = y if l == x else x if l == y else l
        new_ids[idx] = [label[t] for t in orig[idx]]
        prev = {new_ids[i]: xywh[i] for i in idx}
    return out.assign(track_id=new_ids)