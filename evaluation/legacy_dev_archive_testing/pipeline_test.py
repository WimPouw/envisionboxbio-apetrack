"""
Pipeline test: MegaDetector v6 (boxes) -> ByteTrack (IDs within a clip) -> OpenApePose (16 keypoints per box).

    python pipeline_test.py --detector owlv2              --clips all --out results/owlv2
    python pipeline_test.py --detector MDV6-mit-yolov9-c  --clips 14 15 18 grooming2 --out results/mit_c

Reads the clips from ../samples/ (read only); --clips takes Gibbons clip numbers or parts of a file name.
Writes, per clip, to --out:
    <name>.csv          long format: frame, time_s, track_id, det_class, det_conf, box_x, box_y, box_w, box_h,
                        and <keypoint>_x / _y / _likelihood for the 16 OpenApePose keypoints
    <name>_labeled.mp4  boxes (colour per track ID) and keypoints (red = low to green = high confidence)
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "..", "samples")
MODELS = os.path.abspath(os.path.join(HERE, "..", "models"))
WEIGHTS = os.path.join(MODELS, "hrnet_w48_oap_256x192_full.pth")
sys.path.insert(0, HERE)
import OpenApePose as oap                                  # copy of Lab 5's OpenApePose code (model, crop, peaks)

KEEP_CLASSES = (0, 1)                                      # MegaDetector: 0 animal, 1 person (apes are often "person")
COLOURS = [(255, 0, 255), (0, 200, 255), (255, 200, 0), (0, 255, 0), (255, 255, 255), (0, 0, 255)]


# ---------------------------------------------------------------------------------------------------
# MegaDetector v6 (MIT), loaded ONCE. PytorchWildlife's single_image_detection reloads the model on every
# call, which is far too slow for video, so we call its transform / network / post-processing directly.
# ---------------------------------------------------------------------------------------------------
class Detector:
    def __init__(self, version, device):
        from PytorchWildlife.models.detection import MegaDetectorV6MIT
        from PIL import Image
        self.Image = Image
        torch.hub.set_dir(MODELS)                                                      # download next to our other weights
        self.md = MegaDetectorV6MIT(device=device, pretrained=True, version=version)   # version must be explicit
        self.md._load_model(weights=self.md.weights, device=device, url=self.md.url)  # load once, not per frame
        self.device = device

    def __call__(self, frame_bgr, conf):
        self.md.cfg.task.nms.min_confidence = conf
        im = self.Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image, _, rev = self.md.transform(im)
        with torch.no_grad():
            pred = self.md.model(image.to(self.device)[None])
            res = self.md.post_proccess(pred, rev.to(self.device)[None])[0]       # [cls, x1, y1, x2, y2, conf]
        res = res.cpu().numpy() if len(res) else np.zeros((0, 6))
        return sv.Detections(xyxy=res[:, 1:5].astype(float), confidence=res[:, 5].astype(float),
                             class_id=res[:, 0].astype(int))


# ---------------------------------------------------------------------------------------------------
# OWLv2 (Google, Apache-2.0): open-vocabulary detector, prompted with text instead of trained on camera traps.
# Finds close-up and partly visible apes that MegaDetector misses. Weights cached in models/hf.
# ---------------------------------------------------------------------------------------------------
class OWLv2:
    PROMPTS = [["a photo of a chimpanzee", "a photo of a bonobo", "a photo of a gorilla",
                "a photo of an orangutan", "a photo of a gibbon", "a photo of a siamang",
                "a photo of an ape"]]

    def __init__(self, device, name="google/owlv2-base-patch16-ensemble", tiles=1, overlap=0.25, prompts=None):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        from PIL import Image
        self.Image = Image
        self.proc = Owlv2Processor.from_pretrained(name, cache_dir=os.path.join(MODELS, "hf"))
        self.model = Owlv2ForObjectDetection.from_pretrained(name, cache_dir=os.path.join(MODELS, "hf")).to(device).eval()
        self.device = device
        self.fp16 = device == "cuda"
        self.tiles, self.overlap = tiles, overlap
        if prompts is not None:
            self.PROMPTS = [list(prompts)]

    def __call__(self, frame_bgr, conf, nms=0.5):
        """tiles > 1: also detect on tiles x tiles overlapping crops, each enlarged to the model's input size, so
        small (far-away) apes become larger to the model. About (tiles^2 + 1) x slower."""
        if self.tiles <= 1:
            return self._detect(frame_bgr, conf, nms)
        H, W = frame_bgr.shape[:2]
        n, ov = self.tiles, self.overlap
        tw, th = W / (n - (n - 1) * ov), H / (n - (n - 1) * ov)
        parts = [self._detect(frame_bgr, conf, None)]
        for iy in range(n):
            for ix in range(n):
                x0, y0 = int(ix * tw * (1 - ov)), int(iy * th * (1 - ov))
                d = self._detect(frame_bgr[y0:int(y0 + th), x0:int(x0 + tw)], conf, None)
                d.xyxy = d.xyxy + [x0, y0, x0, y0]
                parts.append(d)
        det = sv.Detections.merge(parts)
        return det if nms is None else det.with_nms(threshold=nms, class_agnostic=True)

    def _detect(self, frame_bgr, conf, nms):
        im = self.Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inp = self.proc(text=self.PROMPTS, images=im, return_tensors="pt").to(self.device)
        side = max(im.size)                                                            # OWLv2 pads to a square
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model(**inp)                                                    # fp16: ~5x faster than fp32
        out.logits, out.pred_boxes = out.logits.float(), out.pred_boxes.float()
        r = self.proc.post_process_grounded_object_detection(out, threshold=conf, target_sizes=[(side, side)])[0]
        det = sv.Detections(xyxy=r["boxes"].cpu().numpy().astype(float), confidence=r["scores"].cpu().numpy().astype(float),
                            class_id=np.zeros(len(r["scores"]), dtype=int))                # one class: ape
        det.xyxy = np.clip(det.xyxy, 0, [im.size[0], im.size[1]] * 2)
        return det if nms is None else det.with_nms(threshold=nms, class_agnostic=True)   # the prompts overlap


def read_video(path):
    """fps and a frame generator. ffmpeg instead of OpenCV: OpenCV silently fails on interlaced camcorder
    video (.MTS), and yadif only touches frames that are actually interlaced."""
    import imageio_ffmpeg
    frames = imageio_ffmpeg.read_frames(path, pix_fmt="bgr24", output_params=["-vf", "yadif=deint=interlaced"])
    meta = next(frames)
    w, h = meta["size"]
    return meta["fps"], (np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in frames)


# ---------------------------------------------------------------------------------------------------
# OpenApePose on all boxes of a frame at once (plus their mirrored crops for the flip test)
# ---------------------------------------------------------------------------------------------------
def keypoints_for_boxes(model, frame, boxes_xywh, pad, device):
    if not len(boxes_xywh):
        return np.zeros((0, 16, 3))
    crops, Ms = [], []
    for x, y, w, h in boxes_xywh:
        M = oap.crop_matrix(x - pad * w, y - pad * h, w * (1 + 2 * pad), h * (1 + 2 * pad))
        c = (cv2.warpAffine(frame, M, (oap.W, oap.H))[:, :, ::-1] / 255.0 - oap.MEAN) / oap.STD
        crops += [c, c[:, ::-1]]
        Ms.append(M)
    batch = torch.tensor(np.stack(crops).transpose(0, 3, 1, 2).copy(), dtype=torch.float32, device=device)
    with torch.no_grad():
        hm = model(batch).cpu().numpy()
    out = []
    for b, M in enumerate(Ms):
        mirrored = hm[2 * b + 1][oap.FLIP][:, :, ::-1]
        mirrored = np.concatenate([mirrored[:, :, :1], mirrored[:, :, :-1]], axis=2)
        px, py, conf = oap.peaks((hm[2 * b] + mirrored) / 2)
        back = cv2.invertAffineTransform(M)
        out.append(np.c_[back[0, 0] * px + back[0, 1] * py + back[0, 2], back[1, 0] * px + back[1, 1] * py + back[1, 2], conf])
    return np.array(out)


BOX_COLS = ["box_x", "box_y", "box_w", "box_h"]
SMOOTH = {"none": 0, "low": 0.1, "medium": 0.25, "high": 0.5}         # Savitzky-Golay window in seconds (order 2)


def detect_raw(path, detector):
    """Pass 1 (slow): every box the detector finds, threshold 0.05, overlapping boxes NOT merged, so threshold,
    merging and tracking can be tuned afterwards without detecting again. Columns: frame, conf, x0, y0, x1, y1,
    fps, n_frames (pixels of the original video)."""
    fps, frames = read_video(path)
    rows, n = [], 0
    for i, frame in enumerate(frames):
        d = detector(frame, 0.05, nms=None) if isinstance(detector, OWLv2) else detector(frame, 0.05)
        d = d[np.isin(d.class_id, KEEP_CLASSES)]
        rows += [{"frame": i, "conf": c, "x0": a, "y0": b, "x1": x, "y1": y} for (a, b, x, y), c in zip(d.xyxy, d.confidence)]
        n = i + 1
    return pd.DataFrame(rows, columns=["frame", "conf", "x0", "y0", "x1", "y1"]).assign(fps=fps, n_frames=n)


def track(dets, n_frames, fps, conf, nms, act, buffer_s, match):
    """Threshold -> merge overlapping boxes (NMS) -> ByteTrack. act: score needed to START a track (weaker boxes
    can only continue one); buffer_s: how long a lost track is kept; match: 1 - IoU needed to link a box to a track
    (0.9 = IoU >= 0.1, lenient, for fast-moving apes). One row per box, x/y = top-left corner."""
    tr = sv.ByteTrack(track_activation_threshold=act, lost_track_buffer=max(1, int(round(buffer_s * fps))),
                      minimum_matching_threshold=match, frame_rate=int(round(fps)))
    tr.reset()                                                           # track IDs start at 1 in every clip
    by_frame = {f: g for f, g in dets[dets.conf >= conf].groupby("frame")}
    rows = []
    for i in range(n_frames):
        g = by_frame.get(i)
        d = sv.Detections(xyxy=g[["x0", "y0", "x1", "y1"]].to_numpy(float), confidence=g.conf.to_numpy(float),
                          class_id=np.zeros(len(g), int)) if g is not None else sv.Detections.empty()
        if len(d):
            d = d.with_nms(threshold=nms, class_agnostic=True)
        d = tr.update_with_detections(d)
        for (x0, y0, x1, y1), tid, c in zip(d.xyxy, d.tracker_id, d.confidence):
            rows.append({"frame": i, "time_s": i / fps, "track_id": int(tid), "det_class": 0, "det_conf": float(c),
                         "box_x": x0, "box_y": y0, "box_w": x1 - x0, "box_h": y1 - y0})
    return pd.DataFrame(rows, columns=["frame", "time_s", "track_id", "det_class", "det_conf"] + BOX_COLS)


def stitch_tracks(boxes, fps, max_gap_s, max_dist=1.0, max_size_ratio=2.0):
    """Give a track that starts shortly after another one ended the older track's ID, when it starts close to
    where the old one ended: an ape that was hidden (behind a pole, in foliage) and reappears. ByteTrack itself
    looks for it where its motion model predicts, which drifts during longer occlusions.
    Conditions: new track starts 1..max_gap_s after the old one ended; start-to-end distance < max_dist x ape
    size (sqrt of box area); sizes differ < max_size_ratio. Candidate joins are made one-to-one, closest first, so
    two apes cannot take over each other's ID. Chains (A -> B -> C) are followed."""
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
    new_id, used_old, used_new = {}, set(), set()
    for _, old, new in sorted(cand):
        if old not in used_old and new not in used_new:
            new_id[new] = old
            used_old.add(old)
            used_new.add(new)
    def root(t):
        while t in new_id:
            t = new_id[t]
        return t
    out = boxes.copy()
    out["track_id"] = out.track_id.map(root)
    return out


def fill_gaps(boxes, fps, max_gap_s):
    """Linearly interpolate a track's box over gaps of at most max_gap_s seconds (frames where the detector missed
    it but it was found before and after). Filled rows get det_conf NaN."""
    max_gap = int(round(max_gap_s * fps))
    if max_gap < 1 or not len(boxes):
        return boxes.assign(interpolated=False)
    new = []
    for tid, g in boxes.sort_values("frame").groupby("track_id"):
        f = g.frame.to_numpy()
        for a, b in zip(np.flatnonzero(np.diff(f) > 1), np.flatnonzero(np.diff(f) > 1) + 1):
            if f[b] - f[a] - 1 > max_gap:
                continue
            ra, rb = g.iloc[a], g.iloc[b]
            for fr in range(f[a] + 1, f[b]):
                t = (fr - f[a]) / (f[b] - f[a])
                row = {c: ra[c] + t * (rb[c] - ra[c]) for c in BOX_COLS}
                row.update(frame=fr, time_s=fr / fps, track_id=tid, det_class=ra.det_class, det_conf=np.nan,
                           interpolated=True)
                new.append(row)
    boxes = boxes.assign(interpolated=False)
    return pd.concat([boxes, pd.DataFrame(new)], ignore_index=True) if new else boxes


def smooth_boxes(boxes, fps, level):
    """Savitzky-Golay (order 2) over each track's box centre, width and height. Only within runs of consecutive
    frames: a track is never smoothed across frames where it was not detected. Keeps the raw box as raw_box_*."""
    from scipy.signal import savgol_filter
    out = boxes.sort_values(["track_id", "frame"]).reset_index(drop=True)
    for col in BOX_COLS:
        out["raw_" + col] = out[col]
    win = int(round(SMOOTH[level] * fps)) | 1                            # odd number of frames
    if SMOOTH[level] == 0 or not len(out):
        return out
    win = max(win, 5)                                                    # order 2 needs >= 5 frames to smooth at all
    cx, cy = out.box_x + out.box_w / 2, out.box_y + out.box_h / 2
    sig = np.c_[cx, cy, out.box_w, out.box_h].astype(float)
    run = ((out.frame.diff() != 1) | (out.track_id.diff() != 0)).cumsum()
    for _, idx in out.groupby(run).groups.items():
        L = len(idx)
        w = min(win, L if L % 2 else L - 1)
        if w >= 5:
            sig[idx] = savgol_filter(sig[idx], w, 2, axis=0)
    out["box_w"], out["box_h"] = np.maximum(sig[:, 2], 1), np.maximum(sig[:, 3], 1)
    out["box_x"], out["box_y"] = sig[:, 0] - out.box_w / 2, sig[:, 1] - out.box_h / 2
    return out


def pose_and_render(path, out_dir, boxes, pose_model, device, pad, level):
    """Pass 2 (fast): OpenApePose on the (smoothed) boxes, CSV and labelled video. Smoothed box thick in the track
    colour, raw detector box thin grey, keypoints red (low) to green (high confidence)."""
    name = os.path.splitext(os.path.basename(path))[0]
    fps, frames = read_video(path)
    by_frame = {f: g for f, g in boxes.groupby("frame")}
    rows, writer, n = [], None, 0
    tmp = os.path.join(out_dir, f"{name}_tmp.mp4")
    for i, frame in enumerate(frames):
        n = i + 1
        g = by_frame.get(i, boxes.iloc[:0])
        xywh = g[BOX_COLS].to_numpy()
        kp = keypoints_for_boxes(pose_model, frame, xywh, pad, device)
        H, W = frame.shape[:2]
        scale = 960 / W                                                  # labelled video from the ACTUAL frame size
        view = np.ascontiguousarray(cv2.resize(frame, (960, int(round(H * scale)))))
        for j, r in enumerate(g.itertuples(index=False)):
            row = r._asdict()
            for k, name_k in enumerate(oap.KEYPOINTS):
                row[f"{name_k}_x"], row[f"{name_k}_y"], row[f"{name_k}_likelihood"] = kp[j, k]
            rows.append(row)
            col = COLOURS[r.track_id % len(COLOURS)]
            rx, ry, rw, rh = (np.array([r.raw_box_x, r.raw_box_y, r.raw_box_w, r.raw_box_h]) * scale).astype(int)
            cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (170, 170, 170), 1)
            x, y, w, h = (xywh[j] * scale).astype(int)
            interp = bool(getattr(r, "interpolated", False))
            cv2.rectangle(view, (x, y), (x + w, y + h), col, 1 if interp else 2)
            cv2.putText(view, f"id {r.track_id}" + (" interp." if interp else ""), (x, max(15, y - 5)), 0, 0.6, col, 2)
            for px, py, c in kp[j]:
                cv2.circle(view, (int(px * scale), int(py * scale)), 4, (0, int(255 * min(c, 1)), int(255 * (1 - min(c, 1)))), -1)
        cv2.putText(view, f"frame {i}   smoothing: {level}", (10, 25), 0, 0.6, (255, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, view.shape[1::-1])
        writer.write(view)
    writer.release()
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-crf", "24",
                    "-pix_fmt", "yuv420p", os.path.join(out_dir, f"{name}_labeled.mp4")], check=True)
    os.remove(tmp)
    return n, df


def run_clip(path, out_dir, det_dir, detector_fn, pose_model, device, p, redetect=False):
    """Pass 1 is saved in det_dir/<name>.csv and reused; pass 2 = track -> fill gaps -> smooth -> pose -> render."""
    name = os.path.splitext(os.path.basename(path))[0]
    cache = os.path.join(det_dir, f"{name}.csv")
    t0 = time.time()
    if os.path.exists(cache) and not redetect:
        dets = pd.read_csv(cache)
    else:
        dets = detect_raw(path, detector_fn())
        dets.round(3).to_csv(cache, index=False)
    fps = float(dets.fps.iloc[0])
    n = int(dets.n_frames.iloc[0]) if "n_frames" in dets else int(dets.frame.max()) + 1
    boxes = track(dets, n, fps, p.conf, p.nms, p.act, p.buffer, p.match)
    boxes = stitch_tracks(boxes, fps, p.stitch, p.stitch_dist)
    boxes = smooth_boxes(fill_gaps(boxes, fps, max(p.gap, p.stitch)), fps, p.smooth)
    n, df = pose_and_render(path, out_dir, boxes, pose_model, device, p.pad, p.smooth)
    return name, n, time.time() - t0, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="owlv2", help="owlv2, MDV6-mit-yolov9-c or MDV6-mit-yolov9-e")
    ap.add_argument("--clips", nargs="+", default=["14", "15", "18"])
    # defaults tuned on the Lab 5 SAM2 boxes of 11 siamang clips (tune_boxes.py)
    ap.add_argument("--conf", type=float, default=0.15, help="detector score threshold")
    ap.add_argument("--nms", type=float, default=0.7, help="merge boxes that overlap more than this (IoU)")
    ap.add_argument("--act", type=float, default=0.25, help="score needed to start a new track")
    ap.add_argument("--buffer", type=float, default=1.0, help="seconds a lost track is kept")
    ap.add_argument("--match", type=float, default=0.9, help="ByteTrack matching threshold (1 - IoU)")
    ap.add_argument("--gap", type=float, default=0.5, help="fill gaps in a track up to this many seconds")
    ap.add_argument("--stitch", type=float, default=2.0, help="re-join a track that reappears within this many "
                    "seconds (occlusion); 0 = off. Stitched gaps are filled and marked interpolated")
    ap.add_argument("--stitch-dist", type=float, default=1.0, help="... if it reappears within this x ape size")
    ap.add_argument("--smooth", default="low", choices=list(SMOOTH), help="Savitzky-Golay smoothing of the boxes")
    ap.add_argument("--pad", type=float, default=0.1, help="context around the box for OpenApePose (DeepWild: 0-0.1 best)")
    ap.add_argument("--tiles", type=int, default=1, help="OWLv2 on tiles x tiles enlarged crops too (far-away apes)")
    ap.add_argument("--redetect", action="store_true", help="ignore saved detections (results/dets_<detector>/)")
    ap.add_argument("--out", default=None, help="default: results/<detector>_smooth-<level>")
    args = ap.parse_args()
    tag = "owlv2" if args.detector == "owlv2" else args.detector.replace("MDV6-mit-yolov9-", "mit_")
    tag += f"_tiles{args.tiles}" if args.tiles > 1 else ""
    out = args.out or os.path.join(HERE, "results", f"{tag}_smooth-{args.smooth}")
    det_dir = os.path.join(HERE, "results", f"dets_{tag}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oap.WEIGHTS = WEIGHTS
    pose_model = oap.OpenApePose().to(device).eval()
    _det = []
    def detector_fn():                                                   # loaded only when pass 1 has to run
        if not _det:
            _det.append(OWLv2(device, tiles=args.tiles) if args.detector == "owlv2" else Detector(args.detector, device))
        return _det[0]
    videos = sorted(f for f in os.listdir(VIDEO) if f.lower().endswith((".mp4", ".mts", ".mov", ".avi")))
    def matches(f, c):
        return c == "all" or (f"_{c}_" in f if c.isdigit() else c in f)
    for path in [os.path.join(VIDEO, f) for f in videos if any(matches(f, c) for c in args.clips)]:
        name, n, secs, df = run_clip(path, out, det_dir, detector_fn, pose_model, device, args, args.redetect)
        n_ids = df.track_id.nunique() if len(df) else 0
        frames_with = df.frame.nunique() if len(df) else 0
        print(f"clip {name}: {n} frames in {secs:.0f} s ({n / secs:.1f} fps) | frames with a box {frames_with / n:.0%} | "
              f"track IDs {n_ids} | boxes/frame {len(df) / n:.2f}", flush=True)
