"""
Pipeline test, SAM2 route: OWLv2 (finds apes) -> SAM2 (follows each ape as a mask) -> OpenApePose (16 keypoints).

    python sam2_pipeline_test.py --clips 00019_corde grooming2 14 15 --out results/sam2_owlv2
    python sam2_pipeline_test.py --clips all

No ByteTrack: the SAM2 object number IS the track ID. SAM2 cannot add objects once tracking has started and keeps
all frames of a video in memory, so the video is processed in chunks of --chunk seconds that overlap by one frame:
    - each chunk starts from the last masks of the previous chunk, with the same IDs (identity carries over);
    - OWLv2 runs on the first frame of every chunk; an ape that no mask covers yet becomes a new object;
    - an object without any OWLv2 detection on it at --drop chunk starts in a row is dropped (mask drifted off).
The box of an ape is the outline of its mask. SAM2 runs on 960-wide frames (it works at 1024 internally).

Writes, per clip, to --out: <name>.csv (same columns as pipeline_test.py, det_conf = mask pixel fraction in its box)
and <name>_labeled.mp4 (masks, boxes, IDs, keypoints).
"""
import argparse
import os
import shutil
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HERE, "..", "models", "hf"))     # all downloads stay in models/

import cv2
import numpy as np
import pandas as pd
import torch

import pipeline_test as pt
from pipeline_test import oap

SMALL_W = 960
SAM2_MODEL = "facebook/sam2.1-hiera-large"


def box_of(mask):
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def run_clip(path, out_dir, sam, owl, pose_model, device, conf, pad, chunk_s, drop):
    name = os.path.splitext(os.path.basename(path))[0]
    fps, frames = pt.read_video(path)
    chunk = max(2, int(round(chunk_s * fps)))
    tmp_jpg = tempfile.mkdtemp(prefix="sam2_")
    tmp_mp4 = os.path.join(out_dir, f"{name}_tmp.mp4")
    writer, rows = None, []
    carry = {}                     # obj_id -> small mask on the first frame of the next chunk
    misses = {}                    # obj_id -> chunk starts in a row without an OWLv2 detection on it
    next_id, i0, t0 = 1, 0, time.time()
    buf = [next(frames)]
    for f in frames:
        buf.append(f)
        if len(buf) < chunk + 1:
            continue
        next_id = process_chunk(buf, i0, fps, name, sam, owl, pose_model, device, conf, pad, drop, carry, misses,
                                next_id, tmp_jpg, rows, writer_box := [writer, tmp_mp4], last=False)
        writer = writer_box[0]
        i0 += len(buf) - 1
        buf = buf[-1:]             # chunks overlap by one frame
    process_chunk(buf, i0, fps, name, sam, owl, pose_model, device, conf, pad, drop, carry, misses,
                  next_id, tmp_jpg, rows, writer_box := [writer, tmp_mp4], last=True)
    writer_box[0].release()
    shutil.rmtree(tmp_jpg, ignore_errors=True)
    n = i0 + len(buf)
    secs = time.time() - t0
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", tmp_mp4, "-c:v", "libx264", "-crf", "24",
                    "-pix_fmt", "yuv420p", os.path.join(out_dir, f"{name}_labeled.mp4")], check=True)
    os.remove(tmp_mp4)
    return name, n, secs, df


def process_chunk(buf, i0, fps, name, sam, owl, pose_model, device, conf, pad, drop, carry, misses, next_id,
                  tmp_jpg, rows, writer_box, last):
    H, W = buf[0].shape[:2]
    scale = W / SMALL_W
    small_h = int(round(H / scale))
    for f in os.listdir(tmp_jpg):
        os.remove(os.path.join(tmp_jpg, f))
    for k, fr in enumerate(buf):
        cv2.imwrite(os.path.join(tmp_jpg, f"{k:05d}.jpg"), cv2.resize(fr, (SMALL_W, small_h)), [cv2.IMWRITE_JPEG_QUALITY, 95])

    # --- who is there at the start of this chunk: carried masks + OWLv2 detections ---
    det = owl(buf[0], conf)
    det_small = det.xyxy / scale
    carried_boxes = {oid: box_of(m) for oid, m in carry.items()}
    matched_dets = set()
    for oid, b in carried_boxes.items():
        hit = [j for j, d in enumerate(det_small) if iou(b, d) >= 0.3]
        matched_dets |= set(hit)
        misses[oid] = 0 if hit else misses.get(oid, 0) + 1
    for oid in [o for o in carry if misses[o] >= drop]:
        del carry[oid], misses[oid]
    new = [(next_id + k, d) for k, d in enumerate(j for j_, j in enumerate(det_small) if j_ not in matched_dets)]
    next_id += len(new)
    for oid, _ in new:
        misses[oid] = 0

    masks = {k: {} for k in range(len(buf))}         # frame in chunk -> obj_id -> small mask
    if carry or new:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam.init_state(tmp_jpg)
            for oid, m in carry.items():
                sam.add_new_mask(state, 0, oid, m)
            for oid, d in new:
                sam.add_new_points_or_box(state, 0, oid, box=d)
            for k, oids, logits in sam.propagate_in_video(state):
                for oid, lg in zip(oids, logits):
                    m = (lg[0] > 0).cpu().numpy()
                    if m.any():
                        masks[k][int(oid)] = m
    carry.clear()
    carry.update(masks[len(buf) - 1])
    for oid in list(misses):
        if oid not in carry:
            misses.pop(oid)                        # SAM2 lost it: gone (a new detection gets a new ID)

    # --- output: skip the last frame (it is the first frame of the next chunk) unless this is the last chunk ---
    for k in range(len(buf) if last else len(buf) - 1):
        frame, i = buf[k], i0 + k
        objs = sorted(masks[k].items())
        boxes = [box_of(m) * scale for _, m in objs]
        xywh = [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in boxes]
        kp = pt.keypoints_for_boxes(pose_model, frame, xywh, pad, device)
        view = np.ascontiguousarray(cv2.resize(frame, (SMALL_W, small_h)))
        for j, (oid, m) in enumerate(objs):
            col = pt.COLOURS[oid % len(pt.COLOURS)]
            view[m] = (0.55 * view[m] + 0.45 * np.array(col)).astype(np.uint8)
            x, y, w, h = xywh[j]
            fill = m.sum() * scale ** 2 / max(w * h, 1)
            row = {"frame": i, "time_s": i / fps, "track_id": oid, "det_class": 0, "det_conf": float(fill),
                   "box_x": x, "box_y": y, "box_w": w, "box_h": h}
            for kk, name_k in enumerate(oap.KEYPOINTS):
                row[f"{name_k}_x"], row[f"{name_k}_y"], row[f"{name_k}_likelihood"] = kp[j, kk]
            rows.append(row)
            s = 1 / scale
            cv2.rectangle(view, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), col, 2)
            cv2.putText(view, f"id {oid}", (int(x * s), max(15, int(y * s) - 5)), 0, 0.6, col, 2)
            for px, py, c in kp[j]:
                cv2.circle(view, (int(px * s), int(py * s)), 4, (0, int(255 * min(c, 1)), int(255 * (1 - min(c, 1)))), -1)
        if k == 0:
            for d in det_small:                    # OWLv2 seeds, thin white
                cv2.rectangle(view, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 255, 255), 1)
        cv2.putText(view, f"frame {i}", (10, 25), 0, 0.6, (255, 255, 255), 2)
        if writer_box[0] is None:
            writer_box[0] = cv2.VideoWriter(writer_box[1], cv2.VideoWriter_fourcc(*"mp4v"), fps, view.shape[1::-1])
        writer_box[0].write(view)
    return next_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["00019_corde", "grooming2", "14", "15"])
    ap.add_argument("--conf", type=float, default=0.2, help="OWLv2 threshold")
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--chunk", type=float, default=1.0, help="seconds per SAM2 chunk (= how often OWLv2 looks)")
    ap.add_argument("--drop", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sam2_owlv2"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    oap.WEIGHTS = pt.WEIGHTS
    pose_model = oap.OpenApePose().to(device).eval()
    owl = pt.OWLv2(device)
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    sam = SAM2VideoPredictor.from_pretrained(SAM2_MODEL, device=device)
    videos = sorted(f for f in os.listdir(pt.VIDEO) if f.lower().endswith((".mp4", ".mts", ".mov", ".avi")))
    def matches(f, c):
        return c == "all" or (f"_{c}_" in f if c.isdigit() else c in f)
    for path in [os.path.join(pt.VIDEO, f) for f in videos if any(matches(f, c) for c in args.clips)]:
        name, n, secs, df = run_clip(path, args.out, sam, owl, pose_model, device, args.conf, args.pad, args.chunk, args.drop)
        n_ids = df.track_id.nunique() if len(df) else 0
        frames_with = df.frame.nunique() if len(df) else 0
        print(f"clip {name}: {n} frames in {secs:.0f} s ({n / secs:.1f} fps) | frames with a box {frames_with / n:.0%} | "
              f"track IDs {n_ids} | boxes/frame {len(df) / n:.2f}", flush=True)
