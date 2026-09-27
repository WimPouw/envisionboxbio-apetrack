"""
Build the demo videos: every video in demo/samples/ tracked with the package at every smoothing level (boxes and
keypoints alike), drawn as skeleton and as points, 960 wide. OWLv2 runs once per video (level "none"); the other
levels reuse its saved detections.

    python tools/make_demo.py

Writes demo/output/<level>/<name>_{skeleton,points}.mp4 and _tracks.csv, and demo/output/manifest.json.
"""
import json
import os
import shutil
import sys

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from envisionboxbio_apetrack import SMOOTHING, ApeTracker, video              # noqa: E402

DEMO = os.path.join(HERE, "..", "demo")
SAMPLES, OUT = os.path.join(DEMO, "samples"), os.path.join(DEMO, "output")
STYLES = ("skeleton", "points")

names = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".mp4", ".mts", ".mov")))
manifest = []
for f in names:
    name, src = os.path.splitext(f)[0], os.path.join(SAMPLES, f)
    entry = {"name": name, "file": f, "levels": {}}
    for level in SMOOTHING:                                          # "none" first: it runs the detector
        out = os.path.join(OUT, level)
        os.makedirs(out, exist_ok=True)
        det = os.path.join(out, f"{name}_detections.csv")
        first = os.path.join(OUT, "none", f"{name}_detections.csv")
        if level != "none" and not os.path.exists(det) and os.path.exists(first):
            shutil.copy(first, det)
        if not os.path.exists(os.path.join(out, f"{name}_{STYLES[0]}.mp4")):
            ApeTracker(box_smoothing=level, keypoint_smoothing=level, draw=STYLES, output_width=960).process_video(src, out)
        import pandas as pd
        tr = pd.read_csv(os.path.join(out, f"{name}_tracks.csv"))
        entry["levels"][level] = {"apes": int(tr.track_id.nunique()) if len(tr) else 0,
                                  "videos": {s: f"output/{level}/{name}_{s}.mp4" for s in STYLES}}
    meta = next(imageio_ffmpeg.read_frames(src))
    n_in = sum(1 for _ in video.read_video(src)[2])                  # frames as the pipeline reads them
    out_video = os.path.join(OUT, "none", f"{name}_{STYLES[0]}.mp4")
    n_out, secs = imageio_ffmpeg.count_frames_and_secs(out_video)
    entry.update(fps=round(meta["fps"], 3), fps_out=round(next(imageio_ffmpeg.read_frames(out_video))["fps"], 3), width=meta["size"][0], height=meta["size"][1], seconds=round(secs, 1),
                 frames_in=n_in, frames_out=n_out, size_mb=round(os.path.getsize(src) / 1e6, 1))
    manifest.append(entry)
    print(f"{name}: {n_in} frames in, {n_out} out, {entry['levels']['low']['apes']} apes (low)", flush=True)
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
