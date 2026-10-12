"""
Build the demo videos: every video in demo/samples/ tracked with the package with every pose model (ensemble,
openapepose, vitpose) at every smoothing level (boxes and keypoints alike), drawn as skeleton and as points, 960 wide.
OWLv2 runs once per video (first model, level "none"); every other model and level reuses its saved detections, so
the models differ only in their keypoints.

    python tools/make_demo.py                                  # all models
    python tools/make_demo.py --models ensemble vitpose        # some of them

Writes demo/output/<model>/<level>/<name>_{skeleton,points}.mp4 and _tracks.csv, and demo/output/manifest.json.
Videos already made are skipped (delete a folder to remake it).
"""
import argparse
import json
import os
import shutil
import sys

import imageio_ffmpeg
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from envisionboxbio_apetrack import SMOOTHING, ApeTracker, video              # noqa: E402
from envisionboxbio_apetrack.tracker import POSE_MODELS                     # noqa: E402

DEMO = os.path.join(HERE, "..", "demo")
SAMPLES, OUT = os.path.join(DEMO, "samples"), os.path.join(DEMO, "output")
STYLES = ("skeleton", "points")

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=list(POSE_MODELS), choices=POSE_MODELS)
args = ap.parse_args()

names = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".mp4", ".mts", ".mov")))
manifest = []
for f in names:
    name, src = os.path.splitext(f)[0], os.path.join(SAMPLES, f)
    entry = {"name": name, "file": f, "models": {}}
    # detections: the first run's, or an earlier make_demo run's (demo/output/none/, before the pose model option)
    sources = [os.path.join(OUT, m, "none", f"{name}_detections.csv") for m in args.models]
    sources.append(os.path.join(OUT, "none", f"{name}_detections.csv"))
    for model in args.models:
        entry["models"][model] = {}
        for level in SMOOTHING:                                      # "none" first
            out = os.path.join(OUT, model, level)
            os.makedirs(out, exist_ok=True)
            det = os.path.join(out, f"{name}_detections.csv")
            first = next((s for s in sources if os.path.exists(s)), None)
            if not os.path.exists(det) and first is not None:
                shutil.copy(first, det)
            if not os.path.exists(os.path.join(out, f"{name}_{STYLES[0]}.mp4")):
                ApeTracker(pose_model=model, box_smoothing=level, keypoint_smoothing=level, draw=STYLES,
                           output_width=960).process_video(src, out)
            tr = pd.read_csv(os.path.join(out, f"{name}_tracks.csv"))
            entry["models"][model][level] = {"apes": int(tr.track_id.nunique()) if len(tr) else 0,
                                             "videos": {s: f"output/{model}/{level}/{name}_{s}.mp4" for s in STYLES}}
    meta = next(imageio_ffmpeg.read_frames(src))
    n_in = sum(1 for _ in video.read_video(src)[2])                  # frames as the pipeline reads them
    out_video = os.path.join(OUT, args.models[0], "none", f"{name}_{STYLES[0]}.mp4")
    n_out, secs = imageio_ffmpeg.count_frames_and_secs(out_video)
    entry.update(fps=round(meta["fps"], 3), fps_out=round(next(imageio_ffmpeg.read_frames(out_video))["fps"], 3), width=meta["size"][0], height=meta["size"][1], seconds=round(secs, 1),
                 frames_in=n_in, frames_out=n_out, size_mb=round(os.path.getsize(src) / 1e6, 1))
    manifest.append(entry)
    print(f"{name}: {n_in} frames in, {n_out} out, {entry['models'][args.models[0]]['low']['apes']} apes "
          f"({args.models[0]}, low)", flush=True)
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
