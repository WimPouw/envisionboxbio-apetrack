"""
Diagnose box problems: raw MegaDetector detections (all classes, conf >= 0.05) per frame, saved to
results/diag/<name>_raw.csv, then coverage at several thresholds with and without ByteTrack.

    python diag_boxes.py --clips 14 15 00019_corde --detector MDV6-mit-yolov9-c
"""
import argparse, os
import cv2, numpy as np, pandas as pd, supervision as sv
import pipeline_test as pt

ap = argparse.ArgumentParser()
ap.add_argument("--detector", default="MDV6-mit-yolov9-c")
ap.add_argument("--clips", nargs="+", default=["14", "15", "00019_corde"])
args = ap.parse_args()
out = os.path.join(pt.HERE, "results", "diag"); os.makedirs(out, exist_ok=True)
det = pt.Detector(args.detector, "cuda")
tag = args.detector.split("-")[-1]
for f in sorted(os.listdir(pt.VIDEO)):
    if not any((f"_{c}_" in f if c.isdigit() else c in f) for c in args.clips):
        continue
    name = os.path.splitext(f)[0]
    cap = cv2.VideoCapture(os.path.join(pt.VIDEO, f)); fps = cap.get(cv2.CAP_PROP_FPS)
    rows, dets, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        d = det(fr, 0.05); dets.append(d)
        for (x0, y0, x1, y1), c, k in zip(d.xyxy, d.confidence, d.class_id):
            rows.append(dict(frame=i, cls=k, conf=c, x0=x0, y0=y0, x1=x1, y1=y1))
        i += 1
    raw = pd.DataFrame(rows); raw.to_csv(os.path.join(out, f"{name}_{tag}_raw.csv"), index=False)
    n = i
    line = [f"{name[:40]:40s} n={n}"]
    for th in (0.05, 0.1, 0.2, 0.3):
        r = raw[(raw.conf >= th) & raw.cls.isin([0, 1])]
        line.append(f"raw>={th}: {r.frame.nunique()/n:.0%} ({len(r)/n:.2f}/fr)")
    # ByteTrack with the pipeline's settings (conf 0.2)
    for label, kw in [("BT default", {}), ("BT act=0.1", dict(track_activation_threshold=0.1))]:
        tr = sv.ByteTrack(frame_rate=int(round(fps)), **kw); cov, ids = 0, set()
        for d in dets:
            d = d[(d.confidence >= 0.2) & np.isin(d.class_id, [0, 1])]
            t = tr.update_with_detections(d); cov += len(t) > 0; ids |= set(t.tracker_id.tolist())
        line.append(f"{label}: {cov/n:.0%} ids={len(ids)}")
    cls = raw[raw.conf >= 0.2].cls.value_counts().to_dict()
    line.append(f"classes>=0.2 {cls}")
    print(" | ".join(line), flush=True)
