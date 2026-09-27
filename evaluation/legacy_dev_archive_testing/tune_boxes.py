"""
Tune the box steps against pseudo-ground truth: the Lab 5 SAM2 boxes of ONE siamang per clip (clicked by hand on
the first frame, then followed by SAM2), in ../testdata_lab5/<name>.csv (frame, x, y, w, h, ...).

    python tune_boxes.py                 # detect once (cached), then grid search, prints the best settings
    python tune_boxes.py --show 20       # show more rows

Step 1 (slow, once): OWLv2 on every frame, threshold 0.05, NO merging of overlapping boxes -> results/dets_owlv2/.
Step 2 (fast, per setting): merge overlaps (NMS) -> threshold -> ByteTrack -> fill short gaps -> smooth, scored on
    found     share of frames where some box has IoU >= 0.5 with the siamang's box
    mIoU      mean IoU of the best box with the siamang's box (0 when there is none)
    switches  how often the track ID on the siamang changes (IoU >= 0.5 frames), summed over clips
    jitter    frame-to-frame wobble of (our box - SAM2 box): mean |second difference| of the centre and size
              difference, in % of the siamang's box diagonal. Real motion cancels out; smoothing lowers it.
Metrics are averaged over clips (every clip counts equally).
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import supervision as sv

import pipeline_test as pt

GT_DIR = os.path.join(pt.HERE, "..", "testdata_lab5")
DET_DIR = os.path.join(pt.HERE, "results", "dets_owlv2")


def detect_all(names):
    os.makedirs(DET_DIR, exist_ok=True)
    owl = None
    for name in names:
        out = os.path.join(DET_DIR, f"{name}.csv")
        if os.path.exists(out):
            continue
        owl = owl or pt.OWLv2("cuda")
        fps, frames = pt.read_video(os.path.join(pt.VIDEO, f"{name}.mp4"))
        rows = []
        for i, fr in enumerate(frames):
            d = owl(fr, 0.05, nms=None)
            rows += [{"frame": i, "conf": c, "x0": a, "y0": b, "x1": x, "y1": y} for (a, b, x, y), c in zip(d.xyxy, d.confidence)]
        pd.DataFrame(rows).assign(fps=fps).round(3).to_csv(out, index=False)
        print("detected", name, flush=True)


track = pt.track                       # threshold -> NMS -> ByteTrack, as in the pipeline


def score(boxes, gt):
    """Per clip: found, mIoU, switches, jitter (see module doc). Vectorised: one join of our boxes with the SAM2 box."""
    g = gt[gt.w > 0][["frame", "x", "y", "w", "h"]]
    m = g.merge(boxes[["frame", "track_id"] + pt.BOX_COLS], on="frame", how="left")
    ix = np.clip(np.minimum(m.x + m.w, m.box_x + m.box_w) - np.maximum(m.x, m.box_x), 0, None)
    iy = np.clip(np.minimum(m.y + m.h, m.box_y + m.box_h) - np.maximum(m.y, m.box_y), 0, None)
    m["iou"] = (ix * iy / (m.w * m.h + m.box_w * m.box_h - ix * iy)).fillna(0)
    best = m.loc[m.groupby("frame").iou.idxmax()].set_index("frame").reindex(range(int(gt.frame.max()) + 1))
    iou = best.iou.fillna(0).to_numpy()
    ok = iou >= 0.5
    ids = best.track_id.to_numpy()[ok]
    diag = np.hypot(best.w, best.h).to_numpy()
    d = np.c_[(best.box_x + best.box_w / 2) - (best.x + best.w / 2), (best.box_y + best.box_h / 2) - (best.y + best.h / 2),
              best.box_w - best.w, best.box_h - best.h] / diag[:, None]
    d[~ok] = np.nan
    j = np.abs(d[:-2] - 2 * d[1:-1] + d[2:]).mean(1)
    return {"found": ok[gt.frame.min():].mean(), "mIoU": iou.mean(), "switches": int((ids[1:] != ids[:-1]).sum()),
            "jitter": 100 * np.nanmean(j) if np.isfinite(j).any() else np.nan}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()
    names = sorted(f[:-4] for f in os.listdir(GT_DIR) if f.endswith(".csv"))
    detect_all(names)
    data = {}
    for n in names:
        dets = pd.read_csv(os.path.join(DET_DIR, f"{n}.csv"))
        gt = pd.read_csv(os.path.join(GT_DIR, f"{n}.csv"))
        data[n] = (dets, gt, float(dets.fps.iloc[0]))

    grid = {"conf": [0.1, 0.15, 0.2, 0.3], "nms": [0.3, 0.5, 0.7], "act": [0.15, 0.25, 0.35],
            "buffer_s": [0.5, 1.0, 2.0], "match": [0.7, 0.8, 0.9]}
    post = {"gap_s": [0, 0.25, 0.5], "smooth": list(pt.SMOOTH)}
    baseline = dict(conf=0.2, nms=0.5, act=0.25, buffer_s=30 / 60, match=0.8, gap_s=0, smooth="none")
    results = []
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    for k, c in enumerate([{**{k: baseline[k] for k in grid}}] + combos):
        tracks = {n: track(d, int(gt.frame.max()) + 1, fps, **c) for n, (d, gt, fps) in data.items()}
        for gap, sm in itertools.product(*post.values()):
            per = [score(pt.smooth_boxes(pt.fill_gaps(tracks[n], fps, gap), fps, sm), gt) for n, (d, gt, fps) in data.items()]
            m = pd.DataFrame(per)
            results.append({**c, "gap_s": gap, "smooth": sm, "found": m.found.mean(), "mIoU": m.mIoU.mean(),
                            "switches": int(m.switches.sum()), "jitter": m.jitter.mean(), "baseline": k == 0})
        if k % 50 == 0:
            print(f"{k}/{len(combos)} settings", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(pt.HERE, "results", "tune_boxes.csv"), index=False)
    pd.set_option("display.width", 200)
    cols = list(grid) + list(post) + ["found", "mIoU", "switches", "jitter"]
    base = res[res.baseline & (res.gap_s == 0) & (res.smooth == "none")]
    print("\nBASELINE (current settings)\n", base[cols].round(3).to_string(index=False))
    print("\nBEST by mIoU, then fewest switches, then least jitter\n",
          res.sort_values(["mIoU", "switches", "jitter"], ascending=[False, True, True])[cols].head(args.show).round(3).to_string(index=False))
    for sm in pt.SMOOTH:
        r = res[res.smooth == sm].sort_values(["mIoU", "switches"], ascending=[False, True]).iloc[0]
        print(f"best with smoothing {sm:6s}: mIoU {r.mIoU:.3f} found {r.found:.3f} switches {r.switches} jitter {r.jitter:.2f}")
