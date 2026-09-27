"""
Tune detection + pose against DeepWild (Wiltshire et al. 2023): hand-labelled wild chimpanzees and bonobos,
18 keypoints per ape, often several apes per frame. Single frames, so this tests the detector and OpenApePose,
not tracking or smoothing (see tune_boxes.py for those).

    python tune_deepwild.py            # build ground truth + cache (slow, once), then grid search

Labels: ../testdata_wiltshire/DeepWild-main/DeepWild Download/DeepWild1.1/labeled-data/*/CollectedData_*.csv
Images: the labeled-data/<folder>/<img> files anywhere under ../testdata_wiltshire (matched by folder + file name).

Keypoints compared (DeepWild -> OpenApePose): nose, eyes, lower_neck -> neck, shoulders, elbows, wrists, knees,
ankles, and the midpoint of left_hip/right_hip -> hip. 15 in total (DeepWild's ears and OpenApePose's head have
no counterpart).

Scores, over all labelled apes (an ape with >= 4 labelled keypoints):
    PCK@0.1 / @0.2   share of its keypoints within 0.1 / 0.2 x ape size of the label; ape size = sqrt(area of the
                     box around ALL its labelled keypoints). An ape without a matching detection scores 0.
    found            share of labelled apes matched to a detected ape (Hungarian on keypoint distance)
    extra/img        detected apes not matched to a labelled ape (not all apes in a frame are labelled!)
"oracle" = boxes around the labelled keypoints (+15% each side): what OpenApePose does with perfect boxes.
"""
import argparse
import collections
import csv
import glob
import itertools
import os
import pickle

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

import pipeline_test as pt
from pipeline_test import oap

ROOT = os.path.join(pt.HERE, "..", "testdata_wiltshire", "DeepWild-main")
LABELS = os.path.join(ROOT, "DeepWild Download", "DeepWild1.1", "labeled-data")
CACHE = os.path.join(pt.HERE, "results", "deepwild_cache.pkl")
PADS = [0.0, 0.1, 0.2, 0.3, 0.4]
MAP = {"nose": "nose", "left_eye": "left_eye", "right_eye": "right_eye", "neck": "lower_neck",
       "left_shoulder": "left_shoulder", "right_shoulder": "right_shoulder", "left_elbow": "left_elbow",
       "right_elbow": "right_elbow", "left_wrist": "left_wrist", "right_wrist": "right_wrist",
       "left_knee": "left_knee", "right_knee": "right_knee", "left_ankle": "left_ankle", "right_ankle": "right_ankle"}
COMPARED = list(MAP) + ["hip"]                                 # OpenApePose names
OAP_IDX = [oap.KEYPOINTS.index(k) for k in COMPARED]


def load_ground_truth():
    """-> list of dicts: image path, site, apes = [ {kp: (15,2) array with NaN, extent: (x0,y0,x1,y1)} ]"""
    images = {}
    for f in glob.glob(os.path.join(ROOT, "**", "labeled-data", "*", "*.*"), recursive=True):
        if f.lower().endswith((".png", ".jpg")) and "DeepWild_paper" not in f:
            images.setdefault((os.path.basename(os.path.dirname(f)).lower(), os.path.basename(f).lower()), []).append(f)
    out = []
    for f in sorted(glob.glob(os.path.join(LABELS, "*", "CollectedData_*.csv"))):
        R = list(csv.reader(open(f)))
        nidx = next(i for i, c in enumerate(R[3]) if c in ("x", "y"))          # 1 or 3 index columns
        inds, parts, coords = R[1], R[2], R[3]
        for r in R[4:]:
            if not r:
                continue
            path = "/".join(c for c in r[:nidx] if c).replace("\\", "/").split("/")
            key = (path[-2].lower() if len(path) > 1 else os.path.basename(os.path.dirname(f)).lower(), path[-1].lower())
            if key not in images:
                continue
            apes = collections.defaultdict(dict)
            for i in range(nidx, len(r)):
                if r[i] not in ("", "NaN", "nan"):
                    apes[inds[i]].setdefault(parts[i], [np.nan, np.nan])["xy".index(coords[i])] = float(r[i])
            gt_apes = []
            for pts in apes.values():
                allxy = np.array([v for v in pts.values() if np.isfinite(v).all()])
                if len(allxy) < 4:
                    continue
                kp = np.full((len(COMPARED), 2), np.nan)
                for j, k in enumerate(COMPARED[:-1]):
                    if MAP[k] in pts:
                        kp[j] = pts[MAP[k]]
                if "left_hip" in pts and "right_hip" in pts:
                    kp[-1] = (np.array(pts["left_hip"]) + np.array(pts["right_hip"])) / 2
                gt_apes.append({"kp": kp, "extent": (*allxy.min(0), *allxy.max(0))})
            if gt_apes:
                out.append({"images": images[key], "site": "".join(c for c in key[0] if c.isalpha()), "apes": gt_apes})
    return out


def build_cache(gt):
    owl = pt.OWLv2("cuda")
    oap.WEIGHTS = pt.WEIGHTS
    pose = oap.OpenApePose().to("cuda").eval()
    cache = []
    for n, item in enumerate(gt):
        img = next((im for im in (cv2.imread(p) for p in item["images"]) if im is not None), None)
        if img is None:                                     # every copy of this image is unreadable
            print("unreadable:", item["images"][0], flush=True)
            cache.append(None)
            continue
        d = owl(img, 0.05, nms=None)
        det_xywh = [(a, b, x - a, y - b) for a, b, x, y in d.xyxy]
        orc_xywh = []
        for x0, y0, x1, y1 in (a["extent"] for a in item["apes"]):
            mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
            orc_xywh.append((x0 - mx, y0 - my, x1 - x0 + 2 * mx, y1 - y0 + 2 * my))
        entry = {"det_xyxy": d.xyxy, "det_conf": d.confidence, "det_kp": {}, "oracle_kp": {}}
        for pad in PADS:
            entry["det_kp"][pad] = pt.keypoints_for_boxes(pose, img, det_xywh, pad, "cuda")
            entry["oracle_kp"][pad] = pt.keypoints_for_boxes(pose, img, orc_xywh, pad, "cuda")
        cache.append(entry)
        if n % 100 == 0:
            print(f"cached {n}/{len(gt)}", flush=True)
    return cache


def score_image(gt_apes, pred_kp):
    """pred_kp: (n_pred, 16, 3). Returns per labelled ape the per-keypoint normalised errors (inf when unmatched)."""
    errs = []
    sizes = [np.sqrt(max((a["extent"][2] - a["extent"][0]) * (a["extent"][3] - a["extent"][1]), 1)) for a in gt_apes]
    if len(pred_kp):
        P = pred_kp[:, OAP_IDX, :2]
        cost = np.zeros((len(gt_apes), len(P)))
        for i, a in enumerate(gt_apes):
            ok = np.isfinite(a["kp"][:, 0])
            e = np.linalg.norm(P[:, ok] - a["kp"][ok], axis=2) / sizes[i]
            cost[i] = np.minimum(e, 1).mean(1)
        gi, pi = linear_sum_assignment(cost)
        match = {g: p for g, p in zip(gi, pi) if cost[g, p] < 0.5}
    else:
        match = {}
    for i, a in enumerate(gt_apes):
        ok = np.isfinite(a["kp"][:, 0])
        e = np.full(len(COMPARED), np.nan)
        e[ok] = np.linalg.norm(pred_kp[match[i], OAP_IDX, :2][ok] - a["kp"][ok], axis=1) / sizes[i] if i in match else np.inf
        errs.append(e)
    return errs, len(match), (len(pred_kp) - len(match)) if len(pred_kp) else 0


def summarise(all_errs, found, n_apes, extra, n_img):
    E = np.concatenate([e[np.newaxis] for e in all_errs])
    v = E[~np.isnan(E)]
    return {"PCK@0.1": (v < 0.1).mean(), "PCK@0.2": (v < 0.2).mean(), "found": found / n_apes, "extra/img": extra / n_img}


def evaluate(gt, cache, conf, nms, pad, oracle=False, idx=None):
    import supervision as sv
    errs, found, extra, n_apes = [], 0, 0, 0
    for k in (range(len(gt)) if idx is None else idx):
        item, c = gt[k], cache[k]
        if oracle:
            pk = c["oracle_kp"][pad]
        else:
            keep = np.flatnonzero(c["det_conf"] >= conf)
            if len(keep):
                d = sv.Detections(xyxy=c["det_xyxy"][keep], confidence=c["det_conf"][keep], class_id=np.zeros(len(keep), int))
                d.data["i"] = keep
                keep = d.with_nms(threshold=nms, class_agnostic=True).data["i"]
            pk = c["det_kp"][pad][keep] if len(keep) else np.zeros((0, 16, 3))
        e, f, x = score_image(item["apes"], pk)
        errs += e; found += f; extra += x; n_apes += len(item["apes"])
    n_img = len(gt) if idx is None else len(idx)
    return summarise(errs, found, n_apes, extra, n_img), errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    gt = load_ground_truth()
    print(f"{len(gt)} labelled frames with images, {sum(len(g['apes']) for g in gt)} apes,",
          dict(collections.Counter(g["site"] for g in gt)), flush=True)
    if os.path.exists(CACHE) and not args.rebuild:
        cache = pickle.load(open(CACHE, "rb"))
    else:
        cache = build_cache(gt)
        pickle.dump(cache, open(CACHE, "wb"))
    gt, cache = [g for g, c in zip(gt, cache) if c is not None], [c for c in cache if c is not None]

    rows = []
    for pad in PADS:
        rows.append({"boxes": "oracle", "conf": None, "nms": None, "pad": pad, **evaluate(gt, cache, 0, 0, pad, oracle=True)[0]})
    for conf, nms, pad in itertools.product([0.05, 0.1, 0.15, 0.2, 0.3], [0.3, 0.5, 0.7], PADS):
        rows.append({"boxes": "owlv2", "conf": conf, "nms": nms, "pad": pad, **evaluate(gt, cache, conf, nms, pad)[0]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(pt.HERE, "results", "tune_deepwild.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nORACLE boxes (upper bound for OpenApePose)\n", res[res.boxes == "oracle"].round(3).to_string(index=False))
    base = res[(res.boxes == "owlv2") & (res.conf == 0.2) & (res.nms == 0.5) & (res["pad"] == 0.2)]
    print("\nBASELINE (conf 0.2, nms 0.5, pad 0.2)\n", base.round(3).to_string(index=False))
    print("\nBEST OWLv2 settings by PCK@0.2\n", res[res.boxes == "owlv2"].sort_values("PCK@0.2", ascending=False).head(10).round(3).to_string(index=False))

    best = res[res.boxes == "owlv2"].sort_values("PCK@0.2", ascending=False).iloc[0]
    print(f"\nBest setting per site (conf {best.conf}, nms {best.nms}, pad {best['pad']}) vs baseline:")
    for site in sorted({g["site"] for g in gt}):
        idx = [k for k, g in enumerate(gt) if g["site"] == site]
        b = evaluate(gt, cache, 0.2, 0.5, 0.2, idx=idx)[0]
        t = evaluate(gt, cache, best.conf, best.nms, best["pad"], idx=idx)[0]
        print(f"  {site:8s} n={len(idx):3d}  PCK@0.2 {b['PCK@0.2']:.3f} -> {t['PCK@0.2']:.3f}   found {b['found']:.2f} -> {t['found']:.2f}")
    _, errs = evaluate(gt, cache, best.conf, best.nms, best["pad"])
    E = np.array(errs)
    print("\nPCK@0.2 per keypoint (best setting, matched apes only):")
    M = E[np.isfinite(E).any(1)]
    for j, k in enumerate(COMPARED):
        v = M[:, j][~np.isnan(M[:, j])]
        print(f"  {k:15s} {(v < 0.2).mean():.2f}  (n={len(v)})")
