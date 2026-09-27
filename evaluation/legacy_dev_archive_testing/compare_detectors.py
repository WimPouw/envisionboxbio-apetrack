"""
Zero-shot detector comparison on a few frames per clip. Rows: methods, columns: frames.

    python compare_detectors.py        -> results/detectors/<clip>.jpg and timings

MD-c        MegaDetector v6 MIT yolov9-c, conf >= 0.2 (animal + person)
MD-c pad    same, on the frame padded to twice its size (close-up apes look smaller, like on a camera trap)
GDINO       Grounding DINO tiny, prompt "ape. gibbon. monkey."
OWLv2       OWLv2 base ensemble, prompts pipeline_test.OWLv2.PROMPTS (the great apes, gibbon, siamang, ape)
"""
import os, time
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
import pipeline_test as pt

HF = os.path.join(pt.MODELS, "hf")
OUT = os.path.join(pt.HERE, "results", "detectors"); os.makedirs(OUT, exist_ok=True)
CLIPS = ["00019_corde.MTS", "grooming2.MTS", "Gibbons_ChibaZoologicalPark_14_brachiating_vocalizing.mp4",
         "Gibbons_ChibaZoologicalPark_15_brachiating_vocalizing.mp4"]
dev = "cuda"
md = pt.Detector("MDV6-mit-yolov9-c", dev)
gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", cache_dir=HF)
gm = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny", cache_dir=HF).to(dev).eval()
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir=HF)
om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir=HF).to(dev).eval()
OWL_TEXT = pt.OWLv2.PROMPTS


def md_plain(fr):
    d = md(fr, 0.2); d = d[np.isin(d.class_id, [0, 1])]
    return d.xyxy, d.confidence


def md_pad(fr):
    h, w = fr.shape[:2]
    big = cv2.copyMakeBorder(fr, h // 2, h // 2, w // 2, w // 2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    d = md(big, 0.2); d = d[np.isin(d.class_id, [0, 1])]
    xyxy = d.xyxy - [w // 2, h // 2, w // 2, h // 2]
    return np.clip(xyxy, 0, [w, h, w, h]), d.confidence


@torch.no_grad()
def gdino(fr, th=0.3):
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    inp = gp(images=im, text="ape. gibbon. monkey.", return_tensors="pt").to(dev)
    r = gp.post_process_grounded_object_detection(gm(**inp), inp.input_ids, threshold=th, text_threshold=th,
                                                  target_sizes=[im.size[::-1]])[0]
    return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()


@torch.no_grad()
def owl(fr, th=0.2):
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    inp = op(text=OWL_TEXT, images=im, return_tensors="pt").to(dev)
    s = max(im.size)                                              # OWLv2 pads to a square
    r = op.post_process_grounded_object_detection(om(**inp), threshold=th, target_sizes=[(s, s)])[0]
    return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()


def nms(boxes, scores, iou=0.5):
    if not len(boxes): return boxes, scores
    keep = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in boxes], scores.tolist(), 0.0, iou)
    keep = np.array(keep).reshape(-1)
    return boxes[keep], scores[keep]


METHODS = [("MD-c", md_plain), ("MD-c pad", md_pad), ("GDINO", gdino), ("OWLv2", owl)]
times = {m: [] for m, _ in METHODS}
for clip in CLIPS:
    cap = cv2.VideoCapture(os.path.join(pt.VIDEO, clip)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(np.linspace(0.1 * n, 0.9 * n, 4).astype(int)); frames, i = [], 0
    while len(frames) < 4:
        ok, fr = cap.read()
        if not ok: break
        if i in want: frames.append((i, fr))
        i += 1
    rows = []
    for mname, fn in METHODS:
        tiles = []
        for fi, fr in frames:
            torch.cuda.synchronize(); t = time.time()
            b, s = fn(fr); b, s = nms(np.asarray(b), np.asarray(s))
            torch.cuda.synchronize(); times[mname].append(time.time() - t)
            v = fr.copy()
            for (x0, y0, x1, y1), c in zip(b, s):
                cv2.rectangle(v, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 6)
                cv2.putText(v, f"{c:.2f}", (int(x0) + 8, int(y0) + 55), 0, 2, (0, 255, 0), 5)
            cv2.putText(v, f"{mname}  frame {fi}", (20, 80), 0, 2.5, (255, 255, 255), 7)
            tiles.append(cv2.resize(v, (480, 270)))
        rows.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(OUT, os.path.splitext(clip)[0][:40] + ".jpg"), np.vstack(rows))
    print("done", clip, flush=True)
for m, t in times.items():
    print(f"{m:9s} {np.median(t[2:]) * 1000:.0f} ms/frame")
