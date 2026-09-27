"""
AnimalRTPose boxes on the same frames as compare_detectors.py (runs in the animalrtpose-dev env: it needs the
authors' ultralytics fork, https://github.com/wux024/ultralytics, branch animalrtpose).

    python compare_animalrtpose.py      -> results/detectors/<clip>_animalrtpose.jpg and timings

Any of the 30 APT-36K species counts as an animal box; the predicted species is printed on the box.
"""
import os, time
import cv2, imageio_ffmpeg, numpy as np, torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "..", "models", "animalRTpose")
VIDEO = os.path.join(HERE, "..", "samples")
OUT = os.path.join(HERE, "results", "detectors")
CLIPS = ["00019_corde.MTS", "grooming2.MTS", "Gibbons_ChibaZoologicalPark_14_brachiating_vocalizing.mp4",
         "Gibbons_ChibaZoologicalPark_15_brachiating_vocalizing.mp4"]
SIZES = ["n", "s", "x"]
models = {s: YOLO(os.path.join(W, f"animalrtpose-{s}.pt")) for s in SIZES}
times = {s: [] for s in SIZES}
for clip in CLIPS:
    gen = imageio_ffmpeg.read_frames(os.path.join(VIDEO, clip), pix_fmt="bgr24", output_params=["-vf", "yadif=deint=interlaced"])
    meta = next(gen); w, h = meta["size"]
    allf = [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in gen]
    n = len(allf)
    frames = [(i, allf[i]) for i in np.linspace(0.1 * n, 0.9 * n, 4).astype(int)]     # same frames as compare_detectors
    rows = []
    for s, m in models.items():
        tiles = []
        for fi, fr in frames:
            torch.cuda.synchronize(); t = time.time()
            r = m.predict(fr, conf=0.2, verbose=False, device=0)[0]
            torch.cuda.synchronize(); times[s].append(time.time() - t)
            v = fr.copy()
            for (x0, y0, x1, y1), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                cv2.rectangle(v, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 6)
                cv2.putText(v, f"{c:.2f} {m.names[int(k)]}", (int(x0) + 8, int(y0) + 55), 0, 1.8, (0, 255, 0), 5)
            cv2.putText(v, f"ARTP-{s}  frame {fi}", (20, 80), 0, 2.5, (255, 255, 255), 7)
            tiles.append(cv2.resize(v, (480, 270)))
        rows.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(OUT, os.path.splitext(clip)[0][:40] + "_animalrtpose.jpg"), np.vstack(rows))
    print("done", clip, flush=True)
for s, t in times.items():
    print(f"ARTP-{s} {np.median(t[2:]) * 1000:.0f} ms/frame")
