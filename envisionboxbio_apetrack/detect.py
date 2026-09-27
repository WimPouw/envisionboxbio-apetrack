"""Ape detection with OWLv2 (Minderer et al. 2023; Apache-2.0), an open-vocabulary detector: it finds whatever
the text prompts describe. Weights are downloaded from the Hugging Face Hub on first use (no account needed)."""
import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
from PIL import Image


class OWLv2Detector:
    def __init__(self, model_name, prompts, device, tiles=1, overlap=0.25):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        try:                                                         # cached: no network access at all
            self.proc = Owlv2Processor.from_pretrained(model_name, local_files_only=True)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name, local_files_only=True)
        except OSError:                                              # first use: one-time download (~0.6 GB)
            self.proc = Owlv2Processor.from_pretrained(model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.model = self.model.to(device).eval()
        self.prompts = [list(prompts)]
        self.device = device
        self.fp16 = device.startswith("cuda")                        # ~5x faster, same boxes (within 2 px)
        self.tiles, self.overlap = tiles, overlap

    def __call__(self, frame_bgr, threshold):
        """All boxes scoring >= threshold, overlapping boxes NOT merged: (xyxy array, scores)."""
        if self.tiles <= 1:
            return self._detect(frame_bgr, threshold)
        H, W = frame_bgr.shape[:2]
        n, ov = self.tiles, self.overlap
        tw, th = W / (n - (n - 1) * ov), H / (n - (n - 1) * ov)
        b0, s0 = self._detect(frame_bgr, threshold)
        boxes, scores = [b0], [s0]
        for iy in range(n):
            for ix in range(n):
                x0, y0 = int(ix * tw * (1 - ov)), int(iy * th * (1 - ov))
                b, s = self._detect(frame_bgr[y0:int(y0 + th), x0:int(x0 + tw)], threshold)
                boxes.append(b + [x0, y0, x0, y0])
                scores.append(s)
        return np.concatenate(boxes), np.concatenate(scores)

    def _detect(self, frame_bgr, threshold):
        im = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inp = self.proc(text=self.prompts, images=im, return_tensors="pt").to(self.device)
        side = max(im.size)                                          # OWLv2 pads the image to a square
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            out = self.model(**inp)
        out.logits, out.pred_boxes = out.logits.float(), out.pred_boxes.float()
        r = self.proc.post_process_grounded_object_detection(out, threshold=threshold, target_sizes=[(side, side)])[0]
        xyxy = np.clip(r["boxes"].cpu().numpy().astype(float), 0, [im.size[0], im.size[1]] * 2)
        return xyxy, r["scores"].cpu().numpy().astype(float)


def detect_video(detector, frames, fps, n_frames_hint=None, every=1, threshold=0.05, progress=None):
    """Raw detections for a whole video (saved, so tracking settings can be changed without detecting again).
    Columns: frame, conf, x0, y0, x1, y1 (pixels of the input video)."""
    rows, n = [], 0
    for i, frame in enumerate(frames):
        n = i + 1
        if progress is not None:
            progress.update(1)
        if i % every:
            continue
        xyxy, conf = detector(frame, threshold)
        rows += [{"frame": i, "conf": c, "x0": a, "y0": b, "x1": x, "y1": y} for (a, b, x, y), c in zip(xyxy, conf)]
    return pd.DataFrame(rows, columns=["frame", "conf", "x0", "y0", "x1", "y1"]).assign(fps=fps, n_frames=n, every=every)
