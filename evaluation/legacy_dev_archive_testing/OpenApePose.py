"""
Lab 5 preparation, step 2: place ape keypoints inside the SAM2 boxes with OpenApePose.

    python OpenApePose.py               # every clip that has boxes (data/boxes/) but no pose yet
    python OpenApePose.py --redo        # all of them again

OpenApePose (Desai et al. 2023, eLife, https://doi.org/10.7554/eLife.86873) is a pose model trained on
photographs of apes, including gibbons. It is "top-down": it does not look for the animal itself, it places
16 keypoints inside a box you give it -- here, the box around the SAM2 mask from step 1.

The model file (models/hrnet_w48_oap_256x192_full.pth) was trained with MMPose 0.26. We do not need MMPose:
the network is a standard HRNet-W48, which we rebuild with timm and load the weights into. Then we follow the
model's own configuration:
    crop     the box, enlarged by PAD, widened to the model's 3:4 aspect and enlarged 1.25x, resized to 192x256
    input    RGB, scaled to 0-1, normalised with the ImageNet mean and standard deviation
    output   16 heatmaps of 48x64, one per keypoint: the peak is the keypoint, its height the confidence
    extras   the crop is also run mirrored ("flip test") and the two averaged; each peak is moved a quarter
             pixel towards its higher neighbour (the model's standard refinement)

Writes, per clip:
    data/pose/<name>.csv             one row per frame: frame, time_s, and for each of the 16 keypoints
                                     <keypoint>_x, <keypoint>_y (pixels of the 1920x1080 video) and
                                     <keypoint>_likelihood (confidence 0 to 1; -1 = no box in this frame)
    data/pose/<name>_labeled.mp4     keypoints and box on the video, coloured red (confidence 0) to green (1)
"""
import argparse
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
DATA = os.path.join(LAB, "data")
WEIGHTS = os.path.join(LAB, "models", "hrnet_w48_oap_256x192_full.pth")
PAD = 0.5          # enlarge each SAM2 box by 50% on every side. The mask hugs the body tightly (thin arms can
                   # fall outside it), and the model was trained on photos with the whole ape plus surroundings;
                   # on clip 14 this raised the frames with a confident hip from 23% to 81%

KEYPOINTS = ["nose", "left_eye", "right_eye", "head", "neck", "left_shoulder", "left_elbow", "left_wrist",
             "right_shoulder", "right_elbow", "right_wrist", "hip", "left_knee", "left_ankle", "right_knee", "right_ankle"]
FLIP = [KEYPOINTS.index(k.replace("left", "X").replace("right", "left").replace("X", "right")) for k in KEYPOINTS]
W, H = 192, 256                                      # the model's input size
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class OpenApePose(torch.nn.Module):
    """HRNet-W48 (from timm) + a 1x1 convolution that turns its high-resolution features into 16 heatmaps."""
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("hrnet_w48", pretrained=False, features_only=True, feature_location="")
        self.head = torch.nn.Conv2d(48, len(KEYPOINTS), 1)
        weights = torch.load(WEIGHTS, map_location="cpu", weights_only=False)["state_dict"]
        # strict=False: MMPose does not build the low-resolution outputs of the last block, which we do not use
        self.backbone.load_state_dict({k[len("backbone."):]: v for k, v in weights.items() if k.startswith("backbone.")},
                                      strict=False)
        self.head.load_state_dict({"weight": weights["keypoint_head.final_layer.weight"],
                                   "bias": weights["keypoint_head.final_layer.bias"]})

    def forward(self, x):
        return self.head(self.backbone(x)[1])        # [1] = the 48-channel branch at 1/4 of the input resolution


def crop_matrix(x, y, w, h):
    """Affine transform from the image to the 192x256 crop around the box (x, y, w, h)."""
    cx, cy = x + w / 2, y + h / 2
    w, h = (w, w * H / W) if w > h * W / H else (h * W / H, h)       # widen to the model's aspect (3:4)
    w, h = w * 1.25, h * 1.25                                         # and enlarge by 1.25
    src = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]])
    return cv2.getAffineTransform(src, np.float32([[0, 0], [W, 0], [0, H]]))


def peaks(hm):
    """The peak of each heatmap, moved a quarter pixel towards its higher neighbour; in crop pixels."""
    k, hh, ww = hm.shape
    idx = hm.reshape(k, -1).argmax(1)
    px, py, conf = (idx % ww).astype(float), (idx // ww).astype(float), hm.reshape(k, -1).max(1)
    for j in range(k):
        xi, yi = int(px[j]), int(py[j])
        if 0 < xi < ww - 1:
            px[j] += 0.25 * np.sign(hm[j, yi, xi + 1] - hm[j, yi, xi - 1])
        if 0 < yi < hh - 1:
            py[j] += 0.25 * np.sign(hm[j, yi + 1, xi] - hm[j, yi - 1, xi])
    return px * W / ww, py * H / hh, conf


def keypoints_in_box(model, frame, box, device):
    """16 rows of (x, y, confidence) in image pixels."""
    M = crop_matrix(*box)
    crop = (cv2.warpAffine(frame, M, (W, H))[:, :, ::-1] / 255.0 - MEAN) / STD      # OpenCV gives BGR -> RGB
    batch = torch.tensor(np.stack([crop, crop[:, ::-1]]).transpose(0, 3, 1, 2).copy(), dtype=torch.float32, device=device)
    with torch.no_grad():
        hm = model(batch).cpu().numpy()
    mirrored = hm[1][FLIP][:, :, ::-1]                                # undo the mirroring, swap left and right
    mirrored = np.concatenate([mirrored[:, :, :1], mirrored[:, :, :-1]], axis=2)     # the model's 1-pixel shift
    px, py, conf = peaks((hm[0] + mirrored) / 2)
    back = cv2.invertAffineTransform(M)                               # crop -> image
    return np.c_[back[0, 0] * px + back[0, 1] * py + back[0, 2], back[1, 0] * px + back[1, 1] * py + back[1, 2], conf]


def track_clip(model, name, fps, device):
    boxes = pd.read_csv(os.path.join(DATA, "boxes", f"{name}.csv"))
    cap = cv2.VideoCapture(os.path.join(DATA, "video", f"{name}.mp4"))
    data = np.full((len(boxes), len(KEYPOINTS), 3), [np.nan, np.nan, -1.0])
    silent = os.path.join(DATA, "pose", f"{name}_tmp.mp4")
    writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    for i, b in boxes.iterrows():
        ok, frame = cap.read()
        if not ok:
            break
        view = cv2.resize(frame, (960, 540))
        if b.source != "none":
            data[i] = keypoints_in_box(model, frame, (b.x - PAD * b.w, b.y - PAD * b.h, b.w * (1 + 2 * PAD), b.h * (1 + 2 * PAD)), device)
            cv2.rectangle(view, (int(b.x / 2), int(b.y / 2)), (int((b.x + b.w) / 2), int((b.y + b.h) / 2)), (0, 200, 0), 1)
            for x, y, c in data[i]:
                cv2.circle(view, (int(x / 2), int(y / 2)), 4, (0, int(255 * min(c, 1)), int(255 * (1 - min(c, 1)))), -1)
        cv2.putText(view, f"frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(view)
    cap.release()
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
                    os.path.join(DATA, "pose", f"{name}_labeled.mp4")], check=True)
    os.remove(silent)
    pose = pd.DataFrame({"frame": np.arange(len(boxes)), "time_s": np.arange(len(boxes)) / fps})
    for j, k in enumerate(KEYPOINTS):
        pose[f"{k}_x"], pose[f"{k}_y"], pose[f"{k}_likelihood"] = data[:, j, 0], data[:, j, 1], data[:, j, 2]
    pose.round(3).to_csv(os.path.join(DATA, "pose", f"{name}.csv"), index=False)
    conf = data[:, :, 2]
    print(f"  {name}: mean keypoint confidence {conf[conf >= 0].mean():.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true", help="redo clips that already have pose")
    args = ap.parse_args()
    os.makedirs(os.path.join(DATA, "pose"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OpenApePose().to(device).eval()
    clips = pd.read_csv(os.path.join(DATA, "clips.csv")).set_index("name")
    for f in sorted(os.listdir(os.path.join(DATA, "boxes"))):
        if not f.endswith(".csv"):
            continue
        name = f[:-4]
        if os.path.exists(os.path.join(DATA, "pose", f"{name}.csv")) and not args.redo:
            continue
        track_clip(model, name, clips.loc[name, "fps"], device)
