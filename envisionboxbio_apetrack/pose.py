"""
OpenApePose (Desai et al. 2023, eLife, https://doi.org/10.7554/eLife.86873): 16 keypoints inside a box.

HRNet-W48 (built with timm) + a 1x1 convolution that turns the high-resolution branch into 16 heatmaps.
Top-down: crop the box (enlarged by `pad`, widened to 3:4 and enlarged 1.25x), resize to 192x256, run it and its
mirror image ("flip test"), average, and take each heatmap's peak (moved a quarter pixel towards its higher
neighbour) as the keypoint and its height as the likelihood.
"""
import cv2
import numpy as np
import torch

KEYPOINTS = ["nose", "left_eye", "right_eye", "head", "neck", "left_shoulder", "left_elbow", "left_wrist",
             "right_shoulder", "right_elbow", "right_wrist", "hip", "left_knee", "left_ankle", "right_knee", "right_ankle"]
FLIP = [KEYPOINTS.index(k.replace("left", "X").replace("right", "left").replace("X", "right")) for k in KEYPOINTS]
W, H = 192, 256                                                      # model input size
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class OpenApePose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("hrnet_w48", pretrained=False, features_only=True, feature_location="")
        self.head = torch.nn.Conv2d(48, len(KEYPOINTS), 1)

    def forward(self, x):
        return self.head(self.backbone(x)[1])                        # [1] = 48-channel branch at 1/4 resolution


def load_model(weights_path, device):
    """The fp16 safetensors file holds the complete state dict of OpenApePose(): strict loading."""
    from safetensors.torch import load_file
    model = OpenApePose()
    model.load_state_dict({k: v.float() for k, v in load_file(weights_path).items()}, strict=True)
    return model.to(device).eval()


def crop_matrix(x, y, w, h):
    """Affine transform from the image to the 192x256 crop around box (x, y, w, h)."""
    cx, cy = x + w / 2, y + h / 2
    w, h = (w, w * H / W) if w > h * W / H else (h * W / H, h)       # widen to the model's aspect (3:4)
    w, h = w * 1.25, h * 1.25
    src = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx - w / 2, cy + h / 2]])
    return cv2.getAffineTransform(src, np.float32([[0, 0], [W, 0], [0, H]]))


def peaks(hm):
    """Peak of each heatmap, moved a quarter pixel towards its higher neighbour; in crop pixels."""
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


def keypoints_for_boxes(model, frame_bgr, boxes_xywh, pad, device):
    """All boxes of one frame in one batch (plus mirrored crops). Returns (n_boxes, 16, 3): x, y, likelihood."""
    if not len(boxes_xywh):
        return np.zeros((0, len(KEYPOINTS), 3))
    crops, Ms = [], []
    for x, y, w, h in boxes_xywh:
        M = crop_matrix(x - pad * w, y - pad * h, w * (1 + 2 * pad), h * (1 + 2 * pad))
        c = (cv2.warpAffine(frame_bgr, M, (W, H))[:, :, ::-1] / 255.0 - MEAN) / STD       # BGR -> RGB
        crops += [c, c[:, ::-1]]
        Ms.append(M)
    batch = torch.tensor(np.stack(crops).transpose(0, 3, 1, 2).copy(), dtype=torch.float32, device=device)
    with torch.no_grad():
        hm = model(batch).cpu().numpy()
    out = []
    for b, M in enumerate(Ms):
        mirrored = hm[2 * b + 1][FLIP][:, :, ::-1]                   # undo mirroring, swap left/right
        mirrored = np.concatenate([mirrored[:, :, :1], mirrored[:, :, :-1]], axis=2)      # 1-pixel shift
        px, py, conf = peaks((hm[2 * b] + mirrored) / 2)
        back = cv2.invertAffineTransform(M)
        out.append(np.c_[back[0, 0] * px + back[0, 1] * py + back[0, 2], back[1, 0] * px + back[1, 1] * py + back[1, 2], conf])
    return np.array(out)
