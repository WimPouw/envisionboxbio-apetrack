"""
OpenApePose (Desai et al. 2023, eLife, https://doi.org/10.7554/eLife.86873): 16 keypoints inside a box.

HRNet-W48 (built with timm) + a 1x1 convolution that turns the high-resolution branch into 16 heatmaps.
Top-down: crop the box (enlarged by `pad`, widened to 3:4 and enlarged 1.25x), resize to 192x256, run it and its
mirror image ("flip test"), average, and take each heatmap's peak (moved a quarter pixel towards its higher
neighbour) as the keypoint and its height as the likelihood.

ViTPose++ (Xu et al. 2023, IEEE TPAMI; Hugging Face usyd-community/vitpose-plus-base,
AP-10K expert) goes through the same steps: same input size, normalisation and heatmap size, 17 COCO keypoints that
coco_to_keypoints maps to the 16 above (neck and hip are midpoints; it has no head point). `ensemble` combines the two.
"""
import cv2
import numpy as np
import torch

KEYPOINTS = ["nose", "left_eye", "right_eye", "head", "neck", "left_shoulder", "left_elbow", "left_wrist",
             "right_shoulder", "right_elbow", "right_wrist", "hip", "left_knee", "left_ankle", "right_knee", "right_ankle"]
FLIP = [KEYPOINTS.index(k.replace("left", "X").replace("right", "left").replace("X", "right")) for k in KEYPOINTS]
W, H = 192, 256                                                      # model input size
HM_W, HM_H = W // 4, H // 4 # heatmap size (1/4 of the input)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
# ViTPose++'s keypoints (COCO order) and the left/right pairs of both
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
        "right_ankle"]
COCO_FLIP = [COCO.index(k.replace("left", "X").replace("right", "left").replace("X", "right")) for k in COCO]
LR_PAIRS = [(KEYPOINTS.index("left_" + p), KEYPOINTS.index("right_" + p))
            for p in ("eye", "shoulder", "elbow", "wrist", "knee", "ankle")]
# the ensemble takes a limb from one model when the two disagree on it
LIMBS = [[KEYPOINTS.index(k) for k in limb] for limb in (
    ("left_shoulder", "left_elbow", "left_wrist"), ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle"), ("nose", "left_eye", "right_eye"), ("neck", "hip"))]


class OpenApePose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("hrnet_w48", pretrained=False, features_only=True, feature_location="")
        self.head = torch.nn.Conv2d(48, len(KEYPOINTS), 1)

    def forward(self, x):
        return self.head(self.backbone(x)[1])                        # [1] = 48-channel branch at 1/4 resolution


class ViTPose(torch.nn.Module):
    """ViTPose++ base with its AP-10K expert (dataset_index 3: best on the validation set), returning heatmaps like
    OpenApePose. Its crops get no extra padding (as its own processor)."""
    flip = COCO_FLIP
    pad = 0.0

    def __init__(self, hub="usyd-community/vitpose-plus-base", dataset_index=3):
        super().__init__()
        from transformers import VitPoseForPoseEstimation
        self.model = VitPoseForPoseEstimation.from_pretrained(hub)
        self.dataset_index = dataset_index

    def forward(self, x):
        idx = torch.full((len(x),), self.dataset_index, device=x.device)
        return self.model(pixel_values=x, dataset_index=idx).heatmaps


def coco_to_keypoints(k):
    """(n, 17, 3) COCO keypoints -> (n, 16, 3) in KEYPOINTS order: neck = shoulder midpoint, hip = hip midpoint,
    head = NaN (not drawn)."""
    out = np.full((len(k), len(KEYPOINTS), 3), np.nan)
    for j, name in enumerate(KEYPOINTS):
        if name == "neck":
            out[:, j] = (k[:, COCO.index("left_shoulder")] + k[:, COCO.index("right_shoulder")]) / 2
        elif name == "hip":
            out[:, j] = (k[:, COCO.index("left_hip")] + k[:, COCO.index("right_hip")]) / 2
        elif name in COCO:
            out[:, j] = k[:, COCO.index(name)]
    return out


def ensemble(a, v, size, max_disagreement=0.3):
    """OpenApePose (a) and ViTPose++ (v) keypoints of the same boxes, (n, 16, 3) each; size: (n,) ape sizes in pixels
    (square root of the box area). ViTPose++ decides left and right (OpenApePose's training data often labels body
    left/right by image side): per left/right pair, a's two points are swapped when that brings them closer to v's.
    Per limb (each arm, each leg, the face, neck + hip): where the two agree (every point within max_disagreement ape
    sizes), the likelihood-weighted mean; where they do not, the whole limb from the model more confident on it, so a
    point never ends up between two different guesses (e.g. a gibbon's wrist in mid-body). Head: OpenApePose's."""
    a = a.copy()
    for l, r in LR_PAIRS:
        keep = np.linalg.norm(a[:, l, :2] - v[:, l, :2], axis=1) + np.linalg.norm(a[:, r, :2] - v[:, r, :2], axis=1)
        swap = np.linalg.norm(a[:, l, :2] - v[:, r, :2], axis=1) + np.linalg.norm(a[:, r, :2] - v[:, l, :2], axis=1)
        s = swap < keep
        a[s, l], a[s, r] = a[s, r].copy(), a[s, l].copy()
    wa = np.clip(a[..., 2], 1e-3, None)
    wv = np.where(np.isnan(v[..., 0]), 0, np.clip(np.nan_to_num(v[..., 2]), 1e-3, None))
    out = a.copy()
    out[..., :2] = (wa[..., None] * a[..., :2] + wv[..., None] * np.nan_to_num(v[..., :2])) / (wa + wv)[..., None]
    out[..., 2] = (wa * a[..., 2] + wv * np.nan_to_num(v[..., 2])) / (wa + wv)
    far = np.linalg.norm(a[..., :2] - v[..., :2], axis=2) > max_disagreement * np.asarray(size)[:, None]   # NaN: no
    for limb in LIMBS:
        bad = far[:, limb].any(1)
        take_a = bad & (a[:, limb, 2].mean(1) >= v[:, limb, 2].mean(1))
        take_v = bad & ~take_a
        for k in limb:
            out[take_a, k], out[take_v, k] = a[take_a, k], v[take_v, k]
    return out


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


def peaks(hm, dark=True):
    """Peak of each heatmap in crop pixels, refined to sub-pixel precision. dark=True: DARK (Zhang et al. 2020):
    blur the heatmap, take its log, one Newton step (second-order Taylor expansion) from the integer maximum.
    dark=False: moved a quarter pixel towards the higher neighbour. Likelihood = height of the unblurred peak."""
    k, hh, ww = hm.shape
    idx = hm.reshape(k, -1).argmax(1)
    px, py, conf = (idx % ww).astype(float), (idx // ww).astype(float), hm.reshape(k, -1).max(1)
    for j in range(k):
        xi, yi = int(px[j]), int(py[j])
        if dark:
            if not (1 < xi < ww - 2 and 1 < yi < hh - 2):
                continue
            b = cv2.GaussianBlur(hm[j].astype(np.float32), (11, 11), 0)
            b = np.log(np.maximum(b * (hm[j].max() / max(b.max(), 1e-10)), 1e-10))
            dx, dy = 0.5 * (b[yi, xi + 1] - b[yi, xi - 1]), 0.5 * (b[yi + 1, xi] - b[yi - 1, xi])
            dxx = 0.25 * (b[yi, xi + 2] - 2 * b[yi, xi] + b[yi, xi - 2])
            dyy = 0.25 * (b[yi + 2, xi] - 2 * b[yi, xi] + b[yi - 2, xi])
            dxy = 0.25 * (b[yi + 1, xi + 1] - b[yi - 1, xi + 1] - b[yi + 1, xi - 1] + b[yi - 1, xi - 1])
            det = dxx * dyy - dxy ** 2
            if det > 1e-12 and dxx < 0:                             # a proper maximum: take the Newton step
                ox, oy = -(dyy * dx - dxy * dy) / det, -(dxx * dy - dxy * dx) / det
                px[j] += np.clip(ox, -1, 1)
                py[j] += np.clip(oy, -1, 1)
        else:
            if 0 < xi < ww - 1:
                px[j] += 0.25 * np.sign(hm[j, yi, xi + 1] - hm[j, yi, xi - 1])
            if 0 < yi < hh - 1:
                py[j] += 0.25 * np.sign(hm[j, yi + 1, xi] - hm[j, yi - 1, xi])
    return px * W / ww, py * H / hh, conf


# ---------------------------------------------------------------- 2. optical-flow heatmap fusion
class FlowFusion:
    """Per track: fused heatmaps of the previous frame, warped into the current crop (crop geometry + DIS optical
    flow between the two frames inside that crop), averaged with the current heatmaps: fused = a*cur + (1-a)*prev.
    Starts again whenever a track misses a frame."""

    def __init__(self, alpha=0.5):
        self.alpha, self.state = alpha, {}
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        S = np.diag([0.25, 0.25, 1.0])
        self.S, self.Si = S, np.linalg.inv(S)
        gx, gy = np.meshgrid(np.arange(HM_W, dtype=np.float32), np.arange(HM_H, dtype=np.float32))
        self.grid = gx, gy

    def __call__(self, frame_idx, frame, tids, Ms, hms):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = []
        for tid, M, hm in zip(tids, Ms, hms):
            prev = self.state.get(tid)
            if prev is not None and prev[0] == frame_idx - 1:
                _, pM, phm, pgray = prev
                # previous heatmaps -> current crop, through the image (heatmap pixels are crop pixels / 4)
                A = np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(np.vstack([pM, [0, 0, 1]]))
                A = (self.S @ A @ self.Si)[:2]
                aligned = np.stack([cv2.warpAffine(h, A, (HM_W, HM_H)) for h in phm])
                # motion inside the crop: flow from the current to the previous frame, both cropped with M
                c_cur = cv2.warpAffine(gray, M, (W, H))
                c_prev = cv2.warpAffine(pgray, M, (W, H))
                flow = self.dis.calc(c_cur, c_prev, None)
                f = cv2.resize(flow, (HM_W, HM_H), interpolation=cv2.INTER_AREA) / 4
                mx, my = self.grid[0] + f[..., 0], self.grid[1] + f[..., 1]
                warped = np.stack([cv2.remap(h, mx, my, cv2.INTER_LINEAR) for h in aligned])
                hm = self.alpha * hm + (1 - self.alpha) * warped
            self.state[tid] = (frame_idx, M, hm, gray)
            out.append(hm)
        return out

def keypoints_for_boxes(model, frame_bgr, boxes_xywh, pad, device, fusion=None, frame_idx=None, track_ids=None, dark=True):
    """All boxes of one frame in one batch (plus mirrored crops). With `fusion` (a FlowFusion), each track's
    heatmaps are fused with its previous frame's before the peaks are taken. Returns (n_boxes, n_keypoints, 3):
    16 (OpenApePose) or 17 (ViTPose, COCO order)."""
    flip = getattr(model, "flip", FLIP)
    if not len(boxes_xywh):
        return np.zeros((0, len(flip), 3))
    crops, Ms = [], []
    for x, y, w, h in boxes_xywh:
        M = crop_matrix(x - pad * w, y - pad * h, w * (1 + 2 * pad), h * (1 + 2 * pad))
        c = (cv2.warpAffine(frame_bgr, M, (W, H))[:, :, ::-1] / 255.0 - MEAN) / STD       # BGR -> RGB
        crops += [c, c[:, ::-1]]
        Ms.append(M)
    batch = torch.tensor(np.stack(crops).transpose(0, 3, 1, 2).copy(), dtype=torch.float32, device=device)
    with torch.no_grad():
        hm = model(batch).cpu().numpy()
    hms = []
    for b in range(len(Ms)):
        mirrored = hm[2 * b + 1][flip][:, :, ::-1]                   # undo mirroring, swap left/right
        mirrored = np.concatenate([mirrored[:, :, :1], mirrored[:, :, :-1]], axis=2)      # 1-pixel shift
        hms.append((hm[2 * b] + mirrored) / 2)
    if fusion is not None:
        hms = fusion(frame_idx, frame_bgr, track_ids, Ms, hms)
    out = []
    for h_, M in zip(hms, Ms):
        px, py, conf = peaks(h_, dark)
        back = cv2.invertAffineTransform(M)
        out.append(np.c_[back[0, 0] * px + back[0, 1] * py + back[0, 2], back[1, 0] * px + back[1, 1] * py + back[1, 2], conf])
    return np.array(out)


