"""
Developer tool: convert the original OpenApePose checkpoint (MMPose, hrnet_w48_oap_256x192_full.pth, ~765 MB)
into the file the package downloads (openapepose_hrnet_w48_fp16.safetensors, ~130 MB).

    python tools/export_pose_weights.py path/to/hrnet_w48_oap_256x192_full.pth

- loads backbone + head into envisionboxbio_apetrack.pose.OpenApePose and checks that the only keys the
  checkpoint lacks are timm's low-resolution fuse layers of the last stage (MMPose does not build them, the
  model does not use them)
- saves the complete state dict in fp16 (so the package can load it with strict=True) and prints its SHA-256
- checks that the fp16 model gives the same heatmaps as the original (fp32) weights
"""
import hashlib
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from envisionboxbio_apetrack.pose import OpenApePose

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "openapepose_hrnet_w48_fp16.safetensors"
sd = torch.load(src, map_location="cpu", weights_only=False)["state_dict"]
model = OpenApePose().eval()
res = model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=False)
model.head.load_state_dict({"weight": sd["keypoint_head.final_layer.weight"], "bias": sd["keypoint_head.final_layer.bias"]})
unexpected = res.unexpected_keys
bad_missing = [k for k in res.missing_keys if not k.startswith("stage4.2.fuse_layers.") or k.startswith("stage4.2.fuse_layers.0.")]
print(f"checkpoint keys used: {sum(k.startswith('backbone.') for k in sd)} backbone + 2 head | "
      f"missing in checkpoint: {len(res.missing_keys)} (all low-res fuse layers: {not bad_missing}) | unexpected: {len(unexpected)}")
assert not unexpected and not bad_missing, (unexpected, bad_missing)

save_file({k: v.half().contiguous() for k, v in model.state_dict().items()}, out)
sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"saved {out} ({os.path.getsize(out) / 1e6:.0f} MB)\nsha256 {sha}")

from envisionboxbio_apetrack.pose import load_model
fp16 = load_model(out, "cpu")
x = torch.randn(2, 3, 256, 192)
with torch.no_grad():
    a, b = model(x), fp16(x)
print(f"max heatmap difference fp32 vs fp16 weights: {(a - b).abs().max():.2e} (heatmap range {a.min():.2f}..{a.max():.2f})")
