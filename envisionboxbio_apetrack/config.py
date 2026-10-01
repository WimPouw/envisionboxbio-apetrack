"""Default settings. Detection/tracking defaults were tuned on hand-checked siamang boxes (11 clips) and DeepWild (Wiltshire et al. 2023; 1577 hand-labelled chimpanzees and bonobos)."""
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Savitzky-Golay window in seconds (polynomial order 2). "high" lags fast movements such as brachiation.
SMOOTHING = {"none": 0.0, "low": 0.1, "medium": 0.25, "high": 0.5}

POSE_WEIGHTS_FILE = "openapepose_hrnet_w48_fp16.safetensors"
POSE_WEIGHTS_URL = "https://zenodo.org/records/22989835/files/openapepose_hrnet_w48_fp16.safetensors?download=1"
POSE_WEIGHTS_SHA256 = "9ac7636c3b086367a9728164537642c5d87020f1337d181e4ba779330afea8c7"


@dataclass
class Config:
    # --- detection: OWLv2 (Minderer et al. 2023), open-vocabulary, prompted with text ---
    detector_model: str = "google/owlv2-base-patch16-ensemble"
    prompts: Tuple[str, ...] = ("a photo of a chimpanzee", "a photo of a bonobo", "a photo of a gorilla",
                                "a photo of an orangutan", "a photo of a gibbon", "a photo of a siamang",
                                "a photo of an ape")
    detection_threshold: float = 0.15       # keep boxes scoring at least this
    nms_iou: float = 0.7                    # merge boxes that overlap more than this
    tiles: int = 1                          # >1: also detect on tiles x tiles enlarged crops (far-away apes; slower)
    detect_every: int = 1                   # detect on every n-th frame only (CPU); boxes in between are interpolated

    # --- tracking: ByteTrack (Zhang et al. 2022, via supervision) + stitching + gap filling ---
    track_activation_threshold: float = 0.25   # score needed to start a track
    track_buffer_s: float = 1.0                # how long a lost track is kept
    track_matching_threshold: float = 0.9      # 1 - IoU needed to link a box to a track (lenient: fast apes)
    stitch_gap_s: float = 2.0                  # re-join a track that reappears within this time (occlusion)
    stitch_distance: float = 1.0               # ... within this many ape sizes of where it disappeared
    fill_gap_s: float = 0.5                    # interpolate missing boxes within a track up to this long

    # --- smoothing ---
    box_smoothing: str = "low"                 # none | low | medium | high
    keypoint_smoothing: str = "low"            # none | low | medium | high

    # --- pose: OpenApePose (Desai et al. 2023) ---
    pad: float = 0.1                           # context around the box given to the pose model
    pose_weights: Optional[str] = None         # local .safetensors file; None = download on first use

    # --- output video ---
    draw: Tuple[str, ...] = ("points",)        # any of "points", "skeleton"; () = no video
    output_width: Optional[int] = None         # None = same size as the input
    min_likelihood_draw: float = 0.0           # hide keypoints/limbs below this likelihood
    device: Optional[str] = None               # None = cuda if available, else cpu

    extra: dict = field(default_factory=dict)
