# envisionboxbio-apetrack

Zero-shot detection, tracking and pose estimation of apes in video: no training, fine-tuning or labelling.

[TODO: authors, affiliations]

[TODO: demo gif / link to the demo page]

**Pipeline**: OWLv2 (boxes, prompted with ape names) → ByteTrack (track IDs within a video) + stitching across
short occlusions → Savitzky-Golay smoothing → OpenApePose (16 keypoints per ape) → smoothing → CSV + labelled video.

[TODO: short description / intended use]

## Installation

Python 3.10–3.12. Install PyTorch (torch + torchvision, from the same index) first, then the package.

**GPU (NVIDIA, recommended)**
```bash
conda create -n apetrack python=3.11 -y
conda activate apetrack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install envisionboxbio-apetrack
```
Pick the CUDA build that matches your driver: https://pytorch.org/get-started/locally/

**CPU only**
```bash
conda create -n apetrack python=3.11 -y
conda activate apetrack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install envisionboxbio-apetrack
```

No system ffmpeg, accounts or API keys are needed. On first use the package downloads the OWLv2 detector
(~0.6 GB, Hugging Face Hub) and the OpenApePose weights (~130 MB, Zenodo) into your cache folder.

## Usage

```python
from envisionboxbio_apetrack import ApeTracker

tracker = ApeTracker()                                    # default settings
tracker.process_video("clip.mp4", "output/")              # one video
tracker.process_folder("videos/", "output/")              # every video in a folder
```

Settings are passed as keywords (all in `Config`):

```python
tracker = ApeTracker(
    box_smoothing="medium",          # none | low | medium | high
    keypoint_smoothing="medium",     # none | low | medium | high
    draw=("skeleton", "points"),     # labelled videos to write; () = none
    output_width=960,                # None = same size as the input
)
```

### Outputs (per video)

| file | content |
|---|---|
| `<name>_tracks.csv` | one row per ape per frame: `frame`, `time_s`, `track_id`, `det_conf`, `interpolated`, `box_x`, `box_y`, `box_w`, `box_h` (smoothed; `raw_box_*` unsmoothed), and `<keypoint>_x`, `<keypoint>_y`, `<keypoint>_likelihood` for the 16 keypoints |
| `<name>_skeleton.mp4`, `<name>_points.mp4` | labelled video(s) |
| `<name>_detections.csv` | raw detections; reused when you process the video again with other tracking/smoothing settings |
| `apetrack_log.csv` | `process_folder` only: status, time and number of apes per video |

Coordinates are pixels of the input video (x to the right, y down; box x/y = top-left corner).
Keypoints: `nose, left_eye, right_eye, head, neck, left_shoulder, left_elbow, left_wrist, right_shoulder,
right_elbow, right_wrist, hip, left_knee, left_ankle, right_knee, right_ankle` (the ape's own left/right).

### How videos are transformed

| | output |
|---|---|
| frame rate | same as the input (25 fps in → 25 fps out); every input frame is one row/frame out |
| resolution | same as the input, unless `output_width` is set (CSV coordinates are always in input pixels) |
| interlaced video (e.g. camcorder .MTS) | deinterlaced, one frame per frame (frame rate unchanged) |
| audio | copied from the input into the labelled videos |
| codec | H.264 (.mp4) |

### Main settings

| setting | default | meaning |
|---|---|---|
| `detection_threshold` | 0.15 | minimum OWLv2 score |
| `prompts` | chimpanzee, bonobo, gorilla, orangutan, gibbon, siamang, ape | OWLv2 text prompts |
| `nms_iou` | 0.7 | merge boxes overlapping more than this |
| `tiles` | 1 | >1: also detect on enlarged tiles (small, far-away apes; ~tiles²+1 × slower) |
| `detect_every` | 1 | detect on every n-th frame; boxes in between interpolated (faster on CPU) |
| `track_activation_threshold` | 0.25 | score needed to start a new track |
| `track_matching_threshold` | 0.9 | ByteTrack matching (1 − IoU); lenient for fast movement |
| `stitch_gap_s`, `stitch_distance` | 2.0, 1.0 | re-join a track that reappears within 2 s, within 1 ape size |
| `fill_gap_s` | 0.5 | interpolate missing boxes within a track (`interpolated=True`) |
| `box_smoothing`, `keypoint_smoothing` | low | Savitzky-Golay (order 2) window: low 0.1 s, medium 0.25 s, high 0.5 s |
| `pad` | 0.1 | context around the box for OpenApePose |
| `min_likelihood_draw` | 0.0 | hide keypoints below this likelihood in the videos |

### Speed

Measured on a laptop (NVIDIA RTX 3500 Ada 12 GB; 22-core Intel CPU), 1080p video:

| | detection (OWLv2) | pose (OpenApePose, 2 apes) | whole pipeline |
|---|---|---|---|
| GPU | ~0.1 s/frame | ~0.02 s/frame | ~7 fps (incl. model loading and writing the video) |
| CPU | ~3.2 s/frame | ~0.4 s/frame | ~0.25 fps; ~1 fps with `detect_every=5` |

On CPU, use `detect_every=5` (or higher) and `draw=()` if you only need the CSV.

## Limitations

[TODO: discuss ID swaps during close contact; no identity across videos; small/far-away apes (< ~40 px);
limb keypoints on wild apes; monkeys]

## Evaluation

[TODO: summary of the tuning/evaluation (siamang pseudo-ground truth; DeepWild)]

## Citation

[TODO: cite this package]

Please also cite the models it uses:
- OWLv2: [TODO: Minderer, Gritsenko & Houlsby (2023). Scaling open-vocabulary object detection. NeurIPS.]
- ByteTrack: [TODO: Zhang et al. (2022). ByteTrack: multi-object tracking by associating every detection box. ECCV.]
- OpenApePose: [TODO: Desai et al. (2023). OpenApePose, a database of annotated ape photographs for pose estimation. eLife 12:RP86873.]

## Licence

Code: MIT. Model weights: OWLv2 Apache-2.0; OpenApePose [TODO].
