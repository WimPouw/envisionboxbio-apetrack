# envisionboxbio-apetrack

Detection and pose estimation of apes in video without training or labeling (zero-shot).

Wim Pouw, Department of Computational Cognitive Science, Tilburg University

![envisionboxbio-apetrack on gibbons, chimpanzees and bonobos](images/demo_grid.gif)

[TODO: link to the demo page]

**Pipeline**: OWLv2 (boxes, prompted with ape names) -> ByteTrack (track IDs within a video) + stitching across
short occlusions -> Savitzky-Golay smoothing bounding boxes -> OpenApePose (16 keypoints per ape) -> Savitzky-Golay smoothing smoothing keypoints -> CSV + labelled video.

The current tool serves as an easy to use "out of the box" pose tracker for apes. While there are many easy 
to use deep learning frameworks to train one's own model with your own labeled data, to my surprise there
were no easy to use ape-specific tracking that work on any video. The available software, like OpenApePose (Desai et al., 2023) for keypoint detection,
were however there to be easily combined with powerful top-down bounding box detection models (Owlv2), which together with 
software for stable detection over sequences (ByteTrack), makes a promising pose estimation tool. 

OpenApePose (Desai, 2023) was trained on 71,868 annotated images:

- 18,010 chimpanzees (*Pan troglodytes*)
- 11,685 bonobos (*Pan paniscus*) 
- 12,905 gorillas (*Gorilla gorilla*)
- 12,722 orangutans (*Pongo sp.*)
- 9274 gibbons (genus *Hylobates* and *Nomascus*)
- 7272 siamangs (*Symphalangus syndactylus*)

Please reach out to w.pouw@tilburguniversity.edu if you want to help out with proper evaluation against state of the art (but usually less user-friendly) computer vision models.

## Citation

### Citing envisionboxbio-apetrack

- [forthcoming]

### Citing the components

This python package is a productive recombination of available and openly licensed software. Therefore first and foremost these packages need to be cited:

- **OWLv2**: Minderer, M., Gritsenko, A., & Houlsby, N. (2023). Scaling open-vocabulary object detection. *Advances in Neural Information Processing Systems, 36*, 72983–73007. <https://doi.org/10.52202/075280-3191>
- **ByteTrack**: Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., ... & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. In *European Conference on Computer Vision* (pp. 1–21). Springer Nature Switzerland. <https://doi.org/10.1007/978-3-031-20047-2_1>
- **OpenApePose**: Desai, N., Bala, P., Richardson, R., Raper, J., Zimmermann, J., & Hayden, B. (2023). OpenApePose, a database of annotated ape photographs for pose estimation. *eLife, 12*, RP86873. <https://doi.org/10.7554/eLife.86873.3>

### Data used for testing

Some of the chimpanzee samples used for testing were from the open-source data from:

- Wiltshire, C., Lewis‐Cheetham, J., Komedová, V., Matsuzawa, T., Graham, K. E., & Hobaiter, C. (2023). DeepWild: Application of the pose estimation tool DeepLabCut for behaviour tracking in wild chimpanzees and bonobos. *Journal of Animal Ecology, 92*(8), 1560–1574. <https://doi.org/10.1111/1365-2656.13932>

Siamang samples were from [YouTube](https://www.youtube.com/watch?v=MLMRHScRnhs) and recordings related to the following work:

- Pouw, W., Kehy, M., Gamba, M., & Ravignani, A. (2026). Amplitude increases of vocalizations are associated with body accelerations in siamang (*Symphalangus syndactylus*). *International Journal of Comparative Psychology, 39*. <https://doi.org/10.46867/ijcp.53165>

## Installation

Python 3.10–3.12. Install PyTorch (torch + torchvision, from the same index) first, then the package.

**GPU (NVIDIA, recommended)**
```bash
conda create -n apetrack python=3.11 -y
conda activate apetrack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install envisionboxbio-apetrack
```
Pick the CUDA build that matches your driver: https://pytorch.org/get-started/locally/. Pick the torchvision installation that matches your cuda version.

**CPU only**
```bash
conda create -n apetrack python=3.11 -y
conda activate apetrack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install envisionboxbio-apetrack
```

To increase user-friendly nature of this package, no system ffmpeg is needed, accounts or API keys are NOT needed. On first use the package downloads the OWLv2 detector
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
| `<name>_points.mp4` (default), `<name>_skeleton.mp4` | labelled video(s): keypoints as points or as a skeleton (`draw`) |
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
Full evaluation still needs to be performed. Distant apes or overlapping apes are not tracked well. 
It is limited to apes, monkeys will likely not be tracked well.

## Evaluation

- Planned (if you want to collaborate on this reach out to w.pouw@tilburguniversity.edu)

## OpenApe Pose model
We redistributed the trained OpenApePose model (HRNet-W48, file hrnet_w48_oap_256x192_full.pth) on Zenodo (<https://zenodo.org/records/22989835>), which was originally deposited on Dryad under public domain (CC0) (<https://doi.org/10.5061/dryad.c59zw3rds>) and here released under the MIT license in the repository (<https://github.com/desai-nisarg/OpenApePose>). The only change is a conversion to safetensors [fp16], keeping the backbone and keypoint head weights; the model was not retrained or modified. Please cite the original authors (Desai et al., 2023).

## Licence
MIT. Model weights: OWLv2 Apache-2.0; OpenApePose weights CC0 license.
