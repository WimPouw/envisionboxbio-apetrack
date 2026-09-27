"""ApeTracker: video (or folder of videos) in -> per video a tracks CSV and labelled video(s) out."""
import dataclasses
import os
import subprocess
import time
import traceback

import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import detect, pose, render, smooth, track, video
from .config import SMOOTHING, Config
from .weights import pose_weights_path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mts", ".m2ts", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v")
CITE = ("envisionboxbio-apetrack uses OWLv2 (Minderer et al. 2023), ByteTrack (Zhang et al. 2022) and OpenApePose "
        "(Desai et al. 2023). Please cite them; see the README.")


def _has_audio(path):
    err = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    return "Audio:" in err


class ApeTracker:
    """
    tracker = ApeTracker()                                  # defaults (see Config)
    tracker = ApeTracker(box_smoothing="high", draw=("skeleton", "points"))
    tracks = tracker.process_video("clip.mp4", "output/")
    summary = tracker.process_folder("videos/", "output/")
    """

    def __init__(self, config=None, **settings):
        self.config = dataclasses.replace(config or Config(), **settings)
        for level in (self.config.box_smoothing, self.config.keypoint_smoothing):
            if level not in SMOOTHING:
                raise ValueError(f"smoothing must be one of {list(SMOOTHING)}, not {level!r}")
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._detector = self._pose = None
        print(CITE)

    # models are loaded on first use, so re-rendering from saved detections needs no detector
    @property
    def detector(self):
        if self._detector is None:
            c = self.config
            self._detector = detect.OWLv2Detector(c.detector_model, c.prompts, self.device, tiles=c.tiles)
        return self._detector

    @property
    def pose_model(self):
        if self._pose is None:
            self._pose = pose.load_model(pose_weights_path(self.config.pose_weights), self.device)
        return self._pose

    def process_video(self, video_path, output_folder, reuse_detections=True):
        """Writes <name>_detections.csv (raw detections, reused next time), <name>_tracks.csv and
        <name>_<style>.mp4 per drawing style. Returns the tracks table."""
        c = self.config
        os.makedirs(output_folder, exist_ok=True)
        name = os.path.splitext(os.path.basename(video_path))[0]
        det_path = os.path.join(output_folder, f"{name}_detections.csv")

        # 1. detect (slow; saved)
        if reuse_detections and os.path.exists(det_path) and _same_detection_settings(det_path, c):
            dets = pd.read_csv(det_path)
            fps = float(dets.fps.iloc[0]) if len(dets) else video.read_video(video_path)[0]
        else:
            fps, size, frames = video.read_video(video_path)
            with tqdm(desc=f"{name}: detecting", unit="frame", leave=False) as bar:
                dets = detect.detect_video(self.detector, frames, fps, every=c.detect_every,
                                           threshold=min(0.05, c.detection_threshold), progress=bar)
            dets.assign(tiles=c.tiles).round(3).to_csv(det_path, index=False)

        # 2. track, stitch, fill gaps, smooth boxes
        tracks = track.bytetrack(dets, fps, c.detection_threshold, c.nms_iou, c.track_activation_threshold,
                                 c.track_buffer_s, c.track_matching_threshold)
        tracks = track.stitch_tracks(tracks, fps, c.stitch_gap_s, c.stitch_distance)
        tracks = track.fill_gaps(tracks, fps, max(c.fill_gap_s, c.stitch_gap_s, c.detect_every / fps))
        tracks = smooth.smooth_boxes(tracks, fps, c.box_smoothing)

        # 3. keypoints in every box, then smooth them
        kp = np.zeros((len(tracks), len(pose.KEYPOINTS), 3))
        rows_of = tracks.groupby("frame").indices
        fps, size, frames = video.read_video(video_path)
        for i, frame in enumerate(tqdm(frames, desc=f"{name}: pose", unit="frame", leave=False)):
            idx = rows_of.get(i)
            if idx is not None:
                kp[idx] = pose.keypoints_for_boxes(self.pose_model, frame, tracks.iloc[idx][track.BOX_COLS].to_numpy(),
                                                   c.pad, self.device)
        for k, name_k in enumerate(pose.KEYPOINTS):
            tracks[f"{name_k}_x"], tracks[f"{name_k}_y"], tracks[f"{name_k}_likelihood"] = kp[:, k, 0], kp[:, k, 1], kp[:, k, 2]
        tracks = smooth.smooth_keypoints(tracks, pose.KEYPOINTS, fps, c.keypoint_smoothing)
        tracks = tracks.sort_values(["frame", "track_id"]).reset_index(drop=True)
        tracks.round(3).to_csv(os.path.join(output_folder, f"{name}_tracks.csv"), index=False)

        # 4. labelled video(s): same frame rate as the input, input audio copied
        if c.draw:
            self._render(video_path, output_folder, name, tracks)
        return tracks

    def _render(self, video_path, output_folder, name, tracks):
        c = self.config
        fps, (w, h), frames = video.read_video(video_path)
        scale = (c.output_width / w) if c.output_width else 1.0
        out_size = (int(round(w * scale)), int(round(h * scale)))
        audio = video_path if _has_audio(video_path) else None
        writers = {s: video.VideoWriter(os.path.join(output_folder, f"{name}_{s}.mp4"), out_size, fps, audio_from=audio)
                   for s in c.draw}
        rows_of = tracks.groupby("frame").indices
        kp_cols = [f"{k}_{a}" for k in pose.KEYPOINTS for a in ("x", "y", "likelihood")]
        try:
            for i, frame in enumerate(tqdm(frames, desc=f"{name}: video", unit="frame", leave=False)):
                idx = rows_of.get(i, [])
                sub = tracks.iloc[idx]
                kp = sub[kp_cols].to_numpy().reshape(len(sub), len(pose.KEYPOINTS), 3)
                recs = sub.to_dict("records")
                for s, wr in writers.items():
                    wr.write(render.draw_frame(frame, recs, kp, s, scale, c.min_likelihood_draw, frame_index=i))
        finally:
            for wr in writers.values():
                wr.close()

    def process_folder(self, input_folder, output_folder, skip_existing=True, reuse_detections=True):
        """Every video in input_folder (not recursive). Skips videos that already have a tracks CSV (unless
        skip_existing=False), keeps going when one fails, and writes apetrack_log.csv."""
        os.makedirs(output_folder, exist_ok=True)
        files = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(VIDEO_EXTENSIONS))
        log = []
        for f in tqdm(files, desc="videos", unit="video"):
            name = os.path.splitext(f)[0]
            if skip_existing and os.path.exists(os.path.join(output_folder, f"{name}_tracks.csv")):
                log.append({"video": f, "status": "skipped (done before)"})
                continue
            t0 = time.time()
            try:
                tr = self.process_video(os.path.join(input_folder, f), output_folder, reuse_detections)
                log.append({"video": f, "status": "ok", "seconds": round(time.time() - t0, 1),
                            "apes_tracked": tr.track_id.nunique() if len(tr) else 0,
                            "frames_with_ape": tr.frame.nunique() if len(tr) else 0})
            except Exception as e:                                           # log and continue with the next video
                log.append({"video": f, "status": f"failed: {e}", "error": traceback.format_exc()})
        log = pd.DataFrame(log)
        log.to_csv(os.path.join(output_folder, "apetrack_log.csv"), index=False)
        return log


def _same_detection_settings(det_path, c):
    d = pd.read_csv(det_path, nrows=1)
    if not len(d):
        return True
    return int(d.get("every", pd.Series([1])).iloc[0]) == c.detect_every and int(d.get("tiles", pd.Series([1])).iloc[0]) == c.tiles
