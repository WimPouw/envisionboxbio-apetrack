"""
Inspection videos from the pipeline test results.

    python make_inspect_videos.py                        # results/mit_c (left) vs results/owlv2 (right)
    python make_inspect_videos.py --left mit_c --right mit_e
    python make_inspect_videos.py --left owlv2_smooth-low --right none   # one result folder, highlights reel only

For every clip labelled in BOTH result folders writes, to results/inspect_<left>_vs_<right>/:
    <name>.mp4          left and right labelled videos side by side, with the original audio
and one reel of the clips worth looking at first:
    highlights.mp4      multi-ape clips plus a few siamang clips, 30 fps, no audio
"""
import argparse
import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "..", "samples")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
HIGHLIGHTS = ["grooming2", "_14_", "_18_", "_4_", "_21_", "bossou12", "kalinzu9", "sonso2", "wamba12"]
REEL_FPS = 30


def put_label(img, text, y):
    (w, h), _ = cv2.getTextSize(text, 0, 0.8, 2)
    cv2.rectangle(img, (5, y - h - 8), (15 + w, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y), 0, 0.8, (255, 255, 255), 2)


def c_vs_e_frames(name):
    """Yield (fps, frame) with the c and e labelled frames side by side."""
    a = cv2.VideoCapture(os.path.join(C, f"{name}_labeled.mp4"))
    b = None if E is None else cv2.VideoCapture(os.path.join(E, f"{name}_labeled.mp4"))
    fps = a.get(cv2.CAP_PROP_FPS)
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read() if b is not None else (True, None)
        if not (ok1 and ok2):
            break
        put_label(f1, LEFT_NAME, f1.shape[0] - 15)
        if f2 is None:
            yield fps, f1
            continue
        if f2.shape != f1.shape:
            f2 = cv2.resize(f2, f1.shape[1::-1])
        put_label(f2, RIGHT_NAME, f2.shape[0] - 15)
        yield fps, np.hstack([f1, f2])
    a.release()
    if b is not None:
        b.release()


def encode(frames_iter, out, audio_from=None):
    tmp = out[:-4] + "_tmp.mp4"
    writer = None
    for fps, frame in frames_iter:
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
        writer.write(frame)
    writer.release()
    audio = ["-i", audio_from, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"] if audio_from else []
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", tmp, *audio, "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", out], check=True)
    os.remove(tmp)


def reel_frames(names):
    """The highlight clips one after another, resampled to REEL_FPS, with the clip name on top."""
    for name in names:
        frames = list(c_vs_e_frames(name))
        fps = frames[0][0]
        for t in np.arange(0, len(frames) / fps, 1 / REEL_FPS):
            frame = frames[min(int(t * fps), len(frames) - 1)][1].copy()
            put_label(frame, name, 60)
            yield REEL_FPS, frame


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="mit_c")
    ap.add_argument("--right", default="owlv2")
    args = ap.parse_args()
    LEFT_NAME, RIGHT_NAME = args.left, args.right
    C = os.path.join(HERE, "results", args.left)
    E = None if args.right == "none" else os.path.join(HERE, "results", args.right)
    OUT = os.path.join(HERE, "results", f"inspect_{args.left}" + ("" if E is None else f"_vs_{args.right}"))
    os.makedirs(OUT, exist_ok=True)
    names = sorted(f[:-len("_labeled.mp4")] for f in os.listdir(C)
                   if f.endswith("_labeled.mp4") and (E is None or os.path.exists(os.path.join(E, f))))
    for name in ([] if E is None else names):                           # side by side only when comparing
        source = next(f for f in os.listdir(SAMPLES) if os.path.splitext(f)[0] == name)
        encode(c_vs_e_frames(name), os.path.join(OUT, f"{name}.mp4"), audio_from=os.path.join(SAMPLES, source))
    print(f"{len(names)} side-by-side videos in {OUT}")

    reel = [n for key in HIGHLIGHTS for n in names if key in n]
    if reel:
        encode(reel_frames(reel), os.path.join(OUT, "highlights.mp4"))
        print("highlights.mp4:", ", ".join(reel))
