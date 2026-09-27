"""
Composite GIF for the README: a grid of demo outputs (points, low smoothing), gibbons and chimpanzees/bonobos
alternating. Short clips loop. Needs the demo videos (tools/make_demo.py) and: pip install moviepy

    python tools/make_readme_gif.py

Writes images/demo_grid.gif (README) and a copy in demo/images/ (Quarto page).
"""
import os
import shutil

from moviepy import VideoFileClip, clips_array, vfx

# tweakables
FPS = 10              # gif frame rate
TILE_WIDTH = 320      # width of each video in the grid (px)
COLUMNS = 4
DURATION = 4.5        # seconds (6 s -> ~40 MB, 4.5 s -> ~30 MB)
STYLE, LEVEL = "points", "low"
SAMPLES = ["Gibbons_ChibaZoologicalPark_14_brachiating_vocalizing", "bossou12",
           "Gibbons_ChibaZoologicalPark_18_brachiating_vocalizing", "kalinzu10",
           "grooming2", "bossou28",
           "Gibbons_ChibaZoologicalPark_12_brachiating_vocalizing", "sonso2",
           "Gibbons_ChibaZoologicalPark_15_brachiating_vocalizing", "wamba12",
           "Gibbons_ChibaZoologicalPark_4_brachiating_vocalizing", "bossou30"]

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "images", "demo_grid.gif")


def tile(name):
    clip = VideoFileClip(os.path.join(ROOT, "demo", "output", LEVEL, f"{name}_{STYLE}.mp4"), audio=False)
    if clip.duration > DURATION + 2:                                  # long clip: start a quarter in
        start = min(clip.duration / 4, clip.duration - DURATION)
        clip = clip.subclipped(start, start + DURATION)
    else:                                                            # short clip: loop it
        clip = clip.with_effects([vfx.Loop(duration=DURATION)])
    return clip.resized(width=TILE_WIDTH)


clips = [tile(n) for n in SAMPLES]
grid = clips_array([clips[i:i + COLUMNS] for i in range(0, len(clips), COLUMNS)]).with_duration(DURATION)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
grid.write_gif(OUT, fps=FPS)
for c in clips:
    c.close()
os.makedirs(os.path.join(ROOT, "demo", "images"), exist_ok=True)
shutil.copy(OUT, os.path.join(ROOT, "demo", "images", "demo_grid.gif"))
print(f"{OUT}: {grid.w}x{grid.h}, {DURATION} s at {FPS} fps, {os.path.getsize(OUT) / 1e6:.1f} MB")
