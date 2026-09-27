"""Video in/out through the ffmpeg binary that ships with imageio-ffmpeg (no system ffmpeg needed).

Reading: frames as decoded, interlaced frames deinterlaced (yadif, one frame per frame, so the frame rate is
unchanged). Writing: H.264 at the input frame rate, the input's audio copied in (re-encoded to AAC).
"""
import imageio_ffmpeg
import numpy as np


def read_video(path):
    """-> fps, (width, height), generator of BGR frames."""
    gen = imageio_ffmpeg.read_frames(path, pix_fmt="bgr24", output_params=["-vf", "yadif=deint=interlaced"])
    meta = next(gen)
    w, h = meta["size"]
    return meta["fps"], (w, h), (np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in gen)


class VideoWriter:
    """Write BGR frames to an H.264 mp4 at `fps`, copying the audio of `audio_from` (if it has any)."""

    def __init__(self, path, size, fps, audio_from=None, crf=23):
        w, h = size
        self.size = (w - w % 2, h - h % 2)                  # H.264/yuv420p needs even dimensions
        self.gen = imageio_ffmpeg.write_frames(
            path, self.size, pix_fmt_in="bgr24", fps=fps, codec="libx264", quality=None, macro_block_size=1,
            output_params=["-crf", str(crf), "-movflags", "+faststart"], audio_path=audio_from, audio_codec="aac" if audio_from else None,
            ffmpeg_log_level="error")
        self.gen.send(None)

    def write(self, frame):
        self.gen.send(np.ascontiguousarray(frame[: self.size[1], : self.size[0]]))

    def close(self):
        self.gen.close()
