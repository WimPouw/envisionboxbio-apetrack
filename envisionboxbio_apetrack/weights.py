"""Where the OpenApePose weights come from: an explicit file, the APETRACK_POSE_WEIGHTS environment variable, or
a one-time download (checked against its SHA-256) into the user's cache folder."""
import os

from .config import POSE_WEIGHTS_FILE, POSE_WEIGHTS_SHA256, POSE_WEIGHTS_URL


def pose_weights_path(explicit=None):
    if explicit:
        return explicit
    if os.environ.get("APETRACK_POSE_WEIGHTS"):
        return os.environ["APETRACK_POSE_WEIGHTS"]
    if "TODO" in POSE_WEIGHTS_URL:
        raise RuntimeError("The OpenApePose weights are not published yet. Pass Config(pose_weights='path/to/"
                           f"{POSE_WEIGHTS_FILE}') or set the APETRACK_POSE_WEIGHTS environment variable.")
    import pooch
    return pooch.retrieve(POSE_WEIGHTS_URL, known_hash=f"sha256:{POSE_WEIGHTS_SHA256}", fname=POSE_WEIGHTS_FILE,
                          path=pooch.os_cache("envisionboxbio_apetrack"), progressbar=True)
