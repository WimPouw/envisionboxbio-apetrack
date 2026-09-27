# README_developer: to do before/at release

## 0. Status and release order
Done:
- [x] Package code, tested in a clean env (GPU + CPU); tuned defaults; boxes + keypoints smoothing; skeleton/points videos
- [x] OpenApePose weights on Zenodo (https://zenodo.org/records/22989835); URL in `config.py`; first-use download
      + SHA-256 tested from an empty cache
- [x] OpenApePose licence settled: CC0 (Dryad), see README "OpenApe Pose model"
- [x] Citations of OWLv2, ByteTrack, OpenApePose printed on `import envisionboxbio_apetrack` (off: `APETRACK_QUIET=1`);
      text in `envisionboxbio_apetrack/citation.py` = README "Citation" (update both if one changes)
- [x] Demo: 35 samples x 4 smoothing levels x points/skeleton; Quarto page with intro, citations, licence (from
      README at render time) + widget + code dropdowns; README GIF (`images/demo_grid.gif`)

Fix before release (found in the file check):
- [ ] `CITATION.cff`: `cff-version: 0.1.0` -> `cff-version: 1.2.0` (the file-format version, not the package version)
- [ ] `CITATION.cff`: the line `date-released: YYYY-MM-DD   and   doi: <Zenodo DOI>` is not valid: comment it out or
      remove it until after the Zenodo release (an invalid CITATION.cff makes Zenodo's GitHub archiving fail)
- [ ] `setup.py`: `author_email` has a typo (`tilburgunivesity` -> `tilburguniversity`)
- [ ] README `[TODO: link to the demo page]`; Quarto page: GitHub URL of the README, "what to look for" TODO

Release order:
1. [ ] **Local pip install** of this folder, in a fresh env (torch + torchvision first, as in the README):
       ```bash
       conda create -n apetrack-local python=3.11 -y && conda activate apetrack-local
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
       pip install /mnt/data/Projects/envisionboxbio-apetrack        # or: pip install dist/envisionboxbio_apetrack-0.1.0-py3-none-any.whl
       python -c "from envisionboxbio_apetrack import ApeTracker; ApeTracker().process_video('some.mp4', 'out/')"
       ```
2. [ ] **Check all files** (the fixes above; `twine check dist/*`; `git status` shows no large/ignored files).
3. [ ] **GitHub**: `git init`, commit, push, make the repository public.
4. [ ] **Zenodo** (section 2 B + C): switch the repo on in Zenodo -> GitHub release `v0.1.0` -> concept DOI into
       `CITATION.cff` + README -> link the weights record ("is supplement to").
5. [ ] **PyPI** (section 4): GIF link absolute first; TestPyPI, then PyPI.
6. [ ] **Test** from PyPI in a fresh env (step 1 commands, but `pip install envisionboxbio-apetrack`).
7. [ ] **Quarto website** in a separate repository (`demo/_site/`, ~1.5 GB with videos) [you].

## 1. Licences (before publishing anything)
- [x] **OpenApePose weights**: CC0 (Dryad), see README. (Was: ask the authors (Desai et al. 2023, eLife 12:RP86873) for permission to redistribute
      the converted weights and under which licence. `models/MITLicense` in the old Lab 5 folder is an unfilled MIT
      template, so the licence is not clear yet. Fill it into README "Licence" and the Zenodo record.
- [ ] OWLv2 (`google/owlv2-base-patch16-ensemble`): Apache-2.0. Not redistributed (downloaded from Hugging Face).
- [ ] ByteTrack via `supervision` (MIT). Nothing to archive.
- [ ] Fill author list: `setup.py`, `CITATION.cff`, `LICENSE`.

## 2. Zenodo walkthrough: weights + code, both citable and linked
Release order: **A** weights record (its URL goes into the code) -> **B** GitHub release (Zenodo archives the code
automatically) -> **C** link the two -> PyPI (section 4).

Two Zenodo records:
- **weights**: `openapepose_hrnet_w48_fp16.safetensors` (131 MB), uploaded by hand. It can't go in git (GitHub file
  limit 100 MB) or on PyPI (package limit 100 MB); the package downloads it on first use with `pooch` and checks it
  against its SHA-256, so users need no account or API key.
- **code**: created automatically from each GitHub release. Its "concept DOI" (all versions) is the one to cite.

Dry run first (optional): https://sandbox.zenodo.org works the same but its DOIs are not real (separate account).

### A. Weights record (manual upload) — DONE: https://zenodo.org/records/22989835 (URL in config.py, download + SHA-256 tested)
1. Only if the file must be regenerated:
   `python tools/export_pose_weights.py path/to/hrnet_w48_oap_256x192_full.pth weights/openapepose_hrnet_w48_fp16.safetensors`
   (original .pth: https://surfdrive.surf.nl/s/bq5QEBqejXbxztX; the tool prints the SHA-256).
2. https://zenodo.org -> log in (ORCID or GitHub) -> **New upload** (top right).
3. **Files**: drag in `weights/openapepose_hrnet_w48_fp16.safetensors`.
4. **Digital Object Identifier**: "Do you already have a DOI?" -> **No** -> **Get a DOI now!** (reserves it).
5. **Resource type**: Dataset.
6. **Title**: [TODO, e.g. "OpenApePose HRNet-W48 weights (fp16 safetensors) for envisionboxbio-apetrack"].
7. **Creators**: [TODO: agree with the OpenApePose authors: them as creators, you as contributor for the conversion?].
8. **Description**: [TODO: converted from `hrnet_w48_oap_256x192_full.pth` with `tools/export_pose_weights.py`;
   SHA-256 `9ac7636c3b086367a9728164537642c5d87020f1337d181e4ba779330afea8c7`; cite Desai et al. 2023].
9. **Licence**: from section 1 (OpenApePose authors).
10. **Related works**: "Is derived from" -> `10.7554/eLife.86873` (OpenApePose paper). The link to the code
    record is added in C (it does not exist yet).
11. **Publish**. Files of a published record cannot be changed (only via "New version", which gets a new URL).
12. Copy the file URL: record page -> Files -> right-click the download button -> copy link. It looks like
    `https://zenodo.org/records/<id>/files/openapepose_hrnet_w48_fp16.safetensors?download=1`.
13. Paste it into `envisionboxbio_apetrack/config.py` as `POSE_WEIGHTS_URL` (replaces the TODO URL). Keep
    `POSE_WEIGHTS_SHA256` unless you regenerated the file (then use the new one printed in step 1).
14. Test the download like a new user would:
    ```bash
    python -c "import pooch; print(pooch.os_cache('envisionboxbio_apetrack'))"   # delete this folder if it exists
    unset APETRACK_POSE_WEIGHTS
    python -c "from envisionboxbio_apetrack import ApeTracker; ApeTracker().process_video('some.mp4', 'out/')"
    ```
    It must download, verify and run.
Until A is done: `ApeTracker(pose_weights="weights/openapepose_hrnet_w48_fp16.safetensors")` or
`export APETRACK_POSE_WEIGHTS=/path/to/file`.

### Quarto check


### B. Code record (automatic, from a GitHub release)
1. Before: repository public on GitHub; `CITATION.cff` valid (authors filled; Zenodo takes title, authors and
   licence from it, and an invalid file makes the archiving fail); the weights URL from A committed and pushed.
2. zenodo.org -> your name (top right) -> **GitHub** -> **Sync now** -> find `envisionboxbio-apetrack` -> switch **On**.
   This must happen *before* the release: earlier releases are not picked up.
3. GitHub -> repository -> **Releases** -> **Draft a new release** -> tag `v0.1.0` (same as `setup.py`) -> title ->
   **Publish release**.
4. After a few minutes the release appears on the Zenodo GitHub page with a DOI badge, and the record under
   **Uploads**. If it shows an error there: fix it (usually `CITATION.cff`) and make a new release (`v0.1.1`).
5. Open the record -> check title/authors/licence (edit metadata if needed). The **concept DOI** is under
   "Versions" ("Cite all versions? You can cite all versions by using the DOI ...").
6. In the repository: add `doi: <concept DOI>` and `date-released: YYYY-MM-DD` to `CITATION.cff`, the citation and a
   DOI badge to `README.md`; commit. (This lands in the next release; that is fine.)

### C. Link the records
1. Weights record -> **Edit** -> Related works -> "Is supplement to" -> the code concept DOI -> **Publish**
   (metadata edits keep the files and the URL).
2. `README.md` (Citation section): cite the code concept DOI and the weights DOI.

### Later versions
- Code: every new GitHub release -> new code version under the same concept DOI, automatically.
- Weights (only if they change): weights record -> **New version** -> upload -> publish -> new URL and SHA-256 into
  `config.py` -> then the GitHub release. Old package versions keep working (they point to the old file).

OWLv2 is downloaded anonymously from the Hugging Face Hub by `transformers` (no token). Optional: mirror it on
Zenodo too if you want the package independent of Hugging Face.

## 3. What stays out of git (`.gitignore`)
| ignored | why | where it lives |
|---|---|---|
| `build/`, `dist/`, `*.egg-info/`, `__pycache__/`, `*.pyc`, `*.log` | generated | `dist/` -> PyPI (section 4) |
| `weights/` | 131 MB | Zenodo weights record (section 2A) |
| `demo/samples/`, `demo/output/`, `demo/_site/`, `demo/.quarto/` | ~3.5 GB of video; sample permissions unclear | demo website host (section 5); regenerate with `tools/make_demo.py` |
| `evaluation/data/` | third-party data (DeepWild, Lab 5 boxes) | not redistributed; cite DeepWild |

## 4. PyPI (after section 2)
1. Check the name is free: https://pypi.org/project/envisionboxbio-apetrack/ (should 404).
2. Bump the version in `setup.py`, `envisionboxbio_apetrack/__init__.py`, `CITATION.cff`.
3. Build and check:
   ```bash
   pip install build twine
   rm -rf dist && python -m build
   twine check dist/*
   ```
4. Test upload first: `twine upload --repository testpypi dist/*`, then in a fresh env
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple envisionboxbio-apetrack`.
5. Real upload: `twine upload dist/*` (needs a PyPI account + API token in `~/.pypirc`; only you need it, users don't).
Torch is not pinned to a CUDA build on purpose: users install the GPU or CPU torch first (README), pip then keeps it.

## 5. Demo page (Quarto)
- Videos: `python tools/make_demo.py` (GPU; ~1.5 h for all samples). Samples in `demo/samples/` (each < 100 MB;
  grooming2 and wamba12 are cut copies, originals in the old LABEXPERIMENTING folder).
- Page: `cd demo && quarto render` (needs `pip install jupyter tabulate` in the env with the package).
  The settings table and code dropdowns are generated from the package source at render time.
- `demo/output/` and `demo/samples/` are git-ignored (hundreds of MB). Host the rendered `demo/_site/` (with videos)
  e.g. on the envisionBOX website server, not in the git repository. [TODO: decide where]
- Check the licences/permissions of the sample videos before publishing them (Chiba zoo, DeepWild sites, Beekse Bergen).

## 6. Evaluation (not in the package)
- `evaluation/data/` (git-ignored): `testdata_lab5/` (SAM2 boxes of one siamang in 11 Chiba clips, frame-aligned with
  demo/samples) and `testdata_wiltshire/` (DeepWild labels + images).
- `evaluation/legacy_dev/`: the scripts used during testing (`tune_boxes.py`, `tune_deepwild.py`, detector
  comparisons, `sam2_pipeline_test.py` = SAM2 route, on hold). They import the pre-package `pipeline_test.py` and
  expect the old folder layout. [TODO: port to the package API if the numbers should be reproducible from this repo.]
