# README_developer: to do before/at release

## 1. Licences (before publishing anything)
- [ ] **OpenApePose weights**: ask the authors (Desai et al. 2023, eLife 12:RP86873) for permission to redistribute
      the converted weights and under which licence. `models/MITLicense` in the old Lab 5 folder is an unfilled MIT
      template, so the licence is not clear yet. Fill it into README "Licence" and the Zenodo record.
- [ ] OWLv2 (`google/owlv2-base-patch16-ensemble`): Apache-2.0. Not redistributed (downloaded from Hugging Face).
- [ ] ByteTrack via `supervision` (MIT). Nothing to archive.
- [ ] DeepWild (evaluation data): check its licence before sharing any derived evaluation files.
- [ ] Fill author list: `setup.py`, `CITATION.cff`, `LICENSE`.

## 2. Archive the pose weights on Zenodo (so `pip install` needs no account or API key)
The package downloads `openapepose_hrnet_w48_fp16.safetensors` (131 MB) on first use with `pooch`, checked against
its SHA-256. PyPI cannot hold it (100 MB file limit), and Hugging Face would need no token for a public file but
Zenodo gives a DOI.
1. Regenerate (only if needed): `python tools/export_pose_weights.py path/to/hrnet_w48_oap_256x192_full.pth weights/openapepose_hrnet_w48_fp16.safetensors`
   (original .pth: https://surfdrive.surf.nl/s/bq5QEBqejXbxztX; the tool prints the SHA-256).
2. Upload `weights/openapepose_hrnet_w48_fp16.safetensors` to a new Zenodo record (licence from step 1; cite
   Desai et al. 2023). Publish -> get the file URL `https://zenodo.org/records/<id>/files/openapepose_hrnet_w48_fp16.safetensors`.
3. Put that URL in `envisionboxbio_apetrack/config.py` (`POSE_WEIGHTS_URL`, replaces `TODO`). Keep
   `POSE_WEIGHTS_SHA256` as printed (current file: `9ac7636c3b086367a9728164537642c5d87020f1337d181e4ba779330afea8c7`).
4. Test: delete the cache folder (`python -c "import pooch; print(pooch.os_cache('envisionboxbio_apetrack'))"`),
   unset `APETRACK_POSE_WEIGHTS`, run a video: it must download and verify.
Until then: `ApeTracker(pose_weights="weights/openapepose_hrnet_w48_fp16.safetensors")` or
`export APETRACK_POSE_WEIGHTS=/path/to/file`.

OWLv2 is downloaded anonymously from the Hugging Face Hub by `transformers` (no token needed). Optional: mirror it on
Zenodo too if you want the package independent of Hugging Face.

## 3. PyPI
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

## 4. Demo page (Quarto)
- Videos: `python tools/make_demo.py` (GPU; ~1.5 h for all samples). Samples in `demo/samples/` (each < 100 MB;
  grooming2 and wamba12 are cut copies, originals in the old LABEXPERIMENTING folder).
- Page: `cd demo && quarto render` (needs `pip install jupyter tabulate` in the env with the package).
  The settings table and code dropdowns are generated from the package source at render time.
- `demo/output/` and `demo/samples/` are git-ignored (hundreds of MB). Host the rendered `demo/_site/` (with videos)
  e.g. on the envisionBOX website server, not in the git repository. [TODO: decide where]
- Check the licences/permissions of the sample videos before publishing them (Chiba zoo, DeepWild sites, Beekse Bergen).

## 5. Evaluation (not in the package)
- `evaluation/data/` (git-ignored): `testdata_lab5/` (SAM2 boxes of one siamang in 11 Chiba clips, frame-aligned with
  demo/samples) and `testdata_wiltshire/` (DeepWild labels + images).
- `evaluation/legacy_dev/`: the scripts used during testing (`tune_boxes.py`, `tune_deepwild.py`, detector
  comparisons, `sam2_pipeline_test.py` = SAM2 route, on hold). They import the pre-package `pipeline_test.py` and
  expect the old folder layout. [TODO: port to the package API if the numbers should be reproducible from this repo.]
