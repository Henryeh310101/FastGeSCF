# Third-Party Notices

FastGeSCF uses code from these projects:

- SAM2 from Meta Platforms, Inc. The original upstream license is Apache-2.0. Several files under `sam2/` retain Meta copyright headers.
- LightGlue from https://github.com/cvg/LightGlue. It is an external dependency and should be cloned locally into `LightGlue/`; it is not committed to this repository.
- SALAD/VPR code under `salad/`; see `salad/LICENSE`.
- Utility code under `py_utils/`; see `py_utils/LICENSE`.

Large model checkpoints and generated experiment outputs are intentionally excluded from git. Download or place the required checkpoints as described in `docs/EXPERIMENTS.md`.
