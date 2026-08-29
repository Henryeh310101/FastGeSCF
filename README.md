# FastGeSCF

FastGeSCF is a generalizable scene change detection project that combines SAM2, LightGlue registration, and a trained robust change-mask generator.

## Repository Layout

```text
experiments/              Runnable experiment entry points
model/                    FastGeSCF change-detection model
datasets/                 Dataset loaders and preprocessing helpers
sam2/                     Vendored SAM2 runtime used by FastGeSCF
LightGlue/                External LightGlue checkout, ignored by git
salad/                    Vendored SALAD VPR model code
py_utils/                 Evaluation, VPR, and visualization utilities
checkpoints/              Local checkpoint directory, ignored except download script
docs/EXPERIMENTS.md       Detailed experiment commands and dataset path setup
scripts/                  Smoke checks and local validation helpers
```

## Environment

Create the environment from the provided Conda file, then install Python requirements if needed:

```bash
conda env create -f env.yml
conda activate vcd
pip install -r requirements.txt
```

The current local validation was run with Python 3.11.

## External Dependencies

LightGlue is maintained by another project and is intentionally not committed here. Clone it into the repository root before running FastGeSCF registration or inference:

```bash
git clone https://github.com/cvg/LightGlue.git LightGlue
```

The code imports it as `LightGlue.lightglue`, so the checkout directory should be named `LightGlue`.

## Checkpoints

Download SAM2.1 checkpoints:

```bash
bash checkpoints/download_ckpts.sh
```

FastGeSCF trained weights are expected locally at:

```text
results/RobustViT_vl-cmu-cd/best_model.pth
results/RobustViT_changesim/best_model.pth
```

Large checkpoints and generated result folders are ignored by git. Use CLI flags such as `--sam2_checkpoint`, `--outdoor_ckpt`, and `--indoor_ckpt` to point to files stored elsewhere.

## Quick Validation

```bash
bash scripts/run_local_checks.sh
```

This compiles the project modules and checks all experiment CLIs.

Dry-run an experiment against local data without loading heavy models:

```bash
python experiments/train.py --dataset vl-cmu-cd --data_root /path/to/VL-CMU-CD-binary255 --dry_run
python experiments/evaluate_datasets.py --datasets SF-XL --data_root /path/to/SF-XL --dry_run
```

## Common Runs

Single image pair:

```bash
python experiments/infer_pair.py --t0 /path/to/t0.png --t1 /path/to/t1.png --gt /path/to/mask.png
```

Dataset evaluation:

```bash
python experiments/evaluate_datasets.py --datasets SF-XL --data_root /path/to/SF-XL
```

Training:

```bash
python experiments/train.py --dataset vl-cmu-cd --data_root /path/to/VL-CMU-CD-binary255
```

Temporal alignment:

```bash
python experiments/temporal_alignment.py --base /path/to/Query_Seq_Test --env dust
```

Video change detection:

```bash
python experiments/video_change_detection.py --query_dir /path/to/query_frames --ref_dir /path/to/reference_frames
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the full runnable command set.

## License Notes

This repository includes vendored third-party code. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license files inside vendored directories.
