# Experiments

Run commands from the repository root. All experiment scripts support `--device auto`, which selects CUDA when available and falls back to CPU.

For the method overview, paper figures, and reported benchmark tables, see [PAPER.md](PAPER.md).

## Required Artifacts

LightGlue is an external dependency. Clone it into the repository root:

```bash
git clone https://github.com/cvg/LightGlue.git LightGlue
```

The local `LightGlue/` checkout is ignored by git.

SAM2.1 large checkpoint:

```bash
bash checkpoints/download_ckpts.sh
```

FastGeSCF trained weights are expected at:

```text
results/RobustViT_vl-cmu-cd/best_model.pth
results/RobustViT_changesim/best_model.pth
```

These large files are ignored by git. You can override their locations with:

```bash
--sam2_checkpoint /path/to/sam2.1_hiera_large.pt
--outdoor_ckpt /path/to/outdoor_best_model.pth
--indoor_ckpt /path/to/indoor_best_model.pth
```

## Dataset Paths

Pass dataset roots explicitly, set `FASTGESCF_DATA_ROOT`, or set dataset-specific variables:

```bash
export FASTGESCF_DATA_ROOT=/path/to/data
export FASTGESCF_VL_CMU_CD_ROOT=/path/to/VL-CMU-CD-binary255
export FASTGESCF_PSCD_ROOT=/path/to/pscd_or_1024x224
export FASTGESCF_ST_LUCIA_ROOT="/path/to/St Lucia"
export FASTGESCF_NORDLAND_ROOT=/path/to/Nordland
export FASTGESCF_SF_XL_ROOT=/path/to/SF-XL
export FASTGESCF_CHANGESIM_TRAIN_ROOT=/path/to/Query_Seq_Train
export FASTGESCF_CHANGESIM_TEST_ROOT=/path/to/Query_Seq_Test
```

## Smoke Checks

```bash
bash scripts/run_local_checks.sh
```

This verifies Python syntax, core imports, and CLI parsing without loading checkpoints or datasets.

Dry-run checks validate local dataset/checkpoint paths without starting heavy model execution:

```bash
python experiments/train.py --dataset vl-cmu-cd --data_root /path/to/VL-CMU-CD-binary255 --dry_run
python experiments/evaluate_datasets.py --datasets SF-XL --data_root /path/to/SF-XL --dry_run
python experiments/infer_pair.py --t0 /path/to/t0.png --t1 /path/to/t1.png --gt /path/to/mask.png --dry_run
python experiments/video_change_detection.py --query_dir /path/to/query_frames --ref_dir /path/to/reference_frames --dry_run
python experiments/temporal_alignment.py --base /path/to/Query_Seq_Test --env dust --dry_run
```

## Single Image Pair

```bash
python experiments/infer_pair.py \
  --t0 /path/to/t0.png \
  --t1 /path/to/t1.png \
  --gt /path/to/mask.png \
  --output outputs/comparison.png
```

## Dataset Evaluation

```bash
python experiments/evaluate_datasets.py \
  --datasets SF-XL Nordland VL_CMU_CD PSCD ChangeSim \
  --device auto
```

Use `--data_root /path/to/dataset` when evaluating one dataset:

```bash
python experiments/evaluate_datasets.py --datasets SF-XL --data_root /path/to/SF-XL
```

## Training

```bash
python experiments/train.py \
  --dataset vl-cmu-cd \
  --data_root /path/to/VL-CMU-CD-binary255 \
  --batch_size 4 \
  --num_epochs 50
```

Resume or test a saved checkpoint:

```bash
python experiments/train.py \
  --dataset vl-cmu-cd \
  --data_root /path/to/VL-CMU-CD-binary255 \
  --test \
  --ckpt /path/to/best_model.pth
```

## Temporal Alignment

```bash
python experiments/temporal_alignment.py \
  --base /path/to/Query_Seq_Test \
  --env dust \
  --ckpt_path checkpoints/dino_salad.ckpt \
  --output_dir outputs
```

## Video Change Detection

```bash
python experiments/video_change_detection.py \
  --query_dir /path/to/query_frames \
  --ref_dir /path/to/reference_frames \
  --vpr_ckpt_path checkpoints/dino_salad.ckpt \
  --output_dir outputs/video_change_detection
```
