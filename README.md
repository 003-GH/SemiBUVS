# SemiBUVS

This is the official code for the paper "SAM-guided Semi-supervised Breast Lesion Segmentation in Ultrasound Videos with a New Dataset".

This repository currently releases the second-stage training code of SemiBUVS.

This repository does not include datasets, checkpoints, training logs, evaluation outputs, or local pretrained weights. These files should be prepared locally following the structure below.

## Repository Layout

```text
SemiBUVS/
|-- train.py
|-- test.py
|-- evaluate.py
|-- vivim.py
|-- environment.yml
|-- libs/
|   |-- config/
|   |   `-- default.py
|   |-- dataset/
|   |   |-- data.py
|   |   |-- transform.py
|   |   `-- youtube.py
|   `-- utils/
`-- README.md
```

Ignored local directories include `Youtube-VOS-CL/`, `ckpt/`, `output/`, `runs/`, and `segformer-b3-finetuned-ade-512-512/`.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate SemiBUVS
```

If `mamba-ssm` or `causal-conv1d` fails to build, install the wheel versions that match your CUDA, Python, and PyTorch versions.

## Pretrained Backbone

The model loads a SegFormer-B3 backbone with:

```python
OPTION.pretrained_segformer = 'segformer-b3-finetuned-ade-512-512'
```

Prepare it in one of these ways:

1. Put a local Hugging Face model directory at `segformer-b3-finetuned-ade-512-512/`.
2. Change `OPTION.pretrained_segformer` in `libs/config/default.py` to another local path or Hugging Face model identifier.

The local pretrained directory is ignored by Git and should not be uploaded.

## Dataset

The default dataset root is:

```python
OPTION.data_root = 'Youtube-VOS-CL'
```

Place the dataset under the project root, or set `OPTION.data_root` to an absolute path or another relative path.

The required structure is:

```text
Youtube-VOS-CL/
|-- train/
|   |-- label/
|   |   |-- JPEGImages/
|   |   |   |-- A10_avi/
|   |   |   |   |-- 00001.jpg
|   |   |   |   |-- 00002.jpg
|   |   |   |   `-- ...
|   |   |   `-- ...
|   |   |-- Annotations/
|   |   |   |-- A10_avi/
|   |   |   |   |-- 00001.png
|   |   |   |   |-- 00002.png
|   |   |   |   `-- ...
|   |   |   `-- ...
|   |   `-- meta.json
|   `-- unlabel/
|       |-- JPEGImages/
|       |   |-- B10_avi/
|       |   |   |-- 00001.jpg
|       |   |   |-- 00002.jpg
|       |   |   `-- ...
|       |   `-- ...
|       |-- Annotations/
|       |   |-- B10_avi/
|       |   |   |-- 00001.png
|       |   |   |-- 00002.png
|       |   |   `-- ...
|       |   `-- ...
|       `-- meta.json
`-- valid/
    |-- JPEGImages/
    |   |-- 101_avi/
    |   |   |-- 00001.jpg
    |   |   |-- 00002.jpg
    |   |   `-- ...
    |   `-- ...
    |-- Annotations/
    |   |-- 101_avi/
    |   |   |-- 00001.png
    |   |   |-- 00002.png
    |   |   `-- ...
    |   `-- ...
    `-- meta.json
```

### File Naming Rules

- Frame images must be `.jpg`.
- Mask images must be `.png`.
- Frame and mask names should use five-digit frame ids such as `00001.jpg` and `00001.png`.
- Each video must have matching folder names under `JPEGImages/` and `Annotations/`.
- The current code supports binary segmentation. Mask value `1` is foreground; `0` is background. Mask value `255` is reset to background during loading.
- The validation loader uses the first available annotation frame as the reference frame and evaluates frames from that point onward.

### `meta.json` Format

Each split requires a `meta.json` file with a top-level `videos` dictionary. The loader uses the video names from this file to index the dataset.

Example:

```json
{
  "videos": {
    "A10_avi": {
      "objects": {
        "1": {
          "category": "tumor",
          "frames": [
            "00001",
            "00002",
            "00003"
          ]
        }
      }
    }
  }
}
```

The video keys in `meta.json` must match the corresponding folder names in `JPEGImages/` and `Annotations/`.

### Note About `train/unlabel`

Although the split is named `unlabel`, the current implementation still reads masks from `train/unlabel/Annotations/`. These masks are used by the existing training code when computing the pseudo-supervised and contrastive terms. If you want to use truly unlabeled data, the training logic should be modified accordingly.

## Configuration

Default options are defined in `libs/config/default.py`. Common settings include:

```python
OPTION.data_root = 'Youtube-VOS-CL'
OPTION.input_size = (256, 256)
OPTION.sampled_frames = 3
OPTION.epochs = 100
OPTION.train_batch = 4
OPTION.learning_rate = 0.0001
OPTION.checkpoint = 'ckpt'
OPTION.output_dir = 'output'
OPTION.pretrained_segformer = 'segformer-b3-finetuned-ade-512-512'
OPTION.contrastive_memory_size = 32
```

You can override options from the command line:

```bash
python train.py data_root /path/to/Youtube-VOS-CL gpu_id 0
```

Or create a custom config file and run:

```bash
python train.py --cfg path/to/config.yaml
```

## Training

Run:

```bash
python train.py
```

By default, checkpoints are saved to:

```text
ckpt/recurrent{epoch}.pth.tar
```

TensorBoard logs are saved under `runs/`.

Training uses:

- labeled training clips from `Youtube-VOS-CL/train/label`
- unlabeled-split clips from `Youtube-VOS-CL/train/unlabel`
- a student model optimized by backpropagation
- a teacher model updated by EMA
- segmentation loss, consistency loss, and contrastive loss
- contrastive memory size controlled by `OPTION.contrastive_memory_size`

The training script does not run validation during training. Use `test.py` and `evaluate.py` after checkpoints are generated.

## Testing

Run inference for all `.pth.tar` checkpoints in `ckpt/`:

```bash
python test.py
```

Predicted masks are saved under:

```text
output/VOS/{checkpoint_name}/{video_name}/{frame_id}.png
```

To test a different checkpoint directory:

```bash
python test.py test_checkpoint_dir /path/to/checkpoints
```

## Evaluation

After running `test.py`, compute metrics with:

```bash
python evaluate.py \
  --epoch-root output/VOS \
  --gt-root Youtube-VOS-CL/valid/Annotations \
  --best-log output/best_log.txt \
  --result-log output/result_log.txt
```

The evaluator reports Dice, Accuracy, Precision, Recall, F-measure, and Jaccard.

## GitHub Upload Notes

The `.gitignore` excludes datasets, checkpoints, generated outputs, logs, Python caches, local environments, and local pretrained weights. Before pushing to GitHub, check:

```bash
git status --short
```

Make sure large local directories such as `Youtube-VOS-CL/`, `ckpt/`, `output/`, `runs/`, and `segformer-b3-finetuned-ade-512-512/` are not listed.
