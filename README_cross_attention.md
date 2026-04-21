# Cross-Attention Branch

This document describes how to run the `text_cross_attention` experiment on the current `cross_attn` branch.

## What Changed

This branch adds a text-guided fusion path on top of the original VRP-SAM pipeline:

- dataset returns a `class_name` field
- text prompts are built from class names with a prompt template
- text features are extracted with `clip.load('ViT-B/16')`
- text tokens are fused into the decoder with cross-attention

The original visual-only path is still available with `--fusion_type visual`.

## Environment

Use the `vrpsam` conda environment:

```bash
conda activate vrpsam
```

Install the base dependencies from the main README first, then install SAM and CLIP:

```bash
cd segment-anything
pip install -v -e .
cd ..

pip install git+https://github.com/openai/CLIP.git
```

Notes:

- `clip.load('ViT-B/16')` will download weights automatically on first use.
- the first run must have network access

## Dataset Layout

Use the dataset layout required by the current code, not the older README wording.

Expected `--datapath` layout:

```text
$DATAPATH/
├── VOC2012/
│   ├── JPEGImages/
│   └── SegmentationClassAug/
└── MSCOCO14/
    ├── train2014/
    ├── val2014/
    └── annotations/
        ├── train2014/
        └── val2014/
```

Notes:

- PASCAL is read from `VOC2012/JPEGImages` and `VOC2012/SegmentationClassAug`
- COCO is read from `MSCOCO14`, not `COCO2014`
- COCO masks are expected as per-image PNG files under `annotations/train2014` and `annotations/val2014`

## Text Prompt

The current text prompt is built from:

```text
a photo of a {class_name}.
```

You can change it with:

```bash
--text_prompt_template "a segmented {class_name}."
```

## Training Command

Example 4-GPU training command for the cross-attention setting:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=6224 \
    train.py \
    --datapath /path/to/datasets \
    --logpath logs/cross_attn_coco_fold0 \
    --benchmark coco \
    --backbone resnet50 \
    --fold 0 \
    --condition mask \
    --num_query 50 \
    --epochs 50 \
    --lr 1e-4 \
    --bsz 2 \
    --fusion_type text_cross_attention \
    --text_model_name ViT-B/16 \
    --text_prompt_template "a photo of a {class_name}."
```

Example PASCAL run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=6224 \
    train.py \
    --datapath /mnt/disk1/szchen/VLMBenchmark/repo/Datasets_HSN \
    --logpath logs/cross_attn_pascal_fold0 \
    --benchmark pascal \
    --backbone resnet50 \
    --fold 0 \
    --condition mask \
    --num_query 50 \
    --epochs 50 \
    --lr 1e-4 \
    --bsz 2 \
    --fusion_type text_cross_attention \
    --text_model_name ViT-B/16
```

## Visual Baseline

To run the original visual-only baseline on this branch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=6224 \
    train.py \
    --datapath /path/to/datasets \
    --logpath logs/visual_baseline_coco_fold0 \
    --benchmark coco \
    --backbone resnet50 \
    --fold 0 \
    --condition mask \
    --num_query 50 \
    --epochs 50 \
    --lr 1e-4 \
    --bsz 2 \
    --fusion_type visual
```

## Key Arguments

- `--fusion_type`: `visual` or `text_cross_attention`
- `--text_model_name`: currently expected to be `ViT-B/16`
- `--text_prompt_template`: prompt template used to build class text
- `--condition`: one of `point`, `scribble`, `box`, `mask`
- `--benchmark`: `pascal` or `coco`

## Outputs

Training logs and checkpoints are written under the path passed to `--logpath`.

## Common Issues

`ModuleNotFoundError: No module named 'clip'`

- install CLIP with `pip install git+https://github.com/openai/CLIP.git`

`RuntimeError` during first text-model load

- the branch uses `clip.load('ViT-B/16')`
- confirm the machine can download weights on first run

Dataset not found

- check that `--datapath` points to a root containing `VOC2012` and/or `MSCOCO14`
- for COCO, confirm the PNG masks exist under `annotations/train2014` and `annotations/val2014`
