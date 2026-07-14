# RoboTwin `place_block_aba_grid` pi05 fine-tuning

## Data

- Raw source (read-only): `/home/ubuntu/Workspace/RoboTwin/data/place_block_aba_grid/demo_clean`
- Converted LeRobot dataset: `/home/ubuntu/.cache/huggingface/lerobot/local/robotwin_place_block_aba_grid`
- Mapping: `head_camera -> cam_high`, `left_camera -> cam_left_wrist`, and
  `right_camera -> cam_right_wrist`.
- Each training frame uses the 14-D `joint_action/vector` at time *t* as state
  and time *t+1* as the absolute target action. The openpi ALOHA transform
  changes the first 6 joints of each arm to deltas and keeps grippers absolute.

## Commands

```bash
cd /home/ubuntu/Workspace/openpi
uv run --no-sync examples/robotwin/convert_robotwin_data_to_lerobot.py \
  --raw-dir /home/ubuntu/Workspace/RoboTwin/data/place_block_aba_grid/demo_clean \
  --repo-id local/robotwin_place_block_aba_grid --fps 50

CUDA_VISIBLE_DEVICES=1,2 uv run --no-sync scripts/compute_norm_stats.py \
  --config-name pi05_robotwin_place_block_aba_grid_lora

CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  uv run --no-sync scripts/train.py pi05_robotwin_place_block_aba_grid_lora_smoke \
  --exp-name place-block-aba-grid-pi05-base-20260712T145757Z-smoke --overwrite

CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  uv run --no-sync scripts/train.py pi05_robotwin_place_block_aba_grid_lora \
  --exp-name place-block-aba-grid-pi05-base-20260712T145757Z --overwrite
```

The normal training and smoke test use separate config names and checkpoint
directories. Both load `/home/ubuntu/Model/pi05_base/params` read-only.

## Smoke-test blocker (2026-07-12)

The smoke test successfully initialized the dataset, normalization, two-GPU
FSDP mesh, checkpoint directory, and W&B run. It then failed while restoring
the supplied base checkpoint:

```
FAILED_PRECONDITION: Error reading
params.PaliGemma.img.Transformer.encoderblock.MlpBlock_0.Dense_1.kernel/0.0.0
... Truncated Zstd-compressed stream
```

This means `/home/ubuntu/Model/pi05_base/params` is incomplete or corrupt.
Replace it with a verified complete `pi05_base` checkpoint, then rerun the
normalization command and smoke command above before launching the full run.
