#!/bin/bash
# Evaluate TS-DP on all tasks (or the tasks given as arguments).
# Run from the repository root; checkpoint layout as described in README.md.
#   DEVICE=cuda:0 N_PARALLEL=28 bash scripts/eval_all.sh [task ...]
set -e

CKPT=${CKPT:-data/checkpoints}
TSDP=${TSDP:-data/ts_dp}
OUT=${OUT:-data/eval}
DEVICE=${DEVICE:-cuda:0}
N_PARALLEL=${N_PARALLEL:-28}
export MUJOCO_GL=${MUJOCO_GL:-egl}

TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(lift can square transport tool_hang pusht \
  lift_mh can_mh square_mh transport_mh kitchen block_pushing)

for name in "${TASKS[@]}"; do
  case $name in
    lift|can|square|transport)
      task=$name
      target=$CKPT/image/${name}_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt ;;
    tool_hang)
      task=$name
      target="$CKPT/image/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/epoch=0100-test_mean_score=0.773.ckpt" ;;
    lift_mh|can_mh|square_mh|transport_mh)
      task=${name%_mh}
      target=$CKPT/image/${name}/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt ;;
    pusht)
      task=$name
      target=$CKPT/image/pusht/diffusion_policy_transformer/train_1/checkpoints/latest.ckpt ;;
    kitchen)
      task=$name
      target="$CKPT/low_dim/kitchen/diffusion_policy_transformer/train_1/checkpoints/epoch=2700-test_mean_score=0.574.ckpt" ;;
    block_pushing)
      task=$name
      target=$CKPT/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt ;;
    *) echo "unknown task: $name"; exit 1 ;;
  esac
  echo "=== $name"
  python scripts/eval_scheduler.py --task "$task" \
    --checkpoint "$target" \
    --drafter-checkpoint "$TSDP/$name/drafter.ckpt" \
    --ppo-checkpoint "$TSDP/$name/scheduler.pth" \
    --output-dir "$OUT/$name" \
    --device "$DEVICE" --n-parallel-envs "$N_PARALLEL"
done
