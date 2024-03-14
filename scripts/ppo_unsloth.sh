export PJRT_DEVICE=CUDA
python -m torch.distributed.launch --nproc_per_node 2 ppo_unsloth.py \
    --train_type=PPO \
    --project_name=ehrsql-2024-ppo \
    --train_epochs=5 \
    --train_batch_size=8 \
    --learning_rate=1e-3 \
    --load_checkpoint_path=/path/to/adapter \
    --bf16=1 \
    --gradient_accumulation_steps=1 \
    --phase=dev \