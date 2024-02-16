python ppo.py \
    --train_type=PPO \
    --project_name=ehrsql-2024-ppo \
    --train_epochs=3 \
    --train_batch_size=4 \
    --model_name=meta-llama/Llama-2-7b-hf \
    --learning_rate=1e-3 \
    --load_checkpoint_path=/path/to/adapter \
    --bf16=1 \
    --num_samples=400