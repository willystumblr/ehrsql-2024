python data/mimic_iv/build-dpo-data.py \
    --output_dir=data/mimic_iv \
    --load_checkpoint_path=/path/to/sft-checkpoint \
    --bf16=1 \
    --model_name=meta-llama/Llama-2-7b-hf \
    --train_batch_size=4 \
    --num_return_sequences=2 \
    --build_type=train

wait

python data/mimic_iv/build-dpo-data.py \
    --output_dir=data/mimic_iv \
    --load_checkpoint_path=/path/to/sft-checkpoint \
    --bf16=1 \
    --model_name=meta-llama/Llama-2-7b-hf \
    --train_batch_size=4 \
    --num_return_sequences=2 \
    --build_type=valid
