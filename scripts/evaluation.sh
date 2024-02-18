python eval.py \
    --model_name=meta=llama/Llama-2-7b-hf \
    --bf16=1 \
    --load_checkpoint_path=/path/to/checkpoint \
    --test_batch_size=2 \
    --train_type=