python sft.py \
    --train_type=SFT \
    --project_name=ehrsql-2024-sft \
    --model_name=meta-llama/Llama-2-7b-hf \
    --train_epochs=3 \
    --train_batch_size=8 \
    --valid_batch_size=4 \
    --learning_rate=1e-3 \
    --logging_steps=10 \
    --lr_scheduler_type=cosine \
    --bf16=1 \
    --db_id=mimic_iv \
    --evaluation_strategy=epoch \
    --test_batch_size=1 \
    --save_strategy=epoch \
    --load_best_model_at_end=True