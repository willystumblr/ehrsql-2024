CUDA_VISIBLE_DEVICES=0 python sft_unsloth.py \
    --train_type=SFT \
    --project_name=ehrsql-2024-sft \
    --model_name=unsloth/codellama-7b-bnb-4bit \
    --train_epochs=3 \
    --train_batch_size=4 \
    --valid_batch_size=4 \
    --learning_rate=1e-3 \
    --logging_steps=10 \
    --lr_scheduler_type=cosine \
    --bf16=1 \
    --db_id=mimic_iv \
    --phase=dev_final
