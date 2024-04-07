python eval_pipeline.py \
    --model_name_2=willystumblr/ehrsql-2024-sft-text2sql-gemma-2b-it \
    --model_name=willystumblr/ehrsql-2024-sft-unanswerable-roberta-large \
    --bf16=1 \
    --phase=dev \
    --test_batch_size=1 \
    --base_model_name=FacebookAI/roberta-large
