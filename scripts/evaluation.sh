# python eval.py \
#     --model_name=meta-llama/Llama-2-7b-hf \
#     --bf16=1 \
#     --load_checkpoint_path=ckpt/ehrsql-2024/SFT/festive-bao-5 \
#     --test_batch_size=1 \
#     --train_type=SFT

# wait

python eval.py \
	--model_name=meta-llama/Llama-2-7b-hf \
	--bf16=1 \
	--load_adapter_path=ckpt/ehrsql-2024/SFT/festive-bao-5 \
	--load_checkpoint_path=ckpt/ehrsql-2024/PPO/incandescent-ox-24 \
	--test_batch_size=1 \
	--train_type=PPO

