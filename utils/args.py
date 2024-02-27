import argparse

def add_default_args(parser: argparse.ArgumentParser):
    """
    Define and set default arguments for the script.
    """
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--db_id", type=str, default="mimiciii", help="database name")  # NOTE: `mimic_iv` will be used for codabench
    parser.add_argument("--train_data_dir", type=str, help="train data path")
    parser.add_argument("--train_type", type=str, choices=["SFT", "PPO", "DPO"])

    parser.add_argument("--valid_data_dir", type=str, help="valid data path")
    parser.add_argument("--test_data_dir", type=str, help="test data path")
    parser.add_argument("--tables_file", type=str, help="table schema path")
    parser.add_argument("--wandb_dir", type=str, default="./", help="wandb log directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="output directory")
    parser.add_argument("--output_file", type=str, default="prediction_raw.json", help="output file name")

    # basic parameters
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_checkpoint_path", type=str, default=None)
    parser.add_argument("--load_checkpoint_path", type=str, default=None)
    parser.add_argument("--load_adapter_path", type=str, default=None)
    parser.add_argument("--load_ref_checkpoint_path", type=str, default=None)

    # training parameters
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=str, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--evaluation_strategy", type=str, default='no')
    parser.add_argument("--save_strategy", type=str, default='steps')
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=50) #logging_first_step
    parser.add_argument("--logging_first_step", type=bool, default=True)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--build_type", type=str, default=None, choices=["train", "valid"])

    # lora parameters
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=int, default=0.1)
    parser.add_argument("--lora_r", type=int, default=1)


    parser.add_argument("--save_every_epoch", type=bool, default=False)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    # generation parameters
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    return parser

def update_args(new_args, prev_args):
    """
    Update training arguments with the values saved in the checkpoint.
    """
    for arg in vars(prev_args):
        if arg not in new_args:
            setattr(new_args, arg, getattr(prev_args, arg))
    return new_args
    
