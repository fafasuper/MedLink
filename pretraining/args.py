import argparse


def get_pretraining_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", type=str, required=True, choices=["bert", "roberta", "longformer"],
                        help="Type of model to use for pretraining")
    parser.add_argument("--train_file", type=str, required=True, help="The input training data file (a jsonl file).")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")

    # Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0.5, type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--dropout_prob", type=float, default=0.2, help="Dropout probability for all layers.")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")

    # parser.add_argument("--task_weights", nargs=5, type=float, default=[1.0, 1.0, 1.0, 1.0, 1.0], help="Weights for each task loss. Order: MLM, Hypernym, Entity Type, Entity, Synonym")
    parser.add_argument("--num_entity_types", type=int, default=17,
                        help="Number of entity types for entity type prediction task")

    args = parser.parse_args()

    return args