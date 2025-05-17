import os 
import logging
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from transformers import (
    BertConfig, RobertaConfig, LongformerConfig,
    BertTokenizer, RobertaTokenizer, LongformerTokenizer,
    BertModel, RobertaModel, LongformerModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.longformer.modeling_longformer import LongformerLMHead
from pretraining_dataset import PretrainingDataset
from pretraining_model import EntityAwarePreTrainingModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="A parameter name that contains `beta`")
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma`")

try:
    from torch.cuda.amp import autocast, GradScaler
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

MODEL_CLASSES = {
    "bert": (BertConfig, BertTokenizer, BertModel, BertLMPredictionHead),
    "roberta": (RobertaConfig, RobertaTokenizer, RobertaModel, RobertaLMHead),
    "longformer": (LongformerConfig, LongformerTokenizer, LongformerModel, LongformerLMHead),
}

def set_seed(args):
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, model, tokenizer):
    """ Train the model """
    train_dataset = PretrainingDataset(args.train_file, tokenizer, args.max_seq_length, args.model_type)

    if args.local_rank == -1:
        train_sampler = None
        total_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    else:
        train_sampler = DistributedSampler(train_dataset)
        total_batch_size = args.per_gpu_train_batch_size * args.world_size

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=total_batch_size,
        num_workers=1,
        pin_memory=True
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                    len(train_dataset) // (total_batch_size * args.gradient_accumulation_steps)) + 1
    else:
        t_total = len(train_dataset) // (total_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16 and APEX_AVAILABLE:
        scaler = GradScaler()

    # Setup TensorBoard
    tb_writer = SummaryWriter(log_dir="/root/tf-logs/pretrain-medlink-roberta-base-pm-m3-20ksamples")

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)} (estimated)")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {global_step}")
            logger.info(f"  Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    for epoch in range(epochs_trained, int(args.num_train_epochs)):
        if args.local_rank != -1:
            train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = [t.to(args.device) for t in batch]
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                # "token_type_ids": batch[2],
                "mlm_labels": batch[2],
                "entity_type_labels": batch[3],
                # 移除了 entity_name_labels
            }

            logger.info(f"Epoch: {epoch}, Step: {step}, Global Step before accumulation: {global_step}")

            if args.fp16 and APEX_AVAILABLE:
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs["total_loss"]
                    mlm_loss = outputs['mlm_loss']
                    entity_type_loss = outputs['entity_type_loss']
            else:
                outputs = model(**inputs)
                loss = outputs["total_loss"]
                mlm_loss = outputs['mlm_loss']
                entity_type_loss = outputs['entity_type_loss']

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16 and APEX_AVAILABLE:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                logger.info(f"Accumulating gradients for Step: {step}, performing optimizer step at Global Step: {global_step}")
                if args.fp16 and APEX_AVAILABLE:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                logger.info(f"Updated Global Step: {global_step}")

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logger.info(
                        f"Epoch: {epoch}, Step: {global_step}, "
                        f"Loss: {(tr_loss - logging_loss) / args.logging_steps:.5f}, "
                        f"MLM loss: {mlm_loss:.3f}, "
                        f"Entity type loss: {entity_type_loss:.3f}, "
                        f"Task weights: {outputs['task_weights']} "
                    )
                    tb_writer.add_scalar("Loss/total", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar("Loss/mlm", mlm_loss, global_step)
                    tb_writer.add_scalar("Loss/entity_type", entity_type_loss, global_step)
                    # 记录 task_weights 到 TensorBoard
                    task_weights = outputs["task_weights"]  # 获取任务权重
                    tb_writer.add_scalar("TaskWeight/mlm", task_weights["mlm"], global_step)
                    tb_writer.add_scalar("TaskWeight/entity_type", task_weights["entity_type"], global_step)

                    tb_writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    #torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info(f"Saving model checkpoint to {output_dir}")

                    model_to_save = model.module if hasattr(model, "module") else model
                    # model_to_save.save_pretrained(output_dir)
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    tb_writer.close()
    return global_step, tr_loss / global_step

def main():
    from args import get_pretraining_args
    args = get_pretraining_args()
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config_class, tokenizer_class, model_class, mlm_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    
    # 设置 Dropout 概率
    config.hidden_dropout_prob = args.dropout_prob  # 设置全连接层的 Dropout 概率
    config.attention_probs_dropout_prob = args.dropout_prob  # 设置注意力机制中的 Dropout 概率
    
    config.num_entity_types = args.num_entity_types
    
    config._model_type = args.model_type

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, use_fast=True, clean_up_tokenization_spaces=True)
    backbone_model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model = EntityAwarePreTrainingModel(config, backbone_model)
    #model = EntityAwarePreTrainingModel.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    # Distributed and FP16 training
    if args.local_rank != -1:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info("Saving model checkpoint to %s", args.output_dir)
        
        # Prepare config for saving
        if hasattr(config, 'model'):
            delattr(config, 'model')
        if hasattr(config, 'mlm_head'):
            delattr(config, 'mlm_head')
            
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                "module") else model  # Take care of distributed/parallel training
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        try:
            # Save model state
            model_to_save.save_pretrained(args.output_dir)
            # Save tokenizer
            tokenizer.save_pretrained(args.output_dir)
            # Save training arguments
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error during model saving: {str(e)}")
            raise

if __name__ == "__main__":
    main()
