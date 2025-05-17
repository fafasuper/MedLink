import json
import os
import torch
from torch.utils.data import IterableDataset
import random
import logging

logger = logging.getLogger(__name__)

class PretrainingDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_seq_length, model_type):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        self.total_length = self._get_total_lines()
        self.entity_type_to_id = self._load_json("./type_to_id.json")


    def __iter__(self):
        logger.info("Starting new iteration over dataset.")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    yield self.process_example(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding JSON at line {line_num}: {e}")
                    logger.warning(f"Problematic line: {line[:100]}...")  # Log the first 100 characters of the line
                    continue  # Skip this line and continue with the next one

    def __len__(self):
        return self.total_length

    def _get_total_lines(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
                logger.info(f'Total number of lines: {total_lines}')
                return total_lines
        except Exception as e:
            logger.error(f"Error counting lines in file: {e}")
            return 0


    def _load_json(self, file_name):
        try:
            with open(file_name, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {file_name}")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON in file: {file_name}")
            return {}

    def process_example(self, example):
        text = example['text']
        annotations = example['annotations']

        # Tokenize text
        encoded = self.tokenizer(text, add_special_tokens=True, max_length=self.max_seq_length,
                                 truncation=True, padding='max_length', return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        if self.model_type in ['bert']:
            token_type_ids = encoded['token_type_ids'].squeeze()
        else:
            token_type_ids = torch.zeros_like(input_ids)  # For RoBERTa

        # Initialize labels
        mlm_labels = torch.full_like(input_ids, -100, dtype=torch.long)
        entity_type_labels = torch.zeros((len(input_ids), len(self.entity_type_to_id)), dtype=torch.float)


        # Process annotations
        for ann in annotations:
            # 获取所有可能的实体类型ID列表
            entity_type_ids = [self.entity_type_to_id.get(t, -1) for t in ann['types']]


            mention = ' ' + ann['mention'] if self.model_type == 'roberta' else ann['mention']
            mention_tokens = self.tokenizer.encode(mention, add_special_tokens=False)


            for i in range(len(input_ids) - len(mention_tokens) + 1):
                if input_ids[i:i + len(mention_tokens)].tolist() == mention_tokens:
                    # Mask tokens for MLM task
                    original_tokens = input_ids[i:i + len(mention_tokens)].clone()
                    input_ids[i:i + len(mention_tokens)] = self.tokenizer.mask_token_id
                    mlm_labels[i:i + len(mention_tokens)] = original_tokens

                    # Set entity type labels (multi-label)
                    for type_id in entity_type_ids:
                        if type_id != -1:
                            entity_type_labels[i, type_id] = 1.0

                    break

        # Additional masking for MLM
        num_masked = (mlm_labels != -100).sum().item()
        num_to_mask = max(1, int(0.15 * len(input_ids)) - num_masked)
        masked_positions = random.sample(range(1, len(input_ids) - 1), num_to_mask)
        for i in masked_positions:
            if mlm_labels[i] == -100:
                mlm_labels[i] = input_ids[i].item()
                if random.random() < 0.8:
                    input_ids[i] = self.tokenizer.mask_token_id
                elif random.random() < 0.5:
                    input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)

        return (
            input_ids,
            attention_mask,
            mlm_labels,
            entity_type_labels,
           
        )
