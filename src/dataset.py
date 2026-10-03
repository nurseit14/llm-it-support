import json

import torch
from torch.utils.data import Dataset


class ITSupportDataset(Dataset):

    def __init__(
        self,
        path,
        tokenizer,
        max_length=512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples = []

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if line.strip():

                    example = json.loads(line)

                    self.examples.append(example)


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index):

        example = self.examples[index]

        messages = example["messages"]


        # ====================================================
        # COMPLETE CONVERSATION
        # ====================================================

        formatted_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )
        )


        # ====================================================
        # PROMPT WITHOUT ASSISTANT RESPONSE
        # ====================================================

        prompt_messages = messages[:-1]

        prompt_text = (
            self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )


        # ====================================================
        # TOKENIZE COMPLETE CONVERSATION
        # ====================================================

        inputs = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )


        # ====================================================
        # TOKENIZE PROMPT
        # ====================================================

        prompt_inputs = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )


        # Remove temporary batch dimension:
        #
        # [[1, 2, 3]]
        #
        # becomes:
        #
        # [1, 2, 3]

        input_ids = (
            inputs["input_ids"]
            .squeeze(0)
        )

        attention_mask = (
            inputs["attention_mask"]
            .squeeze(0)
        )


        # ====================================================
        # CREATE LABELS
        # ====================================================

        labels = input_ids.clone()


        # Number of tokens before assistant response
        prompt_length = min(
            prompt_inputs["input_ids"].shape[1],
            labels.shape[0]
        )


        # Ignore:
        #
        # system prompt
        # user message
        # assistant generation marker
        #
        # Only assistant response contributes to loss.

        labels[:prompt_length] = -100


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def create_collate_fn(tokenizer):

    def collate_fn(batch):

        input_ids = [
            item["input_ids"]
            for item in batch
        ]

        attention_masks = [
            item["attention_mask"]
            for item in batch
        ]

        labels = [
            item["labels"]
            for item in batch
        ]


        # ====================================================
        # PAD INPUT IDS
        # ====================================================

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )


        # ====================================================
        # PAD ATTENTION MASK
        # ====================================================

        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0
        )


        # ====================================================
        # PAD LABELS
        #
        # -100 means:
        # do not calculate loss for this position
        # ====================================================

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )


        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }


    return collate_fn