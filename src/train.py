import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from model import load_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.jsonl"
)

VAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "lora_adapter"
)

NUM_EPOCHS = 3
LEARNING_RATE = 2e-4


# ============================================================
# 2. HELPER: LOAD DATASET
# ============================================================

def load_dataset(path):
    examples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                example = json.loads(line)
                examples.append(example)

    return examples


# ============================================================
# 3. HELPER: PREPARE ONE EXAMPLE
# ============================================================

def prepare_example(example, tokenizer, device):

    messages = example["messages"]

    # --------------------------------------------------------
    # Complete conversation
    #
    # USER:
    # What is Docker?
    #
    # ASSISTANT:
    # Docker is a platform...
    # --------------------------------------------------------

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )


    # --------------------------------------------------------
    # Prompt WITHOUT assistant response
    #
    # USER:
    # What is Docker?
    #
    # ASSISTANT:
    # --------------------------------------------------------

    prompt_messages = messages[:-1]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # --------------------------------------------------------
    # Tokenize complete conversation
    # --------------------------------------------------------

    inputs = tokenizer(
        formatted_text,
        return_tensors="pt"
    ).to(device)


    # --------------------------------------------------------
    # Tokenize prompt only
    # --------------------------------------------------------

    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    ).to(device)


    input_ids = inputs["input_ids"]

    attention_mask = inputs[
        "attention_mask"
    ]


    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    labels = input_ids.clone()


    # Number of tokens BEFORE assistant answer
    prompt_length = (
        prompt_inputs["input_ids"]
        .shape[1]
    )


    # --------------------------------------------------------
    # Assistant-only loss masking
    #
    # SYSTEM     -> -100
    # USER       -> -100
    # ASSISTANT  -> real token IDs
    #
    # PyTorch ignores -100 when calculating loss.
    # --------------------------------------------------------

    labels[:, :prompt_length] = -100


    return (
        input_ids,
        attention_mask,
        labels
    )


# ============================================================
# 4. LOAD BASE MODEL
# ============================================================

model, tokenizer = load_model()

print("Base model loaded.")
print("Device:", model.device)


# ============================================================
# 5. CONFIGURE LoRA
# ============================================================

lora_config = LoraConfig(
    r=8,

    lora_alpha=16,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


# Attach LoRA adapters
model = get_peft_model(
    model,
    lora_config
)


print("\nLoRA attached:")

model.print_trainable_parameters()


# ============================================================
# 6. LOAD TRAINING + VALIDATION DATA
# ============================================================

training_examples = load_dataset(
    TRAIN_DATA_PATH
)

validation_examples = load_dataset(
    VAL_DATA_PATH
)


print(
    f"\nTraining examples: "
    f"{len(training_examples)}"
)

print(
    f"Validation examples: "
    f"{len(validation_examples)}"
)


# ============================================================
# 7. CREATE OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 8. TRAINING MODE
# ============================================================

model.train()


# ============================================================
# 9. TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    print(
        f"\n========== "
        f"EPOCH {epoch + 1}/{NUM_EPOCHS} "
        f"=========="
    )


    total_training_loss = 0.0


    # ========================================================
    # TRAINING
    # ========================================================

    for example_number, example in enumerate(
        training_examples,
        start=1
    ):

        # ----------------------------------------------------
        # Prepare example
        # ----------------------------------------------------

        (
            input_ids,
            attention_mask,
            labels
        ) = prepare_example(
            example,
            tokenizer,
            model.device
        )


        # ----------------------------------------------------
        # Clear gradients from previous step
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


        loss = outputs.loss


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update LoRA parameters
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Record loss
        # ----------------------------------------------------

        total_training_loss += loss.item()


        print(
            f"Example "
            f"{example_number}/"
            f"{len(training_examples)} "
            f"| Loss: "
            f"{loss.item():.4f}"
        )


    # ========================================================
    # AVERAGE TRAINING LOSS
    # ========================================================

    average_training_loss = (
        total_training_loss
        / len(training_examples)
    )


    print(
        f"\nEpoch {epoch + 1} "
        f"average training loss: "
        f"{average_training_loss:.4f}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    # Switch model into evaluation mode
    model.eval()


    total_validation_loss = 0.0


    # We don't need gradients during validation
    with torch.no_grad():

        for example in validation_examples:

            (
                input_ids,
                attention_mask,
                labels
            ) = prepare_example(
                example,
                tokenizer,
                model.device
            )


            # ----------------------------------------------
            # Forward pass ONLY
            # ----------------------------------------------

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )


            validation_loss = outputs.loss


            total_validation_loss += (
                validation_loss.item()
            )


    # ========================================================
    # AVERAGE VALIDATION LOSS
    # ========================================================

    average_validation_loss = (
        total_validation_loss
        / len(validation_examples)
    )


    print(
        f"Epoch {epoch + 1} "
        f"validation loss: "
        f"{average_validation_loss:.4f}"
    )


    # IMPORTANT:
    # Return to training mode for next epoch
    model.train()


# ============================================================
# 10. SAVE LoRA ADAPTER
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# 11. FINISHED
# ============================================================

print("\nTraining finished!")

print(
    f"LoRA adapter saved to:\n"
    f"{OUTPUT_DIR}"
)