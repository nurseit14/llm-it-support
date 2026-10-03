from pathlib import Path

import torch

from torch.utils.data import DataLoader

from peft import (
    LoraConfig,
    get_peft_model
)

from model import load_model

from dataset import (
    ITSupportDataset,
    create_collate_fn
)


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

BATCH_SIZE = 4

MAX_LENGTH = 512


# ============================================================
# 2. LOAD BASE MODEL
# ============================================================

model, tokenizer = load_model()


print("Base model loaded.")

print(
    "Device:",
    model.device
)


# ============================================================
# 3. LoRA CONFIGURATION
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


# ============================================================
# 4. ATTACH LoRA
# ============================================================

model = get_peft_model(
    model,
    lora_config
)


print("\nLoRA attached:")

model.print_trainable_parameters()


# ============================================================
# 5. CREATE DATASETS
# ============================================================

train_dataset = ITSupportDataset(

    TRAIN_DATA_PATH,

    tokenizer,

    max_length=MAX_LENGTH
)


validation_dataset = ITSupportDataset(

    VAL_DATA_PATH,

    tokenizer,

    max_length=MAX_LENGTH
)


print(
    f"\nTraining examples: "
    f"{len(train_dataset)}"
)


print(
    f"Validation examples: "
    f"{len(validation_dataset)}"
)


# ============================================================
# 6. CREATE COLLATE FUNCTION
# ============================================================

collate_fn = create_collate_fn(
    tokenizer
)


# ============================================================
# 7. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn
)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_fn
)


print(
    f"Training batches: "
    f"{len(train_loader)}"
)


print(
    f"Validation batches: "
    f"{len(validation_loader)}"
)


# ============================================================
# 8. CREATE OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE
)


# ============================================================
# 9. TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):


    print(
        f"\n========== "
        f"EPOCH {epoch + 1}/{NUM_EPOCHS} "
        f"=========="
    )


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()


    total_training_loss = 0.0


    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):


        # ----------------------------------------------------
        # Move batch to same device as model
        # ----------------------------------------------------

        input_ids = (
            batch["input_ids"]
            .to(model.device)
        )


        attention_mask = (
            batch["attention_mask"]
            .to(model.device)
        )


        labels = (
            batch["labels"]
            .to(model.device)
        )


        # ----------------------------------------------------
        # Clear gradients
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

        total_training_loss += (
            loss.item()
        )


        print(
            f"Batch "
            f"{batch_number}/"
            f"{len(train_loader)} "
            f"| Loss: "
            f"{loss.item():.4f}"
        )


    # ========================================================
    # AVERAGE TRAINING LOSS
    # ========================================================

    average_training_loss = (

        total_training_loss

        / len(train_loader)
    )


    print(
        f"\nEpoch {epoch + 1} "
        f"average training loss: "
        f"{average_training_loss:.4f}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    total_validation_loss = 0.0


    with torch.no_grad():


        for batch in validation_loader:


            # ------------------------------------------------
            # Move batch to device
            # ------------------------------------------------

            input_ids = (
                batch["input_ids"]
                .to(model.device)
            )


            attention_mask = (
                batch["attention_mask"]
                .to(model.device)
            )


            labels = (
                batch["labels"]
                .to(model.device)
            )


            # ------------------------------------------------
            # Forward pass only
            # ------------------------------------------------

            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                labels=labels
            )


            loss = outputs.loss


            total_validation_loss += (
                loss.item()
            )


    # ========================================================
    # AVERAGE VALIDATION LOSS
    # ========================================================

    average_validation_loss = (

        total_validation_loss

        / len(validation_loader)
    )


    print(
        f"Epoch {epoch + 1} "
        f"validation loss: "
        f"{average_validation_loss:.4f}"
    )


# ============================================================
# 10. SAVE ADAPTER
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

print(
    "\nTraining finished!"
)


print(
    f"LoRA adapter saved to:\n"
    f"{OUTPUT_DIR}"
)