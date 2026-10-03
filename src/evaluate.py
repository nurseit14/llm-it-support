from pathlib import Path

from peft import PeftModel

from model import load_model


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "lora_adapter"
)


# ==================================================
# GENERATION FUNCTION
# ==================================================

def generate_answer(model, tokenizer, question):

    messages = [
        {
            "role": "system",
            "content": "You are a helpful IT support assistant."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer


# ==================================================
# TEST QUESTIONS
# ==================================================

questions = [
    "What is Docker?",

    "Why would a developer use Docker?",

    "Why should Python projects use virtual environments?",

    "What is the difference between training and inference?",

    "Why does a machine learning model need GPU memory?"
]


# ==================================================
# BASE MODEL
# ==================================================

print("\nLoading base model...")

base_model, tokenizer = load_model()

base_model.eval()


print("\nGenerating base model answers...")


base_answers = []


for question in questions:

    answer = generate_answer(
        base_model,
        tokenizer,
        question
    )

    base_answers.append(answer)


# ==================================================
# FINE-TUNED MODEL
# ==================================================

print("\nLoading LoRA adapter...")


fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

fine_tuned_model.eval()


print("\nGenerating fine-tuned answers...")


fine_tuned_answers = []


for question in questions:

    answer = generate_answer(
        fine_tuned_model,
        tokenizer,
        question
    )

    fine_tuned_answers.append(answer)


# ==================================================
# COMPARISON
# ==================================================

print("\n")
print("=" * 80)
print("BASE MODEL vs FINE-TUNED MODEL")
print("=" * 80)


for question, base_answer, tuned_answer in zip(
    questions,
    base_answers,
    fine_tuned_answers
):

    print("\nQUESTION:")
    print(question)

    print("\nBASE MODEL:")
    print(base_answer)

    print("\nFINE-TUNED MODEL:")
    print(tuned_answer)

    print("\n" + "-" * 80)
