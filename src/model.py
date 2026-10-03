from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    return tokenizer


def load_model():

    tokenizer = load_tokenizer()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    return model, tokenizer


if __name__ == "__main__":

    model, tokenizer = load_model()

    print("Model loaded successfully!")

    print(
        "Model:",
        type(model)
    )

    print(
        "Tokenizer:",
        type(tokenizer)
    )

    print(
        "Device:",
        model.device
    )