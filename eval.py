import argparse
import torch
import time
from transformers import AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from peft import AutoPeftModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    batch_size = args.batch_size

    tokenizer = AutoTokenizer.from_pretrained("./gda-qa-lora/checkpoint-294")

    model = AutoPeftModelForCausalLM.from_pretrained(
        "./gda-qa-lora/checkpoint-294",
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )

    model = torch.compile(model)
    ds: Dataset = load_from_disk("./dataset")
    dataloader = DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    start = time.time()
    with torch.no_grad(): 
        for i, batch in enumerate(dataloader):
            messages = [
                {"role": "user", "content": m} for m in batch['data']
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs, max_new_tokens=40)
            end = time.time()
            print(f"{(batch_size * i) / (end - start)}")


if __name__ == "__main__":
    main()
