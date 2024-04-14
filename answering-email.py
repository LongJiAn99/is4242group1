import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
import json
from transformers import GenerationConfig, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Optional, Sequence


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="t5-small-email/checkpoint-1440")


@dataclass
class InferenceArguments:
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
      )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    inference_dtype: torch.dtype = field(
        default=torch.bfloat16,
        metadata={"help": "The dtype to use for inference."},
    )


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.cuda()
    model.eval()

    generation_config = GenerationConfig(
        temperature=0.0,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        model_max_length=inference_args.model_max_length,
    )

    with open("data/email_testing_data.json", "r") as f:
        questions = json.load(f)
    
    results = []
    for question in questions:
        ctx = "Classify the provided emails, with optional categories including Social, Promotions, Primary, Updates, Purchases.\nSubject: " + question["Subject"] + "\nContent:\n" + question["Content"]
        inputs = tokenizer(ctx, return_tensors="pt")
        outputs = model.generate(
			input_ids=inputs.input_ids.cuda(),
			generation_config=generation_config,
			max_new_tokens=32,
			output_scores=False
		)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Question: " + question["Subject"])
        print("Answer: " + decoded)
        print("------------------------------------------")
        results.append({"subject": question["Subject"], "content": question["Content"], "category": question["Category"], "response": decoded})
    
    with open("answers/email-t5-small-e10.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    inference()