import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from dataclasses import dataclass, field

import torch
import transformers
import json
from transformers import GenerationConfig, T5ForConditionalGeneration, T5Tokenizer
from typing import Optional

import sys


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="t5-small-email/checkpoint-576")


@dataclass
class InferenceArguments:
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
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
        torch_dtype=inference_args.inference_dtype,
        device_map="auto"
    )
    model.cuda()
    model.eval()

    generation_config = GenerationConfig(
        temperature=0.0,
    )

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        model_max_length=inference_args.model_max_length,
    )
    
    instruction = "Classify the provided emails, with optional categories including Marketing, Personal, and Updates.\n"
    while True:
        subject = input("Your email subject: ")
        content = input("Your email content: ")
        ctx = instruction + "Subject: " + subject + "\nContent:\n" + content
        inputs = tokenizer(ctx, return_tensors="pt")
        outputs = model.generate(
			      input_ids=inputs.input_ids.cuda(),
			      generation_config=generation_config,
			      max_new_tokens=32,
			      output_scores=False
		    )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("The email's label: ", decoded)


if __name__ == "__main__":
    inference()
