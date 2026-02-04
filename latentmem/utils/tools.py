import glob
import os
import shutil

import torch
from trl import SFTTrainer

# from https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja
CONVERSATION_TEMPLATE = r"""
{# ───── main loop ───── #}
{%- for message in messages -%}
    {%- set content = message.content if message.content is string else "" -%}
    {%- if (message.role == "user") or (message.role == "system") -%}
        {{ "<|im_start|>" + message.role + "\n"  + content + "<|im_end|>\n" }}
    {%- elif message.role == "assistant" -%}
        {%- generation -%}
        {{ "<|im_start|>assistant\n" + content + "<|im_end|>\n" }}
        {%- endgeneration -%}
    {%- elif message.role == "tool" -%}
    {{ "<|im_start|>" + "user\n"  + content + "<|im_end|>\n" }}
    {%- endif -%}
{%- endfor -%}
{# ───── generation prompt ───── #}
{%- if add_generation_prompt -%}
    {{ "<|im_start|>assistant\n" }}
{%- endif -%}
""".strip()

def remove_trainer_ckpts(output_dir: str):
    ckpt_paths = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for ckpt in ckpt_paths:
        shutil.rmtree(ckpt, ignore_errors=True)

def postprocess_assistant_labels(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer
) -> torch.Tensor:
    if tokenizer.chat_template != CONVERSATION_TEMPLATE:
        raise ValueError(
            "Invalid tokenizer.chat_template detected.\n"
            f"Expected:\n{CONVERSATION_TEMPLATE}\n\n"
            f"Got:\n{tokenizer.chat_template}\n\n"
            "Please ensure that you are using the correct conversation template."
        )
    
    # Encode the token sequence for "<|im_start|>assistant\n"
    pattern_ids: list[int] = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    batch_size, seq_len = input_ids.shape
    new_labels = labels.clone()

    for b in range(batch_size):
        seq = input_ids[b].tolist()
        for i in range(len(seq) - len(pattern_ids) + 1):
            # Mask positions matching the pattern
            if seq[i : i + len(pattern_ids)] == pattern_ids:
                new_labels[b, i : i + len(pattern_ids)] = -100

    return new_labels