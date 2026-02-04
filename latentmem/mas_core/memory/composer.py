import logging
from typing import Optional

from peft import PeftConfig, get_peft_model
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from latentmem.utils.tools import CONVERSATION_TEMPLATE

class Composer(torch.nn.Module):

    def __init__(
        self, 
        pretrained_model_name_or_path: str, 
        latents_len: int,
        peft_config: Optional[PeftConfig] = None
    ):
        super().__init__()
        
        # base model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        # self.model.gradient_checkpointing_enable()  
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if self.tokenizer.chat_template is None:
            logging.info("The model does not have a chat template; replace the model's chat template.")
            self.tokenizer.chat_template = CONVERSATION_TEMPLATE
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token is None:
            logging.info("The model does not have a pad token, so the eos token is used as the pad token instead.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if peft_config is not None:
            self.model = get_peft_model(self.model, peft_config)
        self.tokenizer.padding_side = 'left' 

        # prompt augmentation
        self.query_latents = nn.Parameter(
            torch.randn(latents_len, self.model.config.hidden_size), 
            requires_grad=True
        )
        
        # update weaver configs
        self.config = self.model.config
        self.config.latents_len = latents_len
        
    @property
    def device(self):
        return self.model.device
    
    def text_to_latent(self, texts: list[str]) -> torch.FloatTensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        batch_size = input_ids.size(0)
        latents = self.query_latents.unsqueeze(0).expand(batch_size, -1, -1)
        latents_len = self.config.latents_len
        latents_mask = torch.ones(
            (batch_size, latents_len),
            dtype=attention_mask.dtype,
            device=self.device
        )

        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_len:, :] 

        return latents_hidden_states


