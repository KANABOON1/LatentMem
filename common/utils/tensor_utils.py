import logging

from safetensors import safe_open
import torch
import torch.nn as nn

def erase_after_first_eos(completion_ids: torch.Tensor, pad_token_id: int, eos_token_id: int) -> torch.Tensor:
    is_eos_mask = (completion_ids == eos_token_id)
    first_eos_indices = torch.argmax(is_eos_mask.int(), dim=1)
    seq_len = completion_ids.size(1)
    col_indices = torch.arange(seq_len, device=completion_ids.device)
    mask_to_replace = (col_indices > first_eos_indices.unsqueeze(1)) & is_eos_mask.any(dim=1).unsqueeze(1)
    completion_ids[mask_to_replace] = pad_token_id

    return completion_ids

def fix_model_parameters(model: nn.Module):
    """Freeze all parameters of the given model.

    Args:
        model (nn.Module): The PyTorch model whose parameters will be frozen.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

def log_trainable_params(model: nn.Module):
    """Log all trainable parameters of the given model.

    Args:
        model (nn.Module): The PyTorch model to inspect.
    """
    logging.info("Trainable parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"  {name}: {param.numel()} params, shape={param.shape}")

def pad_batch_embeds(
    batch_inputs_embeds: list[torch.Tensor],
    batch_labels: list[torch.Tensor],
    batch_attention_masks: list[torch.Tensor],
    model_embedding_layer,
    pad_token_id: int,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    max_seq_len = max(emb.shape[1] for emb in batch_inputs_embeds)
    label_pad_value = -100
    mask_pad_value = 0

    pad_token_id = torch.tensor([[pad_token_id]], device=device)
    pad_embedding = model_embedding_layer(pad_token_id)
    
    padded_inputs_embeds = []
    padded_labels = []
    padded_attention_mask = []

    for current_embeds, current_labels, current_mask in zip(batch_inputs_embeds, batch_labels, batch_attention_masks):
        current_len = current_embeds.shape[1]
        padding_len = max_seq_len - current_len

        if padding_len > 0:
            embeds_padding = pad_embedding.repeat(1, padding_len, 1)
            padded_embeds = torch.cat([current_embeds, embeds_padding], dim=1)
            
            labels_padding = torch.full((1, padding_len), label_pad_value, dtype=torch.long, device=device)
            padded_label = torch.cat([current_labels, labels_padding], dim=1)
            
            mask_padding = torch.full((1, padding_len), mask_pad_value, dtype=torch.long, device=device)
            padded_mask = torch.cat([current_mask, mask_padding], dim=1)
        else:
            padded_embeds = current_embeds
            padded_label = current_labels
            padded_mask = current_mask

        padded_inputs_embeds.append(padded_embeds)
        padded_labels.append(padded_label)
        padded_attention_mask.append(padded_mask)
        
    final_inputs_embeds = torch.cat(padded_inputs_embeds, dim=0)
    final_labels = torch.cat(padded_labels, dim=0)
    final_attention_mask = torch.cat(padded_attention_mask, dim=0)
    
    return final_inputs_embeds, final_labels, final_attention_mask


def load_state_dict_from_safetensor(model_path) -> dict:
    """Load a safetensor file from the given path and return a state_dict.

    Args:
        model_path (str): Path to the safetensor file.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of model parameters, 
        where keys are parameter names and values are corresponding tensors.
    """
    model_state_dict = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            model_state_dict[key] = f.get_tensor(key)
    return model_state_dict