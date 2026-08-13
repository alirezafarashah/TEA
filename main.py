import torch
import os
from copy import deepcopy
import json
from torch.utils.data import DataLoader
from diffusers import StableDiffusion3Pipeline

from config import parse_args
from data_utils import vangogh_style_to_safe_prompts, kelly_mckernan_style_to_safe_prompts, harmful_to_safe_prompts, normal_prompts, PromptPairDatasetFromDict, PromptListDataset
from train_utils import Discriminator, apply_lora_to_encoders, EMA, count_trainable_parameters
from trainer import train_sequential, train_single_encoder

import wandb

def freeze_pipeline_components(pipeline):
    for key, component in pipeline.__dict__.items():
        if isinstance(component, torch.nn.Module):
            for param in component.parameters():
                param.requires_grad = False


def setup_encoders(pipeline, device, clip_only=False):
    if clip_only:
        # Encoders are already on GPU; deepcopy in-place.
        original_text_encoder   = pipeline.text_encoder.eval()
        original_text_encoder_2 = pipeline.text_encoder_2.eval()
        original_text_encoder_3 = pipeline.text_encoder_3.eval()

        text_encoder   = deepcopy(original_text_encoder).train()
        text_encoder_2 = deepcopy(original_text_encoder_2).train()
        text_encoder_3 = deepcopy(original_text_encoder_3).train()
    else:
        # Move to CPU before deepcopy to avoid two T5 copies on GPU at once.
        original_text_encoder   = pipeline.text_encoder.cpu().eval()
        original_text_encoder_2 = pipeline.text_encoder_2.cpu().eval()
        original_text_encoder_3 = pipeline.text_encoder_3.cpu().eval()

        text_encoder   = deepcopy(original_text_encoder).train()
        text_encoder_2 = deepcopy(original_text_encoder_2).train()
        text_encoder_3 = deepcopy(original_text_encoder_3).train()

        # Keep T5 in bfloat16 to halve its CPU/GPU footprint.
        text_encoder_3           = text_encoder_3.to(torch.bfloat16)
        original_text_encoder_3  = original_text_encoder_3.to(torch.bfloat16)

    # Enable gradients on trainable copies
    for param in text_encoder.parameters():
        param.requires_grad = True
    for param in text_encoder_2.parameters():
        param.requires_grad = True
    for param in text_encoder_3.parameters():
        param.requires_grad = True

    return (text_encoder, text_encoder_2, text_encoder_3,
            original_text_encoder, original_text_encoder_2, original_text_encoder_3)


def main():
    config = parse_args()
    
    # Expand cache directory
    cache_dir = os.path.expanduser(config.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.save_model_dir, exist_ok=True)
    
    print(f"Loading model from {config.model_name}...")

    # CLIP encoders (idx 1/2) are small enough that the full pipeline fits on
    # GPU — load directly onto CUDA to avoid RAM pressure on 32 GB machines.
    # T5 (idx 3) is large, so load on CPU and keep transformer/VAE off GPU.
    clip_only = config.encoder_idx in (1, 2)
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        device_map="cuda" if clip_only else None,
    )
    if not clip_only:
        # Keep transformer and VAE on CPU; they are only needed at generation time.
        pipeline.transformer.to("cpu")
        pipeline.vae.to("cpu")
    
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.wandb_entity,
            config=config.__dict__ 
        )
    
    # Freeze all pipeline components
    freeze_pipeline_components(pipeline)
    
    # Setup encoders
    (text_encoder, text_encoder_2, text_encoder_3,
     original_text_encoder, original_text_encoder_2, original_text_encoder_3) = setup_encoders(
        pipeline, config.device, clip_only=clip_only
    )
    
    # Apply LoRA or report full fine-tuning parameter counts
    if config.use_lora:
        text_encoder, text_encoder_2, text_encoder_3 = apply_lora_to_encoders(
            text_encoder, text_encoder_2, text_encoder_3, config
        )
    else:
        n1 = count_trainable_parameters(text_encoder)
        n2 = count_trainable_parameters(text_encoder_2)
        n3 = count_trainable_parameters(text_encoder_3)
        print(f"Full fine-tuning: encoder1={n1:,}, encoder2={n2:,}, encoder3={n3:,} trainable parameters.")
    
    # Initialize EMA if enabled
    ema_1 = EMA(text_encoder, decay=config.ema_decay) if config.use_ema else None
    ema_2 = EMA(text_encoder_2, decay=config.ema_decay) if config.use_ema else None
    ema_3 = EMA(text_encoder_3, decay=config.ema_decay) if config.use_ema else None
    
    if config.use_ema:
        print(f"EMA initialized with decay={config.ema_decay}")
    
    # Create discriminators
    discriminator_1 = Discriminator(text_encoder.config.hidden_size).to(config.device)
    discriminator_2 = Discriminator(text_encoder_2.config.hidden_size).to(config.device)
    discriminator_3 = Discriminator(text_encoder_3.config.hidden_size).to(config.device)
    
    # Get tokenizers
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    tokenizer_3 = pipeline.tokenizer_3
    
    # Create encoder configurations
    encoder_configs = [
        {
            'name': 'Encoder 1',
            'encoder': text_encoder,
            'original_encoder': original_text_encoder,
            'discriminator': discriminator_1,
            'tokenizer': tokenizer,
            'ema': ema_1,
            'optimizer_enc': torch.optim.AdamW(
                (p for p in text_encoder.parameters() if p.requires_grad), 
                lr=config.lr_encoder1
            ),
            'optimizer_disc': torch.optim.AdamW(discriminator_1.parameters(), lr=config.lr_disc),
            'max_length': 77,
            'add_special_tokens': False,
            'encoder_type': 'clip'
        },
        {
            'name': 'Encoder 2',
            'encoder': text_encoder_2,
            'original_encoder': original_text_encoder_2,
            'discriminator': discriminator_2,
            'tokenizer': tokenizer_2,
            'ema': ema_2,
            'optimizer_enc': torch.optim.AdamW(
                (p for p in text_encoder_2.parameters() if p.requires_grad), 
                lr=config.lr_encoder2
            ),
            'optimizer_disc': torch.optim.AdamW(discriminator_2.parameters(), lr=config.lr_disc),
            'max_length': 77,
            'add_special_tokens': False,
            'encoder_type': 'clip'
        },
        {
            'name': 'Encoder 3',
            'encoder': text_encoder_3,
            'original_encoder': original_text_encoder_3,
            'discriminator': discriminator_3,
            'tokenizer': tokenizer_3,
            'ema': ema_3,
            'optimizer_enc': torch.optim.AdamW(
                (p for p in text_encoder_3.parameters() if p.requires_grad), 
                lr=config.lr_encoder3
            ),
            'optimizer_disc': torch.optim.AdamW(discriminator_3.parameters(), lr=config.lr_disc),
            'max_length': 77,
            'add_special_tokens': True,
            'encoder_type': 't5'
        }
    ]
    
    # Move all models to CPU initially
    for enc_config in encoder_configs:
        enc_config['original_encoder'] = enc_config['original_encoder'].cpu()
        enc_config['encoder'] = enc_config['encoder'].cpu()
        enc_config['discriminator'] = enc_config['discriminator'].cpu()
    
    print("All models moved to CPU, ready for sequential training.")
    
    # Create dataset and dataloader
    if config.dataset == "vangogh_style_to_safe":
        dataset = PromptPairDatasetFromDict(vangogh_style_to_safe_prompts)
        prompts = vangogh_style_to_safe_prompts
    elif config.dataset == "harmful_to_safe":
        dataset = PromptPairDatasetFromDict(harmful_to_safe_prompts)
        prompts = harmful_to_safe_prompts
    elif config.dataset == "kelly_mckernan_style_to_safe_prompts":
        dataset = PromptPairDatasetFromDict(kelly_mckernan_style_to_safe_prompts)
        prompts = kelly_mckernan_style_to_safe_prompts
    else:
        raise ValueError(f"Unknown dataset choice: {config.dataset}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    normal_dataset = PromptListDataset(normal_prompts)
    
    normal_dataloader = DataLoader(normal_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # Single-encoder mode: train only the requested encoder
    if config.encoder_idx != 0:
        enc_config = encoder_configs[config.encoder_idx - 1]
        enc_config['slot_idx'] = config.encoder_idx
        train_single_encoder(enc_config, dataloader, normal_dataloader, config)
    else:
        train_sequential(encoder_configs, dataloader, normal_dataloader, pipeline, prompts, config)


if __name__ == "__main__":
    main()