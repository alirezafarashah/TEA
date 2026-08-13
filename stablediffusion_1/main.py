import json
import os
from copy import deepcopy

import torch
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader

import wandb

from stablediffusion_1.config import parse_args
from stablediffusion_1.data_utils import (
    PromptPairDatasetFromDict,
    harmful_to_safe_prompts,
    kelly_mckernan_style_to_safe_prompts,
    vangogh_style_to_safe_prompts,
)
from stablediffusion_1.train_utils import Discriminator, EMA, apply_lora_to_encoder, cleanup_models, count_trainable_parameters
from stablediffusion_1.trainer import train


def freeze_pipeline_components(pipeline):
    for _, component in pipeline.__dict__.items():
        if isinstance(component, torch.nn.Module):
            for param in component.parameters():
                param.requires_grad = False


def setup_encoder(pipeline, device):
    original_text_encoder = pipeline.text_encoder
    original_text_encoder.eval().to(device)

    text_encoder = deepcopy(original_text_encoder).to(device).train()
    for param in text_encoder.parameters():
        param.requires_grad = True

    return text_encoder, original_text_encoder


def main():
    config = parse_args()

    cache_dir = os.path.expanduser(config.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.save_model_dir, exist_ok=True)

    print(f"Loading SD1.4 model from {config.model_name} (fp32)...")

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(config.device)

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=config.wandb_entity,
            config=config.__dict__,
        )

    freeze_pipeline_components(pipeline)

    text_encoder, original_text_encoder = setup_encoder(pipeline, config.device)
    if config.use_lora:
        text_encoder = apply_lora_to_encoder(text_encoder, config)
    else:
        n_params = count_trainable_parameters(text_encoder)
        print(f"Full fine-tuning: {n_params:,} trainable parameters in text encoder.")

    ema = EMA(text_encoder, decay=config.ema_decay) if config.use_ema else None
    if config.use_ema:
        print(f"EMA initialized with decay={config.ema_decay}")

    discriminator = Discriminator(text_encoder.config.hidden_size).to(config.device)
    tokenizer = pipeline.tokenizer

    train_state = {
        "name": "Text Encoder",
        "encoder": text_encoder.cpu(),
        "original_encoder": original_text_encoder.cpu(),
        "discriminator": discriminator.cpu(),
        "tokenizer": tokenizer,
        "ema": ema,
        "optimizer_enc": torch.optim.AdamW((p for p in text_encoder.parameters() if p.requires_grad), lr=config.lr_encoder),
        "optimizer_disc": torch.optim.AdamW(discriminator.parameters(), lr=config.lr_disc),
        "max_length": 77,
        "add_special_tokens": False,
    }

    print("Models moved to CPU, ready for training.")

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

    train(dataloader, pipeline, prompts, config, train_state)

    cleanup_models(train_state["encoder"], train_state["discriminator"], train_state["original_encoder"])


if __name__ == "__main__":
    main()
