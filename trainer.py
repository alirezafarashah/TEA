import torch
from tqdm import tqdm
import wandb
import pandas as pd
import random
from train_utils import (
    train_discriminator,
    train_discriminator_pooled,
    train_encoder,
    train_encoder_pooled,
    get_original_embeddings,
    get_token_embeddings,
    tokenize_batch,
    cleanup_models,
    save_weights,
    save_lora_weights,
    get_original_token_embeddings,
)

from utils import generate_images_with_encoders 


def train_sequential(encoder_configs, dataloader, normal_dataloader, pipeline, harmful_to_safe_prompts, config):

    
    # Main training loop
    for epoch in range(config.num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH [{epoch+1}/{config.num_epochs}]")
        print(f"{'='*70}")
        
        # Train each encoder sequentially
        for idx, enc_config in enumerate(encoder_configs):
            print(f"\n{'='*70}")
            print(f"Training {enc_config['name']}")
            print(f"{'='*70}")
            
            # Move current encoder, its frozen reference, and discriminator to GPU.
            enc_config['encoder'].to(config.device)
            enc_config['original_encoder'].to(config.device)
            enc_config['discriminator'].to(config.device)
            
            # Training loop for this encoder
            normal_iter = iter(normal_dataloader)
            for batch_idx, (harmful_batch, safe_batch) in enumerate(
                tqdm(dataloader, desc=f"{enc_config['name']} Unlearning")
            ):
                
                try:
                    normal_batch = next(normal_iter)
                except StopIteration:
                    # If normal dataset runs out, restart it
                    normal_iter = iter(normal_dataloader)
                    normal_batch = next(normal_iter)
                    
                # Tokenize
                harmful_tokens = tokenize_batch(
                    enc_config['tokenizer'], 
                    harmful_batch, 
                    enc_config['max_length'],
                    enc_config['add_special_tokens'],
                    config.device
                )
                safe_tokens = tokenize_batch(
                    enc_config['tokenizer'], 
                    safe_batch, 
                    enc_config['max_length'],
                    enc_config['add_special_tokens'],
                    config.device
                )
                normal_tokens = tokenize_batch(
                    enc_config['tokenizer'], 
                    normal_batch, 
                    enc_config['max_length'],
                    enc_config['add_special_tokens'],
                    config.device
                )
                
                original_safe_emb = get_original_embeddings(
                    enc_config['original_encoder'], 
                    safe_tokens,
                    encoder_type=enc_config['encoder_type']
                )
                
                original_safe_emb_per_token = get_original_token_embeddings(
                    enc_config['original_encoder'], 
                    safe_tokens,
                    encoder_type=enc_config['encoder_type']
                )
                
                original_normal_emb = get_original_embeddings(
                    enc_config['original_encoder'], 
                    normal_tokens,
                    encoder_type=enc_config['encoder_type']
                )
                
                # Train discriminator
                if config.pooled_embed:
                    # print("Training discriminator with pooled embeddings...")
                    loss_disc, accuracy_disc = train_discriminator_pooled(
                        enc_config['encoder'],
                        enc_config['discriminator'],
                        enc_config['optimizer_disc'],
                        harmful_tokens,
                        safe_tokens,
                        safe_emb=original_safe_emb if config.use_fixed_target_emb else None,  # Use original safe embeddings for discriminator training
                        encoder_type=enc_config['encoder_type']
                    )
                else:
                    loss_disc, accuracy_disc = train_discriminator(
                        enc_config['encoder'],
                        enc_config['discriminator'],
                        enc_config['optimizer_disc'],
                        harmful_tokens,
                        safe_tokens,
                        safe_emb=original_safe_emb_per_token if config.use_fixed_target_emb else None,  # Use original safe embeddings for discriminator training
                        encoder_type=enc_config['encoder_type']
                    )
                
                # Train encoder (EMA updated inside if enabled)
                if config.pooled_embed:
                    loss_adv, loss_preserve = train_encoder_pooled(
                        enc_config['encoder'],
                        enc_config['discriminator'],
                        enc_config['optimizer_enc'],
                        harmful_tokens,
                        safe_tokens,
                        normal_tokens,
                        original_safe_emb,
                        original_normal_emb,
                        config.lambda_adv,
                        config.lambda_preserve,
                        ema=enc_config.get('ema', None),
                        encoder_type=enc_config['encoder_type'],
                        preservation_type=config.preservation_type,
                        kl_temperature=config.kl_temperature,
                    )
                else:
                    loss_adv, loss_preserve = train_encoder(
                        enc_config['encoder'],
                        enc_config['discriminator'],
                        enc_config['optimizer_enc'],
                        harmful_tokens,
                        safe_tokens,
                        normal_tokens,
                        original_safe_emb_per_token,
                        original_normal_emb,
                        config.lambda_adv,
                        config.lambda_preserve,
                        ema=enc_config.get('ema', None),
                        encoder_type=enc_config['encoder_type'],
                        preservation_type=config.preservation_type,
                        kl_temperature=config.kl_temperature,
                    )
                
                if config.use_wandb:
                    prefix = enc_config['name'].replace(" ", "_")
                    wandb.log({
                        f"{prefix}/discriminator_loss": loss_disc.item(),
                        f"{prefix}/discriminator_acc": accuracy_disc.item(),
                        f"{prefix}/adversarial_loss": loss_adv.item(),
                        f"{prefix}/preservation_loss": loss_preserve.item(),
                        # f"{prefix}/preservation_loss_normal": loss_preserve_normal.item(),
                        "epoch": epoch,
                        "global_step": epoch * len(dataloader) + batch_idx
                    })
            
            print(f"{enc_config['name']} - Disc Loss: {loss_disc.item():.4f} | "
                  f"Acc: {accuracy_disc.item():.4f} | "
                  f"Adv Loss: {loss_adv.item():.4f} | "
                  f"Preserve: {loss_preserve.item():.4f} | "
                #   f"Preserve Normal: {loss_preserve_normal.item():.4f}"
                  )
            

            cleanup_models(enc_config['encoder'], enc_config['discriminator'])
        
        # Image generation every N epochs
        if (epoch + 1) % config.generate_every_n_epochs == 0:
            print(f"\n{'='*70}")
            print(f"Generating Images for Epoch {epoch+1}")
            print(f"{'='*70}")
            nudity_dataset = pd.read_csv("./datasets/nudity-ring-a-bell.csv") 
            sample_harmful_prompts = nudity_dataset['sensitive prompt'].iloc[[0, 3, 7, 36, 42]].tolist()
            # sample_harmful_prompts = list(harmful_to_safe_prompts.keys())[:config.num_sample_prompts]
            # Move all encoders to GPU for generation
            for enc_config in encoder_configs:
                enc_config['original_encoder'].to(config.device)
                enc_config['encoder'].to(config.device)
                enc_config['original_encoder'].eval()
                enc_config['encoder'].eval()
            
            # Generate with original encoders
            if epoch==0:
                generate_images_with_encoders(
                    pipeline, 
                    sample_harmful_prompts,
                    encoder_configs[0]['original_encoder'],
                    encoder_configs[1]['original_encoder'],
                    encoder_configs[2]['original_encoder'],
                    "original", 
                    epoch,
                    config,
                    config.device
                )
            
            # Generate with trained encoders (regular weights)
            generate_images_with_encoders(
                pipeline,
                sample_harmful_prompts,
                encoder_configs[0]['encoder'],
                encoder_configs[1]['encoder'],
                encoder_configs[2]['encoder'],
                "trained",
                epoch,
                config,
                config.device
            )
            
            # Generate with EMA weights if enabled
            if config.use_ema:
                # Apply EMA weights
                for enc_config in encoder_configs:
                    if enc_config['ema'] is not None:
                        print("load EMA")
                        enc_config['ema'].apply_shadow(enc_config['encoder'])
                
                generate_images_with_encoders(
                    pipeline,
                    sample_harmful_prompts,
                    encoder_configs[0]['encoder'],
                    encoder_configs[1]['encoder'],
                    encoder_configs[2]['encoder'],
                    "trained_ema",
                    epoch,
                    config,
                    config.device
                )
                
         
                for enc_config in encoder_configs:
                    if enc_config['ema'] is not None:
                        enc_config['ema'].restore(enc_config['encoder'])
            
            print(f"Generated and saved images for epoch {epoch+1}.")
            
       
            for enc_config in encoder_configs:
                cleanup_models(enc_config['original_encoder'], enc_config['encoder'])
            
            # Save models
            save_weights(encoder_configs, config.save_model_dir, epoch, use_lora=config.use_lora)
    
    print("\n" + "="*70)
    print("Sequential training with preservation completed!")
    print("="*70)
    
    # Final cleanup
    for enc_config in encoder_configs:
        cleanup_models(enc_config['encoder'], enc_config['discriminator'])


def train_single_encoder(enc_config, dataloader, normal_dataloader, config):
    import os
    slot = enc_config['slot_idx']

    enc_config['encoder'].to(config.device)
    enc_config['original_encoder'].to(config.device)
    enc_config['discriminator'].to(config.device)

    for epoch in range(config.num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH [{epoch+1}/{config.num_epochs}]  —  {enc_config['name']}")
        print(f"{'='*70}")

        normal_iter = iter(normal_dataloader)
        for batch_idx, (harmful_batch, safe_batch) in enumerate(
            tqdm(dataloader, desc=f"{enc_config['name']} Unlearning")
        ):
            try:
                normal_batch = next(normal_iter)
            except StopIteration:
                normal_iter = iter(normal_dataloader)
                normal_batch = next(normal_iter)

            harmful_tokens = tokenize_batch(enc_config['tokenizer'], harmful_batch, enc_config['max_length'], enc_config['add_special_tokens'], config.device)
            safe_tokens    = tokenize_batch(enc_config['tokenizer'], safe_batch,    enc_config['max_length'], enc_config['add_special_tokens'], config.device)
            normal_tokens  = tokenize_batch(enc_config['tokenizer'], normal_batch,  enc_config['max_length'], enc_config['add_special_tokens'], config.device)

            original_safe_emb           = get_original_embeddings(enc_config['original_encoder'], safe_tokens,   encoder_type=enc_config['encoder_type'])
            original_safe_emb_per_token = get_original_token_embeddings(enc_config['original_encoder'], safe_tokens, encoder_type=enc_config['encoder_type'])
            original_normal_emb         = get_original_embeddings(enc_config['original_encoder'], normal_tokens, encoder_type=enc_config['encoder_type'])

            if config.pooled_embed:
                loss_disc, accuracy_disc = train_discriminator_pooled(enc_config['encoder'], enc_config['discriminator'], enc_config['optimizer_disc'], harmful_tokens, safe_tokens, safe_emb=original_safe_emb if config.use_fixed_target_emb else None, encoder_type=enc_config['encoder_type'])
                loss_adv, loss_preserve  = train_encoder_pooled(enc_config['encoder'], enc_config['discriminator'], enc_config['optimizer_enc'], harmful_tokens, safe_tokens, normal_tokens, original_safe_emb, original_normal_emb, config.lambda_adv, config.lambda_preserve, ema=enc_config.get('ema'), encoder_type=enc_config['encoder_type'], preservation_type=config.preservation_type, kl_temperature=config.kl_temperature)
            else:
                loss_disc, accuracy_disc = train_discriminator(enc_config['encoder'], enc_config['discriminator'], enc_config['optimizer_disc'], harmful_tokens, safe_tokens, safe_emb=original_safe_emb_per_token if config.use_fixed_target_emb else None, encoder_type=enc_config['encoder_type'])
                loss_adv, loss_preserve  = train_encoder(enc_config['encoder'], enc_config['discriminator'], enc_config['optimizer_enc'], harmful_tokens, safe_tokens, normal_tokens, original_safe_emb_per_token, original_normal_emb, config.lambda_adv, config.lambda_preserve, ema=enc_config.get('ema'), encoder_type=enc_config['encoder_type'], preservation_type=config.preservation_type, kl_temperature=config.kl_temperature)

            if config.use_wandb:
                prefix = enc_config['name'].replace(" ", "_")
                wandb.log({f"{prefix}/discriminator_loss": loss_disc.item(), f"{prefix}/discriminator_acc": accuracy_disc.item(), f"{prefix}/adversarial_loss": loss_adv.item(), f"{prefix}/preservation_loss": loss_preserve.item(), "epoch": epoch, "global_step": epoch * len(dataloader) + batch_idx})

        print(f"{enc_config['name']} - Disc Loss: {loss_disc.item():.4f} | Acc: {accuracy_disc.item():.4f} | Adv Loss: {loss_adv.item():.4f} | Preserve: {loss_preserve.item():.4f}")

        save_weights([{**enc_config, 'slot_idx': slot}], config.save_model_dir, epoch, use_lora=config.use_lora)

    cleanup_models(enc_config['encoder'], enc_config['discriminator'], enc_config['original_encoder'])