import pandas as pd
import wandb
from tqdm import tqdm

from stablediffusion_1.train_utils import (
    cleanup_models,
    generate_images_with_text_encoder,
    get_original_embeddings,
    get_original_token_embeddings,
    save_weights,
    tokenize_batch,
    train_discriminator,
    train_discriminator_pooled,
    train_encoder,
    train_encoder_pooled,
)


def train(dataloader, pipeline, prompts_dict, config, train_state):
    encoder = train_state["encoder"]
    original_encoder = train_state["original_encoder"]
    discriminator = train_state["discriminator"]
    tokenizer = train_state["tokenizer"]
    ema = train_state.get("ema", None)
    optimizer_enc = train_state["optimizer_enc"]
    optimizer_disc = train_state["optimizer_disc"]

    max_length = train_state.get("max_length", 77)
    add_special_tokens = train_state.get("add_special_tokens", False)

    for epoch in range(config.num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH [{epoch+1}/{config.num_epochs}]")
        print(f"{'='*70}")

        encoder.to(config.device)
        discriminator.to(config.device)

        for batch_idx, (harmful_batch, safe_batch) in enumerate(tqdm(dataloader, desc="SD1.4 Unlearning")):
            harmful_tokens = tokenize_batch(
                tokenizer, harmful_batch, max_length, add_special_tokens, device=config.device
            )
            safe_tokens = tokenize_batch(tokenizer, safe_batch, max_length, add_special_tokens, device=config.device)

            original_safe_emb = get_original_embeddings(original_encoder, safe_tokens)
            original_safe_emb_per_token = get_original_token_embeddings(original_encoder, safe_tokens)

            if config.pooled_embed:
                loss_disc, accuracy_disc = train_discriminator_pooled(
                    encoder,
                    discriminator,
                    optimizer_disc,
                    harmful_tokens,
                    safe_tokens,
                    safe_emb=original_safe_emb if config.use_fixed_target_emb else None,
                )
                loss_adv, loss_preserve = train_encoder_pooled(
                    encoder,
                    discriminator,
                    optimizer_enc,
                    harmful_tokens,
                    safe_tokens,
                    original_safe_emb_per_token,
                    config.lambda_adv,
                    config.lambda_preserve,
                    ema=ema,
                    preservation_type=config.preservation_type,
                    kl_temperature=config.kl_temperature,
                )
            else:
                loss_disc, accuracy_disc = train_discriminator(
                    encoder,
                    discriminator,
                    optimizer_disc,
                    harmful_tokens,
                    safe_tokens,
                    safe_emb=original_safe_emb_per_token if config.use_fixed_target_emb else None,
                )
                loss_adv, loss_preserve = train_encoder(
                    encoder,
                    discriminator,
                    optimizer_enc,
                    harmful_tokens,
                    safe_tokens,
                    original_safe_emb_per_token,
                    config.lambda_adv,
                    config.lambda_preserve,
                    ema=ema,
                    preservation_type=config.preservation_type,
                    kl_temperature=config.kl_temperature,
                )

            if config.use_wandb:
                wandb.log(
                    {
                        "discriminator_loss": loss_disc.item(),
                        "discriminator_acc": accuracy_disc.item(),
                        "adversarial_loss": loss_adv.item(),
                        "preservation_loss": loss_preserve.item(),
                        "epoch": epoch,
                        "global_step": epoch * len(dataloader) + batch_idx,
                    }
                )

        print(
            f"Epoch {epoch+1} - Disc Loss: {loss_disc.item():.4f} | "
            f"Acc: {accuracy_disc.item():.4f} | "
            f"Adv Loss: {loss_adv.item():.4f} | "
            f"Preserve: {loss_preserve.item():.4f}"
        )

        cleanup_models(encoder, discriminator)

        if (epoch + 1) % config.generate_every_n_epochs == 0:
            print(f"\n{'='*70}")
            print(f"Generating Images for Epoch {epoch+1}")
            print(f"{'='*70}")

            nudity_dataset = pd.read_csv("./datasets/nudity-ring-a-bell.csv")
            sample_harmful_prompts = nudity_dataset["sensitive prompt"].iloc[[0, 3, 7, 36, 42]].tolist()

            sample_coco_prompts = [
                "A cat sitting on a couch with a laptop in front of it.",
                "A person walking down a street while holding an umbrella.",
                "A man and a woman on a sidewalk standing in front of several suitcases.",
                "A cow on a mountaintop standing in the grass.",
                "A brown dog sitting with a man on a porch swing.",
            ]

            original_encoder.to(config.device).eval()
            encoder.to(config.device).eval()

            if epoch == 0:
                generate_images_with_text_encoder(
                    pipeline, sample_harmful_prompts, original_encoder, "original", epoch, config, config.device
                )
                generate_images_with_text_encoder(
                    pipeline, sample_coco_prompts, original_encoder, "original_coco", epoch, config, config.device
                )

            generate_images_with_text_encoder(
                pipeline, sample_harmful_prompts, encoder, "trained", epoch, config, config.device
            )
            generate_images_with_text_encoder(
                pipeline, sample_coco_prompts, encoder, "trained_coco", epoch, config, config.device
            )

            if ema is not None:
                ema.apply_shadow(encoder)
                generate_images_with_text_encoder(
                    pipeline, sample_harmful_prompts, encoder, "trained_ema", epoch, config, config.device
                )
                generate_images_with_text_encoder(
                    pipeline, sample_coco_prompts, encoder, "trained_ema_coco", epoch, config, config.device
                )
                ema.restore(encoder)

            cleanup_models(original_encoder, encoder)

            save_weights(encoder, config.save_model_dir, epoch, ema=ema, use_lora=config.use_lora)

    print("\n" + "=" * 70)
    print("SD1.4 training with preservation completed!")
    print("=" * 70)
