import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class EMA:

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Discriminator(nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(self, emb):
        return self.model(emb)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_lora_to_encoder(text_encoder, config):
    clip_lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    text_encoder = get_peft_model(text_encoder, clip_lora_config)
    print("LoRA applied to text encoder.")
    text_encoder.print_trainable_parameters()
    return text_encoder


def tokenize_batch(tokenizer, batch, max_length, add_special_tokens=False, device="cuda"):
    return tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    ).to(device)


def cleanup_models(*models):
    for model in models:
        model.cpu()
    torch.cuda.empty_cache()
    gc.collect()


def _clip_last_layer_hidden_states(outputs):
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    return outputs.last_hidden_state


def get_token_embeddings(encoder, tokens, device="cuda"):
    encoder.to(device)
    outputs = encoder(**tokens, output_hidden_states=True)
    embeddings = _clip_last_layer_hidden_states(outputs).float()
    return embeddings


def get_pooled_embeddings(encoder, tokens, device="cuda"):
    tok_emb = get_token_embeddings(encoder, tokens, device=device)
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    pooled_emb = (tok_emb * mask).sum(1) / mask.sum(1)
    return pooled_emb


def get_original_embeddings(original_encoder, tokens):
    with torch.no_grad():
        embeddings = get_pooled_embeddings(original_encoder, tokens)
    return embeddings


def get_original_token_embeddings(original_encoder, tokens):
    with torch.no_grad():
        embeddings = get_token_embeddings(original_encoder, tokens)
    return embeddings


def preservation_loss(
    new_tok_emb,
    orig_tok_emb,
    attention_mask,
    preservation_type,
    encoder=None,
    temperature=1.0,
):
    mask = attention_mask.float()

    if preservation_type == "pooled_l2":
        mask_3d = mask.unsqueeze(-1)
        new_pooled = (new_tok_emb * mask_3d).sum(1) / mask_3d.sum(1)
        orig_pooled = (orig_tok_emb * mask_3d).sum(1) / mask_3d.sum(1)
        return F.mse_loss(new_pooled, orig_pooled)

    elif preservation_type == "token_l2":
        diff_sq = ((new_tok_emb - orig_tok_emb) ** 2).sum(-1)
        return (diff_sq * mask).sum() / mask.sum()

    elif preservation_type == "token_cosine":
        cos_sim = F.cosine_similarity(new_tok_emb, orig_tok_emb, dim=-1)
        loss_per_token = 1.0 - cos_sim
        return (loss_per_token * mask).sum() / mask.sum()

    elif preservation_type == "token_kl":
        embed_weight = _get_embed_weight(encoder).to(new_tok_emb.device)
        new_logits = new_tok_emb @ embed_weight.T / temperature
        orig_logits = orig_tok_emb @ embed_weight.T / temperature

        new_log_probs = F.log_softmax(new_logits, dim=-1)
        orig_probs = F.softmax(orig_logits, dim=-1)

        kl_per_token = F.kl_div(new_log_probs, orig_probs, reduction="none").sum(-1)
        return (kl_per_token * mask).sum() / mask.sum()

    else:
        raise ValueError(f"Unknown preservation_type: {preservation_type!r}")


def _get_embed_weight(encoder):
    base = encoder.base_model if hasattr(encoder, "base_model") else encoder
    return base.text_model.embeddings.token_embedding.weight.detach().float()


def train_discriminator_pooled(encoder, discriminator, optimizer_disc, harmful_tokens, safe_tokens, safe_emb=None):
    with torch.no_grad():
        harmful_emb = get_pooled_embeddings(encoder, harmful_tokens)
        if safe_emb is None:
            safe_emb = get_pooled_embeddings(encoder, safe_tokens)

    discriminator.train()
    encoder.eval()
    discriminator.requires_grad_(True)
    encoder.requires_grad_(False)

    optimizer_disc.zero_grad()
    pred_harmful = discriminator(harmful_emb.detach())
    pred_safe = discriminator(safe_emb.detach())

    labels_harmful = torch.ones_like(pred_harmful)
    labels_safe = torch.zeros_like(pred_safe)

    loss_disc = F.binary_cross_entropy_with_logits(pred_harmful, labels_harmful) + F.binary_cross_entropy_with_logits(
        pred_safe, labels_safe
    )
    loss_disc.backward()
    optimizer_disc.step()

    pred_harmful_binary = (torch.sigmoid(pred_harmful) > 0.5).float()
    pred_safe_binary = (torch.sigmoid(pred_safe) > 0.5).float()
    correct_harmful = (pred_harmful_binary == labels_harmful).float().sum()
    correct_safe = (pred_safe_binary == labels_safe).float().sum()
    total_samples = labels_harmful.size(0) + labels_safe.size(0)
    accuracy = (correct_harmful + correct_safe) / total_samples

    del harmful_emb, safe_emb, pred_harmful, pred_safe
    torch.cuda.empty_cache()

    return loss_disc, accuracy


def train_encoder_pooled(
    encoder,
    discriminator,
    optimizer_enc,
    harmful_tokens,
    safe_tokens,
    original_safe_tok_emb,
    lambda_adv,
    lambda_preserve,
    ema=None,
    preservation_type="pooled_l2",
    kl_temperature=1.0,
):
    encoder.train()
    discriminator.eval()
    discriminator.requires_grad_(False)
    encoder.requires_grad_(False)

    for param in optimizer_enc.param_groups[0]["params"]:
        param.requires_grad = True

    optimizer_enc.zero_grad()

    harmful_emb_adv = get_pooled_embeddings(encoder, harmful_tokens)
    pred_adv = discriminator(harmful_emb_adv)
    safe_tok_emb_new = get_token_embeddings(encoder, safe_tokens)

    labels_adv = torch.zeros_like(pred_adv)
    loss_adv = F.binary_cross_entropy_with_logits(pred_adv, labels_adv)

    loss_preserve = preservation_loss(
        safe_tok_emb_new,
        original_safe_tok_emb,
        safe_tokens["attention_mask"],
        preservation_type,
        encoder=encoder,
        temperature=kl_temperature,
    )
    total_loss = lambda_adv * loss_adv + lambda_preserve * loss_preserve

    total_loss.backward()
    optimizer_enc.step()

    if ema is not None:
        ema.update(encoder)

    return loss_adv, loss_preserve


def train_discriminator(encoder, discriminator, optimizer_disc, harmful_tokens, safe_tokens, safe_emb=None):
    with torch.no_grad():
        harmful_emb = get_token_embeddings(encoder, harmful_tokens)
        if safe_emb is None:
            safe_emb = get_token_embeddings(encoder, safe_tokens)

    discriminator.train()
    encoder.eval()
    discriminator.requires_grad_(True)
    encoder.requires_grad_(False)

    optimizer_disc.zero_grad()

    B, T, D = harmful_emb.shape
    harmful_mask = harmful_tokens["attention_mask"].bool()
    safe_mask = safe_tokens["attention_mask"].bool()

    pred_harmful = discriminator(harmful_emb.detach().view(B * T, D)).view(B, T, 1)
    pred_safe = discriminator(safe_emb.detach().view(B * T, D)).view(B, T, 1)

    labels_harmful = torch.ones_like(pred_harmful)
    labels_safe = torch.zeros_like(pred_safe)

    def masked_bce_logits(pred, labels, mask):
        loss_per_token = F.binary_cross_entropy_with_logits(pred, labels, reduction="none").squeeze(-1)
        return (loss_per_token * mask).sum() / mask.sum()

    loss_disc = masked_bce_logits(pred_harmful, labels_harmful, harmful_mask) + masked_bce_logits(
        pred_safe, labels_safe, safe_mask
    )
    loss_disc.backward()
    optimizer_disc.step()

    pred_harmful_binary = (torch.sigmoid(pred_harmful.squeeze(-1)) > 0.5).float()
    pred_safe_binary = (torch.sigmoid(pred_safe.squeeze(-1)) > 0.5).float()
    correct = ((pred_harmful_binary == labels_harmful.squeeze(-1)).float() * harmful_mask).sum() + (
        (pred_safe_binary == labels_safe.squeeze(-1)).float() * safe_mask
    ).sum()
    total = harmful_mask.sum() + safe_mask.sum()
    accuracy = correct / total

    del harmful_emb, safe_emb, pred_harmful, pred_safe
    torch.cuda.empty_cache()

    return loss_disc, accuracy


def train_encoder(
    encoder,
    discriminator,
    optimizer_enc,
    harmful_tokens,
    safe_tokens,
    original_safe_tok_emb,
    lambda_adv,
    lambda_preserve,
    ema=None,
    preservation_type="pooled_l2",
    kl_temperature=1.0,
):
    encoder.train()
    discriminator.eval()
    discriminator.requires_grad_(False)
    encoder.requires_grad_(False)

    for param in optimizer_enc.param_groups[0]["params"]:
        param.requires_grad = True

    optimizer_enc.zero_grad()

    harmful_tok_emb = get_token_embeddings(encoder, harmful_tokens)
    safe_tok_emb = get_token_embeddings(encoder, safe_tokens)

    B, T, D = harmful_tok_emb.shape
    harmful_mask = harmful_tokens["attention_mask"].bool()
    pred_adv = discriminator(harmful_tok_emb.view(B * T, D)).view(B, T, 1)
    labels_adv = torch.zeros_like(pred_adv)
    loss_per_token = F.binary_cross_entropy_with_logits(pred_adv, labels_adv, reduction="none").squeeze(-1)
    loss_adv = (loss_per_token * harmful_mask).sum() / harmful_mask.sum()

    loss_preserve = preservation_loss(
        safe_tok_emb,
        original_safe_tok_emb,
        safe_tokens["attention_mask"],
        preservation_type,
        encoder=encoder,
        temperature=kl_temperature,
    )
    total_loss = lambda_adv * loss_adv + lambda_preserve * loss_preserve

    total_loss.backward()
    optimizer_enc.step()

    if ema is not None:
        ema.update(encoder)

    return loss_adv, loss_preserve


def save_weights(text_encoder, save_dir, epoch, ema=None, use_lora=True):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"text_encoder_epoch_{epoch+1}")
    _save_encoder(text_encoder, save_path, use_lora)

    if ema is not None:
        ema_save_path = os.path.join(save_dir, f"text_encoder_epoch_{epoch+1}_ema")
        ema.apply_shadow(text_encoder)
        _save_encoder(text_encoder, ema_save_path, use_lora)
        ema.restore(text_encoder)

    mode = "LoRA" if use_lora else "full"
    print(f"Text encoder ({mode}) saved to {save_dir}")


def _save_encoder(text_encoder, save_path, use_lora):
    if use_lora:
        text_encoder.save_pretrained(save_path)
    else:
        os.makedirs(save_path, exist_ok=True)
        text_encoder.save_pretrained(save_path)


def generate_images_with_text_encoder(pipeline, prompt_list, text_encoder, encoder_name, epoch, config, device="cuda"):
    orig_encoder = pipeline.text_encoder
    pipeline.text_encoder = text_encoder.eval()

    out_dir = os.path.join(config.output_dir, f"epoch_{epoch+1}_{encoder_name}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        with torch.no_grad():
            for idx, prompt in enumerate(prompt_list):
                gen = torch.Generator(device=device).manual_seed(config.generation_seed)
                result = pipeline(
                    prompt=[prompt],
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=gen,
                )
                image = result.images[0]
                file_name = f"{idx+1}_{prompt[:40].replace(' ', '_')}.png"
                image.save(os.path.join(out_dir, file_name))
    finally:
        pipeline.text_encoder = orig_encoder
