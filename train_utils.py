import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from peft import LoraConfig, get_peft_model
import os
import gc


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

                if self.shadow[name].device != param.data.device:
                    self.shadow[name] = self.shadow[name].to(param.data.device)
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
    
    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class Discriminator(nn.Module):
    
    def __init__(self, emb_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, emb):
        return self.model(emb)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_lora_to_encoders(text_encoder, text_encoder_2, text_encoder_3, config):

    clip_lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"]
    )

    t5_lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q", "k", "v"]
    )

    text_encoder = get_peft_model(text_encoder, clip_lora_config)
    text_encoder_2 = get_peft_model(text_encoder_2, clip_lora_config)
    text_encoder_3 = get_peft_model(text_encoder_3, t5_lora_config)

    print("LoRA applied to all encoders.")
    text_encoder.print_trainable_parameters()

    return text_encoder, text_encoder_2, text_encoder_3


def get_token_embeddings(encoder, tokens, encoder_type="clip", device="cuda"):
    encoder.to(device)
    outputs = encoder(**tokens, output_hidden_states=True)
    if encoder_type == "clip":
        embeddings = outputs.hidden_states[-2].float()  # (B, T, D)
    else:
        embeddings = outputs.last_hidden_state.float()  # (B, T, D)
    return embeddings


def get_pooled_embeddings(encoder, tokens, encoder_type="clip", device="cuda"):
    encoder.to(device)
    outputs = encoder(**tokens, output_hidden_states=True)
    if encoder_type == "clip":
        embeddings = outputs.hidden_states[-2].float()
    else:
        embeddings = outputs.last_hidden_state.float()

    mask = tokens['attention_mask'].unsqueeze(-1).float()
    pooled_emb = (embeddings * mask).sum(1) / mask.sum(1)
    return pooled_emb


def tokenize_batch(tokenizer, batch, max_length, add_special_tokens=False, device="cuda"):
    return tokenizer(
        batch, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        return_tensors="pt"
    ).to(device)


def cleanup_models(*models):
    for model in models:
        model.cpu()
    torch.cuda.empty_cache()
    gc.collect()


def save_weights(encoder_configs, save_dir, epoch, use_lora=True):
    os.makedirs(save_dir, exist_ok=True)

    for idx, enc_cfg in enumerate(encoder_configs):
        slot = enc_cfg.get('slot_idx', idx + 1)  # use slot_idx if present, else positional
        save_path = os.path.join(save_dir, f"encoder_{slot}_epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        enc_cfg['encoder'].save_pretrained(save_path)

        if 'ema' in enc_cfg and enc_cfg['ema'] is not None:
            ema = enc_cfg['ema']
            encoder = enc_cfg['encoder']

            ema_save_path = os.path.join(save_dir, f"encoder_{slot}_epoch_{epoch+1}_ema")
            os.makedirs(ema_save_path, exist_ok=True)
            ema.apply_shadow(encoder)
            encoder.save_pretrained(ema_save_path)
            ema.restore(encoder)

            ema_state_path = os.path.join(save_dir, f"ema_state_{slot}_epoch_{epoch+1}.pt")
            torch.save(ema.state_dict(), ema_state_path)

    mode = "LoRA" if use_lora else "full"
    print(f"Encoder weights ({mode}) saved to {save_dir}")


# Keep old name as alias for backwards compatibility
def save_lora_weights(encoder_configs, save_dir, epoch):
    save_weights(encoder_configs, save_dir, epoch, use_lora=True)

# This is for version with pooled embedding
def train_discriminator_pooled(encoder, discriminator, optimizer_disc, harmful_tokens, safe_tokens, safe_emb=None, encoder_type="clip"):
    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            harmful_emb = get_pooled_embeddings(encoder, harmful_tokens, encoder_type=encoder_type)
            if safe_emb is None:
                safe_emb = get_pooled_embeddings(encoder, safe_tokens, encoder_type=encoder_type)

    
    discriminator.train()
    encoder.eval()
    discriminator.requires_grad_(True)
    encoder.requires_grad_(False)
    
    optimizer_disc.zero_grad()
    pred_harmful = discriminator(harmful_emb.detach())
    pred_safe = discriminator(safe_emb.detach())
    
    labels_harmful = torch.ones_like(pred_harmful)
    labels_safe = torch.zeros_like(pred_safe)
    
    loss_disc = F.binary_cross_entropy(pred_harmful, labels_harmful) + \
                F.binary_cross_entropy(pred_safe, labels_safe)
    loss_disc.backward()
    optimizer_disc.step()
    
    # Calculate accuracy
    pred_harmful_binary = (pred_harmful > 0.5).float()
    pred_safe_binary = (pred_safe > 0.5).float()
    correct_harmful = (pred_harmful_binary == labels_harmful).float().sum()
    correct_safe = (pred_safe_binary == labels_safe).float().sum()
    total_samples = labels_harmful.size(0) + labels_safe.size(0)
    accuracy = (correct_harmful + correct_safe) / total_samples
    
    # Cleanup
    del harmful_emb, safe_emb, pred_harmful, pred_safe
    torch.cuda.empty_cache()
    
    return loss_disc, accuracy

# This is for version with pooled embedding
def train_encoder_pooled(encoder, discriminator, optimizer_enc, harmful_tokens, safe_tokens, normal_tokens,
                  original_safe_emb, original_normal_emb, lambda_adv, lambda_preserve, ema=None,
                  encoder_type="clip", preservation_type="pooled_l2", kl_temperature=1.0):
    encoder.train()
    discriminator.eval()
    discriminator.requires_grad_(False)
    encoder.requires_grad_(False)

    for param in optimizer_enc.param_groups[0]['params']:
        param.requires_grad = True

    optimizer_enc.zero_grad()

    with autocast("cuda", dtype=torch.bfloat16):
        harmful_emb_adv = get_pooled_embeddings(encoder, harmful_tokens, encoder_type=encoder_type)
        pred_adv = discriminator(harmful_emb_adv)
        safe_tok_emb_new = get_token_embeddings(encoder, safe_tokens, encoder_type=encoder_type)

    labels_adv = torch.zeros_like(pred_adv)
    loss_adv = F.binary_cross_entropy(pred_adv, labels_adv)

    # For pooled path, get the original token-level embeddings for preservation loss variants
    # original_safe_emb is pooled; we need token-level for non-pooled_l2 types.
    # When preservation_type == "pooled_l2", we compare pooled embeddings directly.
    if preservation_type == "pooled_l2":
        mask_3d = safe_tokens['attention_mask'].unsqueeze(-1).float()
        safe_emb_new_pooled = (safe_tok_emb_new.float() * mask_3d).sum(1) / mask_3d.sum(1)
        loss_preserve = F.mse_loss(safe_emb_new_pooled, original_safe_emb.float())
    else:
        # original_safe_emb here is expected to be token-level when using non-pooled preservation
        loss_preserve = preservation_loss(
            safe_tok_emb_new.float(),
            original_safe_emb.float(),
            safe_tokens['attention_mask'],
            preservation_type,
            encoder=encoder,
            encoder_type=encoder_type,
            temperature=kl_temperature,
        )

    total_loss = lambda_adv * loss_adv + lambda_preserve * loss_preserve
    total_loss.backward()
    optimizer_enc.step()

    if ema is not None:
        ema.update(encoder)

    return loss_adv, loss_preserve


def get_original_embeddings(original_encoder, tokens, encoder_type="clip"):
    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            embeddings = get_pooled_embeddings(original_encoder, tokens, encoder_type=encoder_type)
    return embeddings
  
  
def get_original_token_embeddings(original_encoder, tokens, encoder_type="clip"):
    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            embeddings = get_token_embeddings(original_encoder, tokens, encoder_type=encoder_type)
    return embeddings


def _get_embed_weight_clip(encoder):
    base = encoder.base_model if hasattr(encoder, "base_model") else encoder
    return base.text_model.embeddings.token_embedding.weight.detach().float()


def _get_embed_weight_t5(encoder):
    base = encoder.base_model if hasattr(encoder, "base_model") else encoder
    return base.shared.weight.detach().float()


def preservation_loss(
    new_tok_emb,
    orig_tok_emb,
    attention_mask,
    preservation_type,
    encoder=None,
    encoder_type="clip",
    temperature=1.0,
):
    mask = attention_mask.float()  # (B, T)

    if preservation_type == "pooled_l2":
        mask_3d = mask.unsqueeze(-1)
        new_pooled = (new_tok_emb * mask_3d).sum(1) / mask_3d.sum(1)
        orig_pooled = (orig_tok_emb * mask_3d).sum(1) / mask_3d.sum(1)
        return F.mse_loss(new_pooled, orig_pooled)

    elif preservation_type == "token_l2":
        diff_sq = ((new_tok_emb - orig_tok_emb) ** 2).sum(-1)  # (B, T)
        return (diff_sq * mask).sum() / mask.sum()

    elif preservation_type == "token_cosine":
        cos_sim = F.cosine_similarity(new_tok_emb, orig_tok_emb, dim=-1)  # (B, T)
        loss_per_token = 1.0 - cos_sim
        return (loss_per_token * mask).sum() / mask.sum()

    elif preservation_type == "token_kl":
        if encoder_type == "clip":
            embed_weight = _get_embed_weight_clip(encoder).to(new_tok_emb.device)
        else:
            embed_weight = _get_embed_weight_t5(encoder).to(new_tok_emb.device)

        new_logits = new_tok_emb @ embed_weight.T / temperature   # (B, T, V)
        orig_logits = orig_tok_emb @ embed_weight.T / temperature  # (B, T, V)

        new_log_probs = F.log_softmax(new_logits, dim=-1)   # (B, T, V)
        orig_probs = F.softmax(orig_logits, dim=-1)          # (B, T, V)

        kl_per_token = F.kl_div(new_log_probs, orig_probs, reduction="none").sum(-1)  # (B, T)
        return (kl_per_token * mask).sum() / mask.sum()

    else:
        raise ValueError(f"Unknown preservation_type: {preservation_type!r}")


def train_discriminator(encoder, discriminator, optimizer_disc, harmful_tokens, safe_tokens, safe_emb=None, encoder_type="clip"):
    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            harmful_emb = get_token_embeddings(encoder, harmful_tokens, encoder_type=encoder_type)  # (B, T, D)
            if safe_emb is None:
                safe_emb = get_token_embeddings(encoder, safe_tokens, encoder_type=encoder_type)    # (B, T, D)

    discriminator.train()
    encoder.eval()
    discriminator.requires_grad_(True)
    encoder.requires_grad_(False)

    optimizer_disc.zero_grad()

    B, T, D = harmful_emb.shape
    # print("shape of safe embed:", safe_emb.shape)
    harmful_mask = harmful_tokens['attention_mask'].bool()  # (B, T)
    safe_mask    = safe_tokens['attention_mask'].bool()     # (B, T)

    pred_harmful = discriminator(harmful_emb.detach().view(B * T, D)).view(B, T, 1)  # (B, T, 1)
    pred_safe    = discriminator(safe_emb.detach().view(B * T, D)).view(B, T, 1)     # (B, T, 1)

    labels_harmful = torch.ones_like(pred_harmful)
    labels_safe    = torch.zeros_like(pred_safe)

    def masked_bce(pred, labels, mask):
        loss_per_token = F.binary_cross_entropy(pred, labels, reduction='none')  # (B, T, 1)
        loss_per_token = loss_per_token.squeeze(-1)                              # (B, T)
        return (loss_per_token * mask).sum() / mask.sum()

    loss_disc = masked_bce(pred_harmful, labels_harmful, harmful_mask) + \
                masked_bce(pred_safe,    labels_safe,    safe_mask)

    loss_disc.backward()
    optimizer_disc.step()

    # Accuracy (masked)
    pred_harmful_binary = (pred_harmful.squeeze(-1) > 0.5).float()  # (B, T)
    pred_safe_binary    = (pred_safe.squeeze(-1)    > 0.5).float()  # (B, T)
    correct = ((pred_harmful_binary == labels_harmful.squeeze(-1)).float() * harmful_mask).sum() + \
              ((pred_safe_binary    == labels_safe.squeeze(-1)   ).float() * safe_mask).sum()
    total   = harmful_mask.sum() + safe_mask.sum()
    accuracy = correct / total

    del harmful_emb, safe_emb, pred_harmful, pred_safe
    torch.cuda.empty_cache()

    return loss_disc, accuracy


def train_encoder(encoder, discriminator, optimizer_enc, harmful_tokens, safe_tokens, normal_tokens,
                  original_safe_emb, original_normal_emb, lambda_adv, lambda_preserve, ema=None,
                  encoder_type="clip", preservation_type="pooled_l2", kl_temperature=1.0):
    encoder.train()
    discriminator.eval()
    discriminator.requires_grad_(False)
    encoder.requires_grad_(False)

    for param in optimizer_enc.param_groups[0]['params']:
        param.requires_grad = True

    optimizer_enc.zero_grad()

    with autocast("cuda", dtype=torch.bfloat16):
        harmful_emb_adv = get_token_embeddings(encoder, harmful_tokens, encoder_type=encoder_type)  # (B, T, D)
        safe_tok_emb    = get_token_embeddings(encoder, safe_tokens,    encoder_type=encoder_type)  # (B, T, D)

    B, T, D = harmful_emb_adv.shape

    # Adversarial loss: masked token-level BCE
    harmful_mask   = harmful_tokens['attention_mask'].bool()  # (B, T)
    pred_adv       = discriminator(harmful_emb_adv.view(B * T, D)).view(B, T, 1)
    labels_adv     = torch.zeros_like(pred_adv)
    loss_per_token = F.binary_cross_entropy(pred_adv, labels_adv, reduction='none').squeeze(-1)  # (B, T)
    loss_adv       = (loss_per_token * harmful_mask).sum() / harmful_mask.sum()

    loss_preserve = preservation_loss(
        safe_tok_emb.float(),
        original_safe_emb.float(),
        safe_tokens['attention_mask'],
        preservation_type,
        encoder=encoder,
        encoder_type=encoder_type,
        temperature=kl_temperature,
    )

    total_loss = lambda_adv * loss_adv + lambda_preserve * loss_preserve
    total_loss.backward()
    optimizer_enc.step()

    if ema is not None:
        ema.update(encoder)

    return loss_adv, loss_preserve
