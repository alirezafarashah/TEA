"""
Multi-concept (celebrity) erasure on SD-v1.4 with one discriminator per celebrity.

All prompt pairs of the celebrity set (celebrity_data.py) are mixed and shuffled, and every pair carries the index of
its celebrity:
  - discriminator k is trained only on the pairs of celebrity k, i.e. that celebrity's prompts vs the same prompts
    with its anchor celebrities;
  - the text encoder makes one update per batch on the whole mixed batch: the adversarial loss of each harmful prompt
    comes from its own celebrity's discriminator, plus token_cosine preservation against the original SD-v1.4 encoder.
    With a single celebrity this is exactly the single-concept training of train_utils.train_encoder.
The full text encoder is fine-tuned and saved after every epoch as <save_dir>/text_encoder_epoch_<k>, which
inference_dataset.py loads with --full_model_path.

Usage (from the repository root):
    python -m stablediffusion_1.multi_celebrity_erasure --celebrity_set celebrity_10 \
        --save_dir ./saved_models_sd14_celebrity_10
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from stablediffusion_1.celebrity_data import CELEBRITY_SETS, build_celebrity_to_safe_prompts
from stablediffusion_1.train_utils import (
    Discriminator,
    get_original_token_embeddings,
    get_token_embeddings,
    preservation_loss,
    save_weights,
    tokenize_batch,
    train_discriminator,
)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-celebrity erasure with one discriminator per celebrity (SD-v1.4)")
    p.add_argument("--celebrity_set", type=str, default="celebrity_10", choices=list(CELEBRITY_SETS))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--lr_disc", type=float, default=1e-3)
    p.add_argument("--lambda_adv", type=float, default=0.1)
    p.add_argument("--lambda_preserve", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--cache_dir", type=str, default="./cache/")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def subset(tokens, idx):
    """Rows `idx` of a tokenizer output (input_ids, attention_mask)."""
    return {k: v[idx] for k, v in tokens.items()}


def train_encoder_multidisc(
    encoder, discriminators, celeb_ids, optimizer_enc, harmful_tokens, safe_tokens, original_safe_tok_emb,
    lambda_adv, lambda_preserve,
):
    """train_utils.train_encoder, except that each harmful prompt is scored by its own celebrity's discriminator."""
    encoder.train()
    for d in discriminators:
        d.eval()
        d.requires_grad_(False)
    encoder.requires_grad_(False)
    for param in optimizer_enc.param_groups[0]["params"]:
        param.requires_grad = True

    optimizer_enc.zero_grad()

    harmful_tok_emb = get_token_embeddings(encoder, harmful_tokens)
    safe_tok_emb = get_token_embeddings(encoder, safe_tokens)

    B, T, D = harmful_tok_emb.shape
    harmful_mask = harmful_tokens["attention_mask"].bool()
    pred_adv = torch.zeros(B, T, 1, device=harmful_tok_emb.device)
    for k in celeb_ids.unique().tolist():
        idx = (celeb_ids == k).nonzero(as_tuple=True)[0]
        n = len(idx)
        pred_adv = pred_adv.index_put((idx,), discriminators[k](harmful_tok_emb[idx].reshape(n * T, D)).view(n, T, 1))
    loss_per_token = F.binary_cross_entropy_with_logits(pred_adv, torch.zeros_like(pred_adv), reduction="none").squeeze(-1)
    loss_adv = (loss_per_token * harmful_mask).sum() / harmful_mask.sum()

    loss_preserve = preservation_loss(
        safe_tok_emb, original_safe_tok_emb, safe_tokens["attention_mask"], "token_cosine", encoder=encoder
    )
    total_loss = lambda_adv * loss_adv + lambda_preserve * loss_preserve
    total_loss.backward()
    optimizer_enc.step()
    return loss_adv, loss_preserve


def main():
    args = parse_args()
    device = args.device
    max_length, add_special_tokens = 77, False  # as in main.py's train_state

    celebrities = CELEBRITY_SETS[args.celebrity_set]
    pairs = []  # (harmful, safe, celebrity index)
    for k, celebrity in enumerate(celebrities):
        for harmful, safes in build_celebrity_to_safe_prompts([celebrity]).items():
            pairs += [(harmful, safe, k) for safe in safes]
    dataloader = DataLoader(pairs, batch_size=args.batch_size, shuffle=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer", cache_dir=args.cache_dir)
    original_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder", cache_dir=args.cache_dir)
    original_encoder.eval().requires_grad_(False).to(device)
    encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder", cache_dir=args.cache_dir)
    encoder.train().requires_grad_(True).to(device)

    discriminators = [Discriminator(encoder.config.hidden_size).to(device) for _ in celebrities]
    optimizers_disc = [torch.optim.AdamW(d.parameters(), lr=args.lr_disc) for d in discriminators]
    optimizer_enc = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad), lr=args.lr_encoder)

    print(f"{args.celebrity_set}: {len(pairs)} pairs, {len(dataloader)} batches/epoch, "
          f"{len(celebrities)} discriminators")
    for epoch in range(args.epochs):
        acc_sum, acc_n = [0.0] * len(celebrities), [0] * len(celebrities)
        for harmful_batch, safe_batch, celeb_ids in dataloader:
            celeb_ids = celeb_ids.to(device)
            harmful_tokens = tokenize_batch(tokenizer, harmful_batch, max_length, add_special_tokens, device=device)
            safe_tokens = tokenize_batch(tokenizer, safe_batch, max_length, add_special_tokens, device=device)
            original_safe_emb_per_token = get_original_token_embeddings(original_encoder, safe_tokens)

            for k in celeb_ids.unique().tolist():
                idx = (celeb_ids == k).nonzero(as_tuple=True)[0]
                _, acc = train_discriminator(
                    encoder, discriminators[k], optimizers_disc[k], subset(harmful_tokens, idx), subset(safe_tokens, idx)
                )
                acc_sum[k] += acc.item()
                acc_n[k] += 1

            loss_adv, loss_preserve = train_encoder_multidisc(
                encoder, discriminators, celeb_ids, optimizer_enc, harmful_tokens, safe_tokens,
                original_safe_emb_per_token, args.lambda_adv, args.lambda_preserve,
            )

        print(f"epoch {epoch + 1}: adv {loss_adv.item():.4f} | preserve {loss_preserve.item():.4f} | mean disc acc per celebrity:")
        for k, celebrity in enumerate(celebrities):
            print(f"    {celebrity:16s} {acc_sum[k] / max(acc_n[k], 1):.3f}")
        save_weights(encoder, args.save_dir, epoch, use_lora=False)


if __name__ == "__main__":
    main()
