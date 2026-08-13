import argparse
import os

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPTextModel
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="SD1.4 inference on a CSV prompt dataset with a trained text encoder (LoRA or full fine-tuned)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base Stable Diffusion 1.x model id or path",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Directory containing text_encoder_epoch_* LoRA adapter folders (from LoRA training)",
    )
    mode_group.add_argument(
        "--full_model_path",
        type=str,
        default=None,
        help="Directory containing a full fine-tuned text encoder saved with save_pretrained "
             "(e.g. text_encoder_epoch_5 from full fine-tuning)",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="1-based epoch index for loading text_encoder_epoch_{epoch} (required when using --lora_path)",
    )
    parser.add_argument("--use_ema", action="store_true", help="Load the _ema variant of the checkpoint")
    parser.add_argument("--csv_path", type=str, default="./datasets/nudity-ring-a-bell.csv")
    parser.add_argument("--output_dir", type=str, default="./inference_sd14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="./cache/")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug)")
    parser.add_argument("--start_index", type=int, default=0, help="Skip the first N rows (resume from this row index)")

    args = parser.parse_args()

    if args.lora_path is not None and args.epoch is None:
        parser.error("--epoch is required when using --lora_path")

    return args


def load_pipeline(base_model_path, device, cache_dir, torch_dtype,
                  lora_path=None, epoch=None, use_ema=False, full_model_path=None):
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading base model: {base_model_path}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )

    suffix = "_ema" if use_ema else ""

    if lora_path is not None:
        adapter_dir = os.path.join(lora_path, f"text_encoder_epoch_{epoch}{suffix}")
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(
                f"LoRA adapter not found: {adapter_dir}. "
                f"Expected a folder saved by LoRA training (e.g. text_encoder_epoch_1 or text_encoder_epoch_1_ema)."
            )
        print(f"Loading text encoder LoRA from {adapter_dir}...")
        pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, adapter_dir)

    elif full_model_path is not None:
        checkpoint_dir = f"{full_model_path}{suffix}" if suffix else full_model_path
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"Full fine-tuned text encoder not found: {checkpoint_dir}. "
                f"Expected a folder saved by full fine-tuning."
            )
        print(f"Loading full fine-tuned text encoder from {checkpoint_dir}...")
        pipeline.text_encoder = CLIPTextModel.from_pretrained(checkpoint_dir, torch_dtype=torch_dtype)

    else:
        print("No adapter or fine-tuned model provided — using base model text encoder.")

    pipeline.to(device)
    pipeline.text_encoder.eval()
    return pipeline


def main():
    args = parse_args()

    print(f"Loading dataset from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    is_coco = "coco_30k_10k.csv" in args.csv_path
    if is_coco:
        print("COCO dataset detected. Sampling 1000 rows...")
        df = df.sample(n=1000, random_state=42)

    if "adv_prompt" in df.columns:
        prompt_col = "adv_prompt"
    elif "sensitive prompt" in df.columns:
        prompt_col = "sensitive prompt"
    elif "prompt" in df.columns:
        prompt_col = "prompt"
    else:
        raise ValueError("No recognized prompt column ('adv_prompt', 'sensitive prompt', 'prompt') found in CSV.")

    before = len(df)
    df = df.dropna(subset=[prompt_col]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with NaN in '{prompt_col}' column.")

    guidance_scale_col = None
    if "evaluation_guidance" in df.columns:
        guidance_scale_col = "evaluation_guidance"
    else:
        print("No 'evaluation_guidance' column found. Using default guidance scale of 7.5.")

    seed_col = None
    for col in ["seed", "validation_seed", "sd_seed", "evaluation_seed"]:
        if col in df.columns:
            seed_col = col
            break
    if seed_col is None:
        print("Warning: No seed column found. Defaulting to 42 for all images.")

    if args.start_index:
        df = df.iloc[args.start_index:].reset_index(drop=True)
        print(f"Resuming from row {args.start_index} ({len(df)} rows remaining).")

    if args.limit:
        df = df.head(args.limit)

    os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32
    pipeline = load_pipeline(
        args.model_name,
        args.device,
        args.cache_dir,
        torch_dtype,
        lora_path=args.lora_path,
        epoch=args.epoch,
        use_ema=args.use_ema,
        full_model_path=args.full_model_path,
    )

    print(f"Starting inference on {len(df)} prompts...")
    autocast_dtype = torch.float16 if torch_dtype == torch.float16 else torch.bfloat16
    ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(args.device == "cuda"))

    with ctx, torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            prompt = row[prompt_col]
            seed = int(row[seed_col]) if seed_col else 42
            guidance_scale = float(row[guidance_scale_col]) if guidance_scale_col else 7.5
            generator = torch.Generator(device=args.device).manual_seed(seed)
            image = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

            if is_coco:
                filename = f"{row['coco_id']}.png"
            else:
                if row.get("case_number") is not None:
                    filename = f"{row['case_number']}_{seed}.png"
                else:
                    filename = f"{index}_{seed}.png"

            image.save(os.path.join(args.output_dir, filename))

    print(f"Done! Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
