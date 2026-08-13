import argparse
import os

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="SD1.4 inference on a CSV prompt dataset with the original (unmodified) model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base Stable Diffusion 1.x model id or path",
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="./cache/")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    is_coco = "coco_30k_10k.csv" in args.csv_path
    if is_coco:
        print("COCO dataset detected. Sampling 1000 rows...")
        df = df.sample(n=1000, random_state=42)

    if "sensitive prompt" in df.columns:
        prompt_col = "sensitive prompt"
    elif "prompt" in df.columns:
        prompt_col = "prompt"
    else:
        raise ValueError("Neither 'sensitive prompt' nor 'prompt' column found in CSV.")

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

    if args.limit:
        df = df.head(args.limit)

    os.makedirs(args.output_dir, exist_ok=True)

    cache_dir = os.path.expanduser(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    torch_dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32

    print(f"Loading base model: {args.model_name}...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.to(args.device)

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
