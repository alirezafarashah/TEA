import torch
import os
import argparse
import pandas as pd
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on Ring-A-Bell Dataset with Unlearned Models")
    
    # Model and LoRA paths
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-large", help="Base model path")
    
    # Dataset details
    parser.add_argument("--csv_path", type=str, default="./nudity-ring-a-bell.csv", help="Path to the dataset CSV")
    parser.add_argument("--output_dir", type=str, default="./inference_ring_a_bell", help="Where to save images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process (for testing)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading dataset from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    is_coco = "coco_30k_10k.csv" in args.csv_path
    if is_coco:
        print("COCO dataset detected. Sampling 1000 rows...")
        df = df.sample(n=1000, random_state=42) # Fixed seed for reproducibility

    if 'sensitive prompt' in df.columns:
        prompt_col = 'sensitive prompt'
    elif 'prompt' in df.columns:
        prompt_col = 'prompt'
    else:
        raise ValueError("Neither 'sensitive prompt' nor 'prompt' column found in CSV.")
    guidance_scale_col = None
    if "evaluation_guidance" in df.columns:
        guidance_scale_col = "evaluation_guidance"
    else:
        print("No 'evaluation_guidance' column found. Using default guidance scale of 7.0.")
    
    seed_col = None
    for col in ['seed', 'validation_seed', 'sd_seed', 'evaluation_seed']:
        if col in df.columns:
            seed_col = col
            break
    
    if seed_col is None:
        print("Warning: No 'seed' column found. Defaulting to 42 for all images.")
    
    if args.limit:
        df = df.head(args.limit)

    os.makedirs(args.output_dir, exist_ok=True)
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        torch_dtype=torch.bfloat16,
        cache_dir="./cache/"
    ).to("cuda")
    print(f"Starting inference on {len(df)} prompts...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row[prompt_col]
        seed = int(row[seed_col]) if seed_col else 42
        
        generator = torch.Generator(device=args.device).manual_seed(seed)
        guidance_scale = float(row[guidance_scale_col]) if guidance_scale_col else 7.0

        image = pipeline(
            prompt=prompt,
            num_inference_steps=28,
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
            
        save_path = os.path.join(args.output_dir, filename)
        image.save(save_path)

    print(f"Done! Images saved to {args.output_dir}")

if __name__ == "__main__":
    main()