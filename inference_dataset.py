import torch
import os
import argparse
import pandas as pd
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from transformers import CLIPTextModelWithProjection, T5EncoderModel
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on Ring-A-Bell Dataset with Unlearned Models")
    
    # Model and checkpoint paths
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-large", help="Base model path")
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        choices=["lora", "full_ft"],
        default="lora",
        help="How to load text encoders: LoRA adapters or full fine-tuned checkpoints",
    )
    parser.add_argument("--lora_path", type=str, default=None, help="Directory where LoRA weights are saved")
    parser.add_argument("--epoch", type=int, required=True, help="Which epoch to load")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights")
    parser.add_argument("--enc1_ckpt", type=str, default=None, help="Path to full fine-tuned encoder_1 checkpoint dir")
    parser.add_argument("--enc2_ckpt", type=str, default=None, help="Path to full fine-tuned encoder_2 checkpoint dir")
    parser.add_argument("--enc3_ckpt", type=str, default=None, help="Path to full fine-tuned encoder_3 checkpoint dir")
    
    # Dataset details
    parser.add_argument("--csv_path", type=str, default="./nudity-ring-a-bell.csv", help="Path to the dataset CSV")
    parser.add_argument("--output_dir", type=str, default="./inference_ring_a_bell", help="Where to save images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Optional: Limit number of samples for testing
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process (for testing)")
    
    return parser.parse_args()

def load_modified_pipeline(base_model_path, lora_save_dir, epoch, use_ema=False, device="cuda"):
    """
    Loads SD3 and attaches the specific LoRA adapters.
    """
    print(f"Loading base model: {base_model_path}...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        cache_dir="./cache/"
    ).to("cuda")

    suffix = "_ema" if use_ema else ""
    print(f"Loading LoRA adapters for Epoch {epoch} (EMA: {use_ema})...")

    # Load Encoder 1 (CLIP G)
    path_enc1 = os.path.join(lora_save_dir, f"encoder_1_epoch_{epoch}{suffix}")
    if os.path.exists(path_enc1):
        pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, path_enc1)
    else:
        print(f"Warning: {path_enc1} not found. Using base Encoder 1.")

    # Load Encoder 2 (CLIP L)
    path_enc2 = os.path.join(lora_save_dir, f"encoder_2_epoch_{epoch}{suffix}")
    if os.path.exists(path_enc2):
        pipeline.text_encoder_2 = PeftModel.from_pretrained(pipeline.text_encoder_2, path_enc2)
    else:
        print(f"Warning: {path_enc2} not found. Using base Encoder 2.")

    # Load Encoder 3 (T5)
    path_enc3 = os.path.join(lora_save_dir, f"encoder_3_epoch_{epoch}{suffix}")
    if os.path.exists(path_enc3):
        pipeline.text_encoder_3 = PeftModel.from_pretrained(pipeline.text_encoder_3, path_enc3)
    else:
        print(f"Warning: {path_enc3} not found. Using base Encoder 3.")

    pipeline.to(device)
    return pipeline


def _load_full_ft_encoder(base_encoder, ckpt_path):
    """Load a full fine-tuned text encoder from ckpt_path, matching base encoder type."""
    if isinstance(base_encoder, T5EncoderModel):
        return T5EncoderModel.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
    return CLIPTextModelWithProjection.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)


def load_modified_pipeline_full_ft(base_model_path, enc1_ckpt, enc2_ckpt, enc3_ckpt, device="cuda"):
    """
    Load SD3 and replace text encoders with full fine-tuned checkpoints.
    """
    print(f"Loading base model: {base_model_path}...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        cache_dir="./cache/"
    ).to("cuda")

    print("Loading full fine-tuned encoder checkpoints...")
    pipeline.text_encoder = _load_full_ft_encoder(pipeline.text_encoder, enc1_ckpt)
    pipeline.text_encoder_2 = _load_full_ft_encoder(pipeline.text_encoder_2, enc2_ckpt)
    pipeline.text_encoder_3 = _load_full_ft_encoder(pipeline.text_encoder_3, enc3_ckpt)

    pipeline.to(device)
    return pipeline

def main():
    args = parse_args()

    if args.checkpoint_type == "lora":
        if not args.lora_path:
            raise ValueError("--lora_path is required when --checkpoint_type lora")
    else:
        missing = [name for name, path in [
            ("--enc1_ckpt", args.enc1_ckpt),
            ("--enc2_ckpt", args.enc2_ckpt),
            ("--enc3_ckpt", args.enc3_ckpt),
        ] if not path]
        if missing:
            raise ValueError(f"Missing required args for --checkpoint_type full_ft: {', '.join(missing)}")
        for path_name, path in [("enc1_ckpt", args.enc1_ckpt), ("enc2_ckpt", args.enc2_ckpt), ("enc3_ckpt", args.enc3_ckpt)]:
            if not os.path.isdir(path):
                raise FileNotFoundError(f"{path_name} directory not found: {path}")
    
    # 1. Load Dataset
    print(f"Loading dataset from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # --- MODIFICATION: Check for COCO dataset and sample 1000 ---
    is_coco = "coco_30k_10k.csv" in args.csv_path
    if is_coco:
        print("COCO dataset detected. Sampling 1000 rows...")
        df = df.sample(n=1000, random_state=42) # Fixed seed for reproducibility

    # Verify columns exist (Support both 'sensitive prompt' and 'prompt')
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
    
    # Check for seed column (Added 'evaluation_seed' for COCO)
    seed_col = None
    for col in ['seed', 'validation_seed', 'sd_seed', 'evaluation_seed']:
        if col in df.columns:
            seed_col = col
            break
    
    if seed_col is None:
        print("Warning: No 'seed' column found. Defaulting to 42 for all images.")
    
    if args.limit:
        df = df.head(args.limit)

    # 2. Setup Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Load Pipeline
    if args.checkpoint_type == "lora":
        pipeline = load_modified_pipeline(
            args.model_name,
            args.lora_path,
            args.epoch,
            args.use_ema,
            args.device
        )
    else:
        pipeline = load_modified_pipeline_full_ft(
            args.model_name,
            args.enc1_ckpt,
            args.enc2_ckpt,
            args.enc3_ckpt,
            args.device
        )

    print(f"Starting inference on {len(df)} prompts...")

    # 4. Iterate and Generate
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


# Generate image
    with ctx, torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            prompt = row[prompt_col]
            
            # Determine seed
            seed = int(row[seed_col]) if seed_col else 42
            guidance_scale = float(row[guidance_scale_col]) if guidance_scale_col else 7.0
            # Create generator for this specific seed
            generator = torch.Generator(device=args.device).manual_seed(seed)
            image = pipeline(
                prompt=prompt,
                num_inference_steps=28,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            
            # Save image
            # --- MODIFICATION: Use coco_id for filename if using COCO dataset ---
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