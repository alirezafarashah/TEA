from PIL import Image
import os
from tqdm import tqdm
import torch


def generate_images_with_encoders(
    pipeline, 
    prompt_list, 
    text_encoder, 
    text_encoder_2, 
    text_encoder_3, 
    encoder_name, 
    epoch,
    config,
    device="cuda"
):
    
    orig_encoder = pipeline.text_encoder
    orig_encoder_2 = pipeline.text_encoder_2
    orig_encoder_3 = pipeline.text_encoder_3
    
    pipeline.text_encoder = text_encoder.eval()
    pipeline.text_encoder_2 = text_encoder_2.eval()
    pipeline.text_encoder_3 = text_encoder_3.eval()
    
    out_dir = os.path.join(config.output_dir, f"epoch_{epoch+1}_{encoder_name}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Set up autocast
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    
    try:
        with ctx, torch.no_grad():
            for idx, prompt in enumerate(prompt_list):
                gen = torch.Generator(device=device).manual_seed(config.generation_seed)
                result = pipeline(
                    prompt=[prompt], 
                    num_inference_steps=config.num_inference_steps, 
                    guidance_scale=config.guidance_scale, 
                    generator=gen
                )
                image = result.images[0]
                file_name = f"{idx+1}_{prompt[:40].replace(' ', '_')}.png"
                image.save(os.path.join(out_dir, file_name))
    finally:
        # Restore original encoders
        pipeline.text_encoder = orig_encoder
        pipeline.text_encoder_2 = orig_encoder_2
        pipeline.text_encoder_3 = orig_encoder_3
