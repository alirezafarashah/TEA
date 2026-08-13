import os
import argparse
import torch
import clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

# --- Default Configuration ---
DEFAULT_ROOT_DIR = './outputs/coco_results/'
DEFAULT_CSV_PATH = './datasets/coco_30k_10k.csv'
DEFAULT_OUTPUT_CSV = 'clip_evaluation_results_sd35.csv'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"  # or "ViT-L/14" depending on what you use

def main():
    parser = argparse.ArgumentParser(description="CLIP score evaluation for generated images.")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_ROOT_DIR,
                        help="Root directory containing generated images (may have subdirectories).")
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH,
                        help="Path to CSV file with coco_id and prompt columns.")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV,
                        help="Path to save the output CSV with per-directory CLIP scores.")
    args = parser.parse_args()

    ROOT_DIR = args.root_dir
    CSV_PATH = args.csv_path
    output_csv = args.output_csv
    print(f"Running on: {DEVICE}")
    
    # 1. Load CLIP Model
    print(f"Loading CLIP model: {MODEL_NAME}...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    
    # 2. Load and Prepare CSV Data
    print("Loading CSV and creating lookup dictionary...")
    try:
        df = pd.read_csv(CSV_PATH)
        # Ensure coco_id is string for consistent matching
        # Create a dictionary for O(1) lookup: coco_id -> prompt
        id_to_prompt = dict(zip(df['coco_id'].astype(str), df['prompt']))
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    results = []

    # 3. Traverse Directories
    # os.walk automatically handles the difference between 'orig' (files immediately) 
    # and 'adv_...' (files in subdirectories)
    for current_root, dirs, files in os.walk(ROOT_DIR):
        
        # Filter for valid image files in the current directory
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # If this directory has no images, skip it (e.g., the parent 'adv_' folders usually only contain subdirs)
        if not image_files:
            continue

        print(f"\nProcessing: {current_root}")
        print(f"Found {len(image_files)} images.")

        scores = []
        
        # Process images in batches usually helps, but one-by-one is safer for varying sizes
        # We will do one-by-one for simplicity and robustness
        for filename in tqdm(image_files, desc="Calculating Similarity"):
            # 4. Parse Coco ID from filename
            # Supports:
            #   "12345.png"              -> coco_id = "12345"
            #   "coco_1k_des_12345.png"  -> coco_id = "12345"
            stem = os.path.splitext(filename)[0]
            prefix = "coco_1k_des_"
            if stem.startswith(prefix):
                coco_id = stem[len(prefix):]
            else:
                coco_id = stem
            
            # 5. Retrieve Prompt
            if coco_id not in id_to_prompt:
                # Optional: Un-comment to warn about missing IDs
                # print(f"Warning: ID {coco_id} not found in CSV.")
                continue
                
            text_prompt = id_to_prompt[coco_id]
            
            try:
                # 6. Preprocess Image and Text
                image = preprocess(Image.open(os.path.join(current_root, filename))).unsqueeze(0).to(DEVICE)
                text = clip.tokenize([text_prompt], truncate=True).to(DEVICE)
                
                # 7. Calculate Similarity
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    
                    # Normalize
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    # Cosine similarity
                    similarity = (image_features @ text_features.T).item()
                    scores.append(similarity)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # 8. Aggregate Results for this Directory
        if scores:
            avg_score = np.mean(scores)
            results.append({
                'directory': current_root,
                'avg_clip_score': avg_score,
                'num_images': len(scores)
            })
            print(f"Directory Avg: {avg_score:.4f}")

    # 9. Final Report
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    results_df = pd.DataFrame(results)
    
    # Sort for better readability (e.g., by directory name)
    results_df = results_df.sort_values('directory')
    
    # Print to console
    print(results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to '{output_csv}'")

if __name__ == "__main__":
    main()