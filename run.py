"""
Usage:
    python run.py -f1 path/to/folder1 -f2 path/to/folder2 -name experiment_name (test flags)

Arguments:
    Required:
        -f1, --folder1: Path to the first folder containing images.
        -f2, --folder2: Path to the second folder containing images.
        -n, --name: Path to output folder + experiment name.
    Optional:
        -st, --statistical: Evaluate statistical differences between images.
        -se, --semantic: Evaluate semantic coherence between images.
        -r, --realism: Evaluate realism of images.

Returns:
    2 CSV files for each evaluation type:
        - {name}_test_type.csv: Detailed results for each image pair.
        - {name}_test_type_summary.csv: Summary statistics for the evaluation type across all images.
"""

import os
import argparse
import pandas as pd
import numpy as np

from  StatisticalDifferences import evaluate_statistical
from RealismEvaluation import evaluate_images
from SemanticCoherence import evaluate_semantic

def get_image_pairs(folder1, folder2):
    images1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('png', 'jpg', 'jpeg'))])
    images2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('png', 'jpg', 'jpeg'))])
    
    if len(images1) != len(images2):
        raise ValueError("Mismatch: The two folders must have the same number of images.")
    
    return list(zip(images1, images2))

def main(folder1, folder2, output_csv):
    image_pairs = get_image_pairs(folder1, folder2)
    stat_scores = pd.DataFrame(columns=["Image 1", "Image 2", "MSE", "PSNR", "SSIM", "LPIPS", "KLD", "FID"])
    sem_scores = pd.DataFrame(columns=["Image 1", "Image 2", "Cosine Similarity", "CLIP Similarity", "CLIP Score", "BLEU Score"])
    real_scores = pd.DataFrame(columns=["Image 1", "Image 2", "BRISQUE", "NIQE", "MS-SSIM", "Deepfake Detector Confidence"])
    
    for image1, image2 in image_pairs:
        print(f"Processing pair: {image1} ↔ {image2}")
        if args.statistical:
            img_stats = evaluate_statistical(image1, image2)
            stat_scores = pd.concat([stat_scores, img_stats], ignore_index=True)
        if args.semantic:
            sem_score = evaluate_semantic(image1, image2)
            sem_scores = pd.concat([sem_scores, sem_score], ignore_index=True)
        if args.realism:
            real_score = evaluate_images(image1, image2)
            real_scores = pd.concat([real_scores, real_score], ignore_index=True)
    if args.statistical:
        stat_scores.to_csv(f"{output_csv}_statistical", index=False, mode='w', header=True)
        stat_scores.describe(include=[np.number]).to_csv(f"{output_csv}_statistical_summary", index=True, mode='w', header=True)    
    if args.semantic:
        sem_scores.to_csv(f"{output_csv}_semantic", index=False, mode='w', header=True)
        sem_scores.describe(include=[np.number]).to_csv(f"{output_csv}_semantic_summary", index=True, mode='w', header=True)
    if args.realism:
        real_scores.to_csv(f"{output_csv}_realism", index=False, mode='w', header=True)
        real_scores.describe(include=[np.number]).to_csv(f"{output_csv}_realism_summary", index=True, mode='w', header=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1","--folder1", type=str, required=True)
    parser.add_argument("-f2", "--folder2", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-st", "--statistical", action="store_true")
    parser.add_argument("-se", "--semantic", action="store_true")
    parser.add_argument("-r", "--realism", action="store_true")
    args = parser.parse_args()
    
    main(args.folder1, args.folder2, args.name)
