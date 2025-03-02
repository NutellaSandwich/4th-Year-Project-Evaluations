"""
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator): A no-reference image quality metric that assesses distortions without needing a reference image.
- NIQE (Natural Image Quality Evaluator): Measures how "natural" an image appears based on statistical modeling of real-world images.
- MS-SSIM (Multiscale Structural Similarity Index): Extends SSIM by evaluating image quality across multiple scales and resolutions.
- Deepfake Detector Confidence: Uses a state-of-the-art deepfake detection model to assess how likely an image is real or fake.
"""

import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from brisque import BRISQUE
import cv2
from skimage import img_as_float
from skimage.restoration import estimate_sigma
from skimage.measure import shannon_entropy
import piq

brisque_model = BRISQUE()

def brisque_score(image):
    image_tensor = transforms.ToTensor()(image).unsqueeze(0) 
    return piq.brisque(image_tensor, data_range=1.0).item()

def calculate_niqe(image):
    image_float = img_as_float(image)
    sigma_estimation = np.mean(estimate_sigma(image_float, channel_axis=-1))
    entropy = shannon_entropy(image_float)
    niqe_score = entropy / (sigma_estimation + 1e-10)
    return niqe_score

def ms_ssim_score(image1, image2):
    return ssim(rgb2gray(image1), rgb2gray(image2), data_range=255, multichannel=True)

def deepfake_detector_confidence(image, model, transform, device):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return torch.sigmoid(output).item()

def evaluate_images(image1_path, image2_path, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        raise FileNotFoundError(f"One or both image files are missing: {image1_path}, {image2_path}")

    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")
    img1_cv = cv2.imread(image1_path)
    img2_cv = cv2.imread(image2_path)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    deepfake_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    deepfake_model.classifier = torch.nn.Linear(deepfake_model.classifier[1].in_features, 1) 
    deepfake_model.to(device)
    deepfake_model.eval()

    results = [
        ["BRISQUE", image2_path, "", brisque_score(img2)],
        ["NIQE", image2_path, "", calculate_niqe(img2_cv)],
        ["MS-SSIM", image1_path, image2_path, ms_ssim_score(img1, img2)],
        ["Deepfake Detector Confidence", image2_path, "", deepfake_detector_confidence(img2, deepfake_model, transform, device)]
    ]

    df = pd.DataFrame(results, columns=["Metric", "Image 1", "Image 2", "Score"])
    df.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image realism and distortions")
    parser.add_argument("--image1", type=str, required=True, help="Path to the first image")
    parser.add_argument("--image2", type=str, required=True, help="Path to the second image")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    evaluate_images(args.image1, args.image2, args.output)