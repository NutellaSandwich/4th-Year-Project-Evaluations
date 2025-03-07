"""
- MSE (Mean Squared Error): Measures pixel-wise error. Higher values indicate greater distortion.
- PSNR (Peak Signal-to-Noise Ratio): Assesses image quality; higher values mean less distortion.
- SSIM (Structural Similarity Index Measure): Evaluates perceived quality by comparing structures. Values close to 1 indicate high similarity.
- LPIPS (Learned Perceptual Image Patch Similarity): Uses deep learning features to measure perceptual similarity. Lower values indicate more similar images.
- KLD (Kullback-Leibler Divergence): Compares the pixel intensity distributions between images. Higher values suggest greater distributional shifts.
- FID (Fréchet Inception Distance): Measures feature distribution differences using an InceptionV3 model. Lower values indicate greater similarity.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from scipy.linalg import sqrtm
from skimage.color import rgb2gray
from PIL import Image
import lpips

lpips_model = lpips.LPIPS(net='alex')

def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def psnr(image1, image2):
    mse_value = mse(image1, image2)
    if mse_value == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_value))

def ssim_score(image1, image2):
    return ssim(rgb2gray(image1), rgb2gray(image2), data_range=1)

def lpips_score(image1, image2):
    transform = transforms.Compose([transforms.ToTensor()])
    img1_tensor = transform(image1).unsqueeze(0)
    img2_tensor = transform(image2).unsqueeze(0)
    return lpips_model(img1_tensor, img2_tensor).item()

def kl_divergence(image1, image2):
    hist1, _ = np.histogram(image1.flatten(), bins=256, density=True)
    hist2, _ = np.histogram(image2.flatten(), bins=256, density=True)
    return entropy(hist1 + 1e-10, hist2 + 1e-10)

def fid_score(image1, image2, model=None, eps=1e-6):
    if model is None:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img1_tensor = transform(image1).unsqueeze(0)
    img2_tensor = transform(image2).unsqueeze(0)

    with torch.no_grad():
        feat1 = model(img1_tensor).cpu().numpy().reshape(1, -1)
        feat2 = model(img2_tensor).cpu().numpy().reshape(1, -1)
    
    mu1, mu2 = np.mean(feat1, axis=0), np.mean(feat2, axis=0)
    
    def calculate_cov(features):
        n_samples = features.shape[0]
        diff = features - np.mean(features, axis=0)
        cov = (diff.T @ diff) / max(n_samples - 1, 1)
        cov += eps * np.eye(cov.shape[0])  
        return cov
    
    sigma1 = calculate_cov(feat1)
    sigma2 = calculate_cov(feat2)
    
    ssdiff = np.sum((mu1 - mu2) ** 2)
    
    cov_product = sigma1 @ sigma2
    covmean = sqrtm(cov_product)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    covmean = np.maximum(covmean, 0)
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def evaluate_statistical(image1_path, image2_path):
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        raise FileNotFoundError(f"One or both image files are missing: {image1_path}, {image2_path}")

    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")
    img1 = img1.resize((512, 512))
    img2 = img2.resize((512, 512))
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    results = {
        "Image 1": image1_path,
        "Image 2": image2_path,
        "MSE": mse(img1_np, img2_np),
        "PSNR": psnr(img1_np, img2_np),
        "SSIM": ssim_score(img1_np, img2_np),
        "LPIPS": lpips_score(img1, img2),
        "KLD": kl_divergence(img1_np, img2_np),
        "FID": fid_score(img1, img2),
    }
    
    df = pd.DataFrame([results])
    numeric = ["MSE", "PSNR", "SSIM", "LPIPS", "KLD", "FID"]
    df[numeric] = df[numeric].astype(np.float64)
    
    return df
