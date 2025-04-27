"""
- Cosine Similarity (Feature Embeddings): Measures the similarity between deep feature vectors extracted from a pre-trained model. Lower values indicate stronger perturbations.
- Identity Distance (ID Score): Uses a face recognition model to compute identity preservation between two images. Higher values suggest greater disruption.
- CLIP Similarity: Uses OpenAI's CLIP model to compare how well the images align with textual descriptions. Lower scores indicate a loss of semantic coherence.
- BLEU Score (for captions): Evaluates the difference in generated textual descriptions of the two images using an image captioning model. Lower scores suggest greater disruption of image semantics.
"""

import os
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import re

clip_score_fn = partial(clip_score, model_name_or_path='openai/clip-vit-base-patch32')

def extract_features(image, model, transform):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    return features.flatten()

def cosine_sim(image1, image2, model, transform):
    feat1 = extract_features(image1, model, transform)
    feat2 = extract_features(image2, model, transform)
    return cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

def clip_similarity(image1, image2, model, processor):
    inputs1 = processor(images=image1, return_tensors="pt")
    inputs2 = processor(images=image2, return_tensors="pt")
    with torch.no_grad():
        features1 = model.get_image_features(**inputs1)
        features2 = model.get_image_features(**inputs2)
    return cosine_similarity(features1, features2).item()

def clip_score(image, prompt):
    prompts = [prompt]
    image_int = np.array(image).astype('uint8')
    images_int = np.expand_dims(image_int, axis=0)
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()

    return round(float(clip_score), 4)

def generate_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.batch_decode(output, skip_special_tokens=True)[0]

def bleu_score(caption1, caption2):
    return sentence_bleu([caption1.split()], caption2.split())

def evaluate_semantic(image1_path, image2_path):
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        raise FileNotFoundError(f"One or both image files are missing: {image1_path}, {image2_path}")

    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")
    img1 = img1.resize((512, 512))
    img2 = img2.resize((512, 512))

    # Regex to get the prompt file from an image name
    prompt = ''
    prompt_dir = ''
    pattern = r"^.*?People250"
    match = re.match(pattern, image2_path)
    prompt_dir = match.group() + '/prompts/'
    match = re.search(r"^(.*\/)(\d+)(?=_)", image2_path)
    prompt_dir = prompt_dir + match.group(2) + '_prompts.txt'

    with open(prompt_dir, "r") as file:
        prompt = file.readline().strip()  # Reads the first line and removes any trailing newline characters

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    caption1 = generate_caption(img1, blip_processor, blip_model)
    caption2 = generate_caption(img2, blip_processor, blip_model)

    results = {
        "Image 1": image1_path,
        "Image 2": image2_path,
        "Cosine Similarity": cosine_sim(img1, img2, resnet, transform),
        "CLIP Similarity": clip_similarity(img1, img2, clip_model, clip_processor),
        "CLIP Score": clip_score(img2, prompt),
        "BLEU Score": bleu_score(caption1, caption2)
    }
    df = pd.DataFrame([results])
    numeric = ["Cosine Similarity", "CLIP Similarity", "CLIP Score", "BLEU Score"]
    df[numeric] = df[numeric].astype(np.float64)
    return df
