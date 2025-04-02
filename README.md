# Deepfake Disruption Evaluator
A set evaluation metrics to measure the performance of adversarial-noise-based models used to disrupt the outputs of deepfaking and inpainting models.
## 📂 Files
### `StatisticalDifferences.py`
Contains metrics measuring the statistical differences between two images. Can be used to measure the visibility of adversarial noise used to protect images; also useful for measuring disruption in protected deepfake output compared to unprotected generation.
The following metrics are included:
- MSE (Mean Squared Error): Measures pixel-wise error. Higher values indicate greater distortion.
- PSNR (Peak Signal-to-Noise Ratio): Assesses image quality; higher values mean less distortion.
- SSIM (Structural Similarity Index Measure): Evaluates perceived quality by comparing structures. Values close to 1 indicate high similarity.
- LPIPS (Learned Perceptual Image Patch Similarity): Uses deep learning features to measure perceptual similarity. Lower values indicate more similar images.
- KLD (Kullback-Leibler Divergence): Compares the pixel intensity distributions between images. Higher values suggest greater distributional shifts.
- FID (Fréchet Inception Distance): Measures feature distribution differences using an InceptionV3 model. Lower values indicate greater similarity.
### `SemanticCoherence.py`
Evaluates how closely the textual descriptions of two images match. Useful for assessing disruptions to scene context that may not be captured by statistical metrics.
The following metrics are included:
- Cosine Similarity (Feature Embeddings): Measures the similarity between deep feature vectors extracted from a pre-trained model. Lower values indicate stronger perturbations.
- Identity Distance (ID Score): Uses a face recognition model to compute identity preservation between two images. Higher values suggest greater disruption.**This metric is only relevant for images clearly containing faces**
- CLIP Similarity: Uses OpenAI's CLIP model to compare how well the images align with textual descriptions. Lower scores indicate a loss of semantic coherence.
- CLIP Score: Measures the compatibility between a generated image and its prompt by calculating the cosine similarity between their CLIP embeddings.
- BLEU Score (for captions): Evaluates the difference in generated textual descriptions of the two images using an image captioning model. Lower scores suggest greater disruption of image semantics.
### `RealismEvaluation.py`
Assesses how 'realistic' an image appears, attempting to be representative of human perception. Useful for evaluating deepfakes and the disruption caused by adversarial noise.
The following metrics are included:
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator): A no-reference image quality metric that assesses distortions without needing a reference image.
- NIQE (Natural Image Quality Evaluator): Measures how "natural" an image appears based on statistical modeling of real-world images.
- MS-SSIM (Multiscale Structural Similarity Index): Extends SSIM by evaluating image quality across multiple scales and resolutions.
- Deepfake Detector Confidence: Uses a state-of-the-art deepfake detection model to assess how likely an image is real or fake.

## 📝 Usage
`python run.py -f1 path/to/folder1 -f2 path/to/folder2 -name experiment_name (test flags)`

Cuda acceleration is recommended for faster processing on large datasets.

### Arguments:
#### Required:
- `-f1`, `--folder1`: Path to the first folder containing images.
- `-f2`, `--folder2`: Path to the second folder containing images.
- `-n`, `--name`: Path to output folder + experiment name.

#### Optional:
- `-st`, `--statistical`: Evaluate statistical differences between images.
- `-se`, `--semantic`: Evaluate semantic coherence between images.
- `-r`, `--realism`: Evaluate realism of images.

### Returns:
2 CSV files for each evaluation type:
- `{name}_test_type.csv`: Detailed results for each image pair.
- `{name}_test_type_summary.csv`: Summary statistics for the evaluation type across all images.

**Important Note:** The Realism Evaluation uses some metrics that are no-reference. These metrics all operate on the second image in the pair, i.e. those in the folder specified by -f2. Therefore when making a comparison between between a real image and a deepfake, the real image should be in the folder specified by -f1 and the deepfake in the folder specified by -f2. Likewise, when comparing a deepfake to a protected deepfake, the deepfake should be in the folder specified by -f1 and the protected deepfake in the folder specified by -f2.

### Example Usage
To evaluate both statistical differences and semantic coherence between an unprotected deepfake and a protected deepfake:
`python run.py -f1 path/to/unprotected -f2 path/to/protected -n path/to/output/experiment_name -st -se`

## 📚 Related Repositories 

- [People250 Dataset](https://github.com/Alf4ed/fourth-year-project-dataset): The People250 dataset consists of 250 images containing people, along with corresponding prompts and masks. This dataset was used for testing and evaluating our deepfake protection model against existing work.
- [ Diffusion Based Adversarial Perturbations for Disrupting Deepfake Generation](https://github.com/JakubCzarlinski/fourth-year-project): The main project repository containing the code for our deepfake protection model.
