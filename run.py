"""
Usage:
    python run_tests_on_folders.py --folder1 path/to/folder1 --folder2 path/to/folder2 --output results.csv

Arguments:
    --folder1: Path to the first folder containing images.
    --folder2: Path to the second folder containing images.
    --output: Path to the output CSV file where results will be stored.
"""

import os
import argparse
import subprocess

TEST_SCRIPTS = {
    "statistical": "StatisticalDifferences.py",
    "semantic": "SemanticCoherence.py",
    "realism": "RealismEvaluation.py"
}

def get_image_pairs(folder1, folder2):
    images1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('png', 'jpg', 'jpeg'))])
    images2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('png', 'jpg', 'jpeg'))])
    
    if len(images1) != len(images2):
        raise ValueError("Mismatch: The two folders must have the same number of images.")
    
    return list(zip(images1, images2))

def run_test(script, image1, image2, output_csv):
    cmd = ["python", script, "--image1", image1, "--image2", image2, "--output", output_csv]
    subprocess.run(cmd, check=True)

def main(folder1, folder2, output_csv):
    image_pairs = get_image_pairs(folder1, folder2)
    
    for image1, image2 in image_pairs:
        print(f"Processing pair: {image1} ↔ {image2}")
        for test_type, script in TEST_SCRIPTS.items():
            print(f"Running {test_type} test...")
            run_test(script, image1, image2, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", type=str, required=True)
    parser.add_argument("--folder2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    main(args.folder1, args.folder2, args.output)