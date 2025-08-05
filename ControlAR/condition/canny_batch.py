import os
import sys
current_directory = os.getcwd()
sys.path.append(current_directory)
import cv2
import torch
import numpy as np
from tqdm import tqdm
from condition.canny import CannyDetector
import argparse

class CannyDetector:
    def __call__(self, img, low_threshold=100, high_threshold=200):
        """
        input: array or tensor, shape (H,W,3)
        output: array, shape (H,W)
        """
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy().astype(np.uint8)
        return cv2.Canny(img, low_threshold, high_threshold)

def process_images(input_folder, output_folder, batch_size=4, low_threshold=100, high_threshold=200):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    
    detector = CannyDetector()
    
    for batch in tqdm(batches, desc="Processing Batches"):
        for img_name in batch:
            img_path = os.path.join(input_folder, img_name)
            # img = cv2.imread(img_path)
            from PIL import Image
            img = Image.open(img_path)
            img = np.array(img)
            if img is None:
                continue
            detected_map = detector(img, low_threshold, high_threshold)
            output_path = os.path.join(output_folder, img_name)
            # cv2.imwrite(output_path, detected_map)
            detected_map = Image.fromarray(detected_map)
            detected_map.save(output_path)
    print("Processing completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert single JSON to ms-swift standard format")
    parser.add_argument("--input_folder", required=True, type=str, help="Path to input folder containing images")
    parser.add_argument("--output_folder", required=True,type=str, help="Path to output folder for processed images")
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    process_images(input_folder, output_folder, batch_size=12, low_threshold=100, high_threshold=200)
