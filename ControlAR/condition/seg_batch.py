import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from mmseg.apis import init_model, inference_model
import argparse

class SegmentationDetector:
    def __init__(self, config_file, checkpoint_file, device='cuda:0'):
        from mmseg.apis import init_model
        self.model = init_model(config_file, checkpoint_file, device=device)

    def __call__(self, img_path):
        result = inference_model(self.model, img_path)
        segmentation_map = result.pred_sem_seg.data[0].cpu().numpy().squeeze().astype(np.uint8)
        return segmentation_map


def process_segmentation_images(input_folder, output_folder, config_file, checkpoint_file, batch_size=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

    detector = SegmentationDetector(config_file, checkpoint_file)

    for batch in tqdm(batches, desc="Processing Batches"):
        for img_name in batch:
            img_path = os.path.join(input_folder, img_name)

            result = detector(img_path)

            output_path = os.path.join(output_folder, img_name)
            Image.fromarray(result).save(output_path)

    print("Segmentation processing completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert single JSON to ms-swift standard format")
    parser.add_argument("--input_folder", required=True, type=str, help="Path to input folder containing images")
    parser.add_argument("--output_folder", required=True,type=str, help="Path to output folder for processed images")
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    config_file = 'mmsegmentation/configs/deeplabv3/deeplabv3_r101-d8_4xb4-320k_coco-stuff164k-512x512.py'
    checkpoint_file = 'ControlAR/evaluations/deeplabv3_r101-d8_512x512_4x4_320k_coco-stuff164k_20210709_155402-3cbca14d.pth'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_segmentation_images(input_folder, output_folder, config_file, checkpoint_file, batch_size=4)
