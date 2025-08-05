import os
import torch
import cv2
import numpy as np
from PIL import Image, PngImagePlugin  
from transformers import DPTImageProcessor, DPTForDepthEstimation
import tqdm
import argparse

PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024 ** 2) 

parser = argparse.ArgumentParser(description="Process images for depth estimation.")
parser.add_argument("--start_idx", type=int, default=0, help="Start index of images to process.")
parser.add_argument("--end_idx", type=int, default=1000000, help="End index of images to process.")
parser.add_argument("--input_folder", required=True, type=str, help="Path to input folder containing images")
parser.add_argument("--output_folder", required=True,type=str, help="Path to output folder for processed images")
parser.add_argument("--dpt-large-path", required=True,type=str, help="Path to output folder for processed images")
args = parser.parse_args()
input_folder = args.input_folder
output_folder = args.output_folder

start_idx = args.start_idx
end_idx = args.end_idx
dpt_large_path = args.dpt_large_path

processor = DPTImageProcessor.from_pretrained(dpt_large_path)
model = DPTForDepthEstimation.from_pretrained(dpt_large_path)
model = model.cuda()

image_dir = input_folder
output_dir = output_folder
os.makedirs(output_dir, exist_ok=True)
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
print(len(image_files))

batch_size = 40

# for i in tqdm.tqdm(range(0, len(image_files), batch_size)):
for i in tqdm.tqdm(range(max(0,start_idx), min(len(image_files),end_idx), batch_size)):
    batch_files = image_files[i:i+batch_size]
    pil_images = []
    if all(os.path.exists(os.path.join(output_dir, f)) for f in batch_files):
        print(f"Skipping batch starting at index {i}, all files already exist.")
        continue
    
    for file in batch_files:
        path = os.path.join(image_dir, file)
        image = Image.open(path)
        img = cv2.imread(path)
        inputs_temp = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
        inputs_temp = 2 * (inputs_temp / 255 - 0.5)
        pil_images.append(image)
    
    inputs = processor(images=pil_images, return_tensors="pt", size=(512, 512))
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # shape: (batch, H, W)
    
    for j, file in enumerate(batch_files):
        depth = predicted_depth[j].cpu().numpy()
        depth = (depth * 255 / depth.max())
        output_path = os.path.join(output_dir, file)
        cv2.imwrite(output_path, depth)
        print(f"Saved {output_path}")
