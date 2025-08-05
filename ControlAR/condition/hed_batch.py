import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from utils import annotator_ckpts_path
import requests

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5

class HEDdetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
        modelpath = os.path.join(annotator_ckpts_path, "ControlNetHED.pth")
        os.makedirs(annotator_ckpts_path, exist_ok=True)
        if not os.path.exists(modelpath):
            response = requests.get(remote_model_path, stream=True)
            response.raise_for_status()
            with open(modelpath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        self.netNetwork = ControlNetHED_Apache2().float()
        self.netNetwork.load_state_dict(torch.load(modelpath, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    def __call__(self, input_image):
        B, C, H, W = input_image.shape
        image_hed = input_image
        edges = self.netNetwork(image_hed)
        edges = [F.interpolate(e, size=(H, W), mode='bilinear', align_corners=False).squeeze(1) for e in edges]
        edges = torch.stack(edges, dim=1)
        edge = 1 / (1 + torch.exp(-torch.mean(edges, dim=1)))
        edge = (edge * 255.0).clamp(0, 255)
        return edge

def process_images(input_folder, output_folder, batch_size=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hed_detector = HEDdetector().to(device).eval()
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    
    for batch in tqdm(batches, desc="Processing Batches"):
        images = []
        image_shapes = []
        image_names = []
        
        for img_name in batch:
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            image_shapes.append((H, W))
            image_names.append(img_name)
            input_tensor = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)
            images.append(input_tensor)
        
        if not images:
            continue
        
        input_batch = torch.stack(images).to(device)
        detected_maps = hed_detector(input_batch)
        # print(detected_maps[0])
        
        for i, img_name in enumerate(image_names):
            output_image = detected_maps[i].cpu().detach().numpy()
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, output_image)
    
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
    process_images(input_folder, output_folder, batch_size=16)
