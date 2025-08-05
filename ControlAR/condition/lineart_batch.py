import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import argparse

norm_layer = nn.InstanceNorm2d
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class LineArt(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True):
        super(LineArt, self).__init__()

        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True)
        ]
        self.model0 = nn.Sequential(*model0)

        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        model4 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7)
        ]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        """
        input: tensor (B, C, H, W)
        output: tensor (B, 1, H, W), scale [0,1]
        """
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out

def process_images(input_folder, output_folder, batch_size=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/data/checkpoints/condition/lineart_model.pth'
    
    lineart_net = LineArt()
    lineart_net.load_state_dict(torch.load(model_path, map_location=device))
    lineart_net = lineart_net.to(device).eval()
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    
    for batch in tqdm(batches, desc="Processing Batches"):
        images = []
        image_names = []
        for img_name in batch:
            img_path = os.path.join(input_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float()
            images.append(tensor_img)
            image_names.append(img_name)
        if not images:
            continue
        input_batch = torch.stack(images).to(device)
        detected_maps = 1-lineart_net(input_batch) 
        for i, img_name in enumerate(image_names):
            output_image = detected_maps[i, 0].cpu().detach().numpy()
            cv2.imwrite(os.path.join(output_folder, img_name), 255 * output_image)
    
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
    process_images(input_folder, output_folder, batch_size=12)
