import os
import shutil
import lpips
import torch
import argparse
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModel
from PIL import Image
import re
import json
import numpy as np

# load model
device = "cuda"
processor_name_or_path = "/data/checkpoints/controlar/pickscore/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "/data/checkpoints/controlar/pickscore/model"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs(images, prompt):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # # get probabilities if you have multiple images to choose from
        # probs = torch.softmax(scores, dim=-1)
    
    return scores.cpu().numpy()

def preprocess_and_truncate(text, max_tokens=77):
    # 删除URL链接
    text = re.sub(r'https?://\S+', '', text)
    # 删除 #标签
    text = re.sub(r'#\S+', '', text)
    # 删除多余空白符
    text = re.sub(r'\s+', ' ', text).strip()
    # 按空格切分，保证不超过 max_tokens
    tokens = text.split()
    return ' '.join(tokens[:max_tokens])

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 统一尺寸（可根据需求修改）
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # 添加 batch 维度

def load_images_batch(image_paths):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    images = [transform(Image.open(p).convert("RGB")) for p in image_paths]
    return torch.stack(images)  # shape: (batch_size, 3, 256, 256)



def find_best_match(ctl_gt_folder, ctl_candidates_folder, output_folder, img_candidates_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(os.path.join(os.path.dirname(ctl_gt_folder), "captions.json"), "r", encoding="utf-8") as file:
        captions = json.load(file)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    for ctl_gt_image_name in os.listdir(ctl_gt_folder):
        output_path = os.path.join(output_folder, ctl_gt_image_name)
        if os.path.exists(output_path):
            print(f"Skipping {ctl_gt_image_name}")
            continue
        ctl_gt_image_path = os.path.join(ctl_gt_folder, ctl_gt_image_name)
        print("ctl_gt_image_path", ctl_gt_image_path)
        ctl_gt_image = load_image(ctl_gt_image_path).to(device)
        caption = captions[ctl_gt_image_name]
        print(caption)

        gt_basename = os.path.splitext(ctl_gt_image_name)[0]
        candidate_images_names = [f for f in os.listdir(ctl_candidates_folder) if f"sample{gt_basename}_" in f]
        candidate_images_names = sorted(candidate_images_names, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        if not candidate_images_names:
            print(f"Warning: No candidates found for {ctl_gt_image_name}")
            continue

        ctl_candidate_paths = [os.path.join(ctl_candidates_folder, name) for name in candidate_images_names]
        candidate_image_paths = [os.path.join(img_candidates_folder, name) for name in candidate_images_names]
        print("ctl_candidate_paths", ctl_candidate_paths)
        print("candidate_image_paths", candidate_image_paths)

        ctl_candidates_batch = load_images_batch(ctl_candidate_paths).to(device)  # shape: (N, 3, 256, 256)
        candidate_images_batch = []
        for candidate_image_path in candidate_image_paths:
            candidate_images_batch.append(Image.open(candidate_image_path))

        # 一次性计算LPIPS
        with torch.no_grad():
            ctl_gt_image_batch = ctl_gt_image.repeat(len(candidate_images_names), 1, 1, 1)  # 复制GT图到相同batch
            lpips_scores = loss_fn_alex(ctl_gt_image_batch, ctl_candidates_batch).squeeze().cpu().numpy()

        # 一次性计算pickscore scores
        pickscore_scores = calc_probs(candidate_images_batch, caption)

        # 分数标准化（0~1）
        lpips_scores = (lpips_scores - lpips_scores.min()) / (lpips_scores.max() - lpips_scores.min())
        pickscore_scores = (pickscore_scores - pickscore_scores.min()) / (pickscore_scores.max() - pickscore_scores.min()) + 1e-6
        
        print(f"lpips_scores {args.lpips_score_scale * lpips_scores}, \npickscore_scores {args.pickscore_score_scale * pickscore_scores}")
        total_scores = args.pickscore_score_scale * pickscore_scores - args.lpips_score_scale * lpips_scores
        print(f"total_scores {total_scores}")
        
        best_index = np.argmax(total_scores)
        best_image_path = candidate_image_paths[best_index]
        output_path = os.path.join(output_folder, ctl_gt_image_name)

        shutil.copy(best_image_path, output_path)
        print(f"Saved best match for {ctl_gt_image_name} -> {os.path.basename(best_image_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best matching image based on LPIPS.")
    parser.add_argument("--ctl_gt_folder", type=str, required=True, help="Path to the ground truth control images folder.")
    parser.add_argument("--ctl_candidates_folder", type=str, required=True, help="Path to the candidates control images folder.")
    parser.add_argument("--img_candidates_folder", type=str, required=True, help="Path to the candidates images folder.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the best matched images.")
    parser.add_argument("--pickscore_score_scale", type=float, default=1.0)
    parser.add_argument("--lpips_score_scale", type=float, default=1.0)
    args = parser.parse_args()
    
    find_best_match(args.ctl_gt_folder,  args.ctl_candidates_folder, args.output_folder, args.img_candidates_folder)