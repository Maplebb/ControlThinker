import os
import sys
from pathlib import Path
current_dir = Path(__file__).resolve()
parent_two_levels_up = current_dir.parents[2]
sys.path.append(str(parent_two_levels_up))
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import json
from language.t5 import T5Embedder
from datasets import Dataset
from tqdm import tqdm


def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def main(json_path, output_path, prompt_num, flan_t5_path,  device='cuda'):
    os.makedirs(output_path, exist_ok=True)

    # Load T5 model
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=flan_t5_path,
        dir_or_name="flan-t5-xl",
        model_max_length=120,
    )

    data = load_json_data(json_path)

    

    for img_name, content in tqdm(data.items()):

        save_dir = os.path.join(output_path, img_name.replace('.png', ''))
        os.makedirs(save_dir, exist_ok=True)
        if all(os.path.exists(os.path.join(save_dir, f'{i}.npz')) for i in range(prompt_num)):
            print(f"skip batch")
            continue
    
        
        prompts = content['prompts']
        if type(prompts) is not list:
            prompts = [prompts]
        prompts = prompts[:prompt_num]
        if len(prompts) < prompt_num:
            prompts += [content['original_prompt']] * (prompt_num - len(prompts))
        print(prompts)

        with torch.no_grad():
            emb, mask = t5_model.get_text_embeddings(prompts)
            mask = mask.sum(axis=1)
            print(mask)
            for i in range(len(emb)):
                temp_emb = emb[i:i+1, :int(mask[i].item())].to(dtype=torch.float32).cpu().numpy()
                print(temp_emb.shape)
                caption_dict = {
                    'prompt': prompts[i],
                    'caption_emb': temp_emb,
                }

                np.savez(os.path.join(save_dir, f'{i}.npz'), **caption_dict)

        caption_dict = {
            'prompts': prompts,
            'embeddings': emb
        }

        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, required=True, help='Path to the JSON file containing prompts.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save caption embeddings.')
    parser.add_argument('--prompt-num', type=int, default=10)
    parser.add_argument('--flan-t5-path', type=str, required=True, help='Path to the Flan-T5 model directory.')
    args = parser.parse_args()

    main(args.json_path, args.output_path, args.prompt_num, args.flan_t5_path, device='cuda' if torch.cuda.is_available() else 'cpu')
