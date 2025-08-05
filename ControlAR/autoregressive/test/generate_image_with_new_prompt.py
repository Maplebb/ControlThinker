# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample.py
import warnings
warnings.filterwarnings('ignore')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image
import os
import sys
from pathlib import Path
current_dir = Path(__file__).resolve()
parent_two_levels_up = current_dir.parents[2]
sys.path.append(str(parent_two_levels_up))
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt_t2i import GPT_models
from autoregressive.models.generate import generate
from condition.hed import HEDdetector, nms
from condition.canny import CannyDetector
from autoregressive.test.metric import SSIM, F1score, RMSE
from condition.midas.depth import MidasDetector
# 移除分布式相关的引用
# import torch.distributed as dist
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from torch.utils.data import DataLoader
# 移除 DistributedSampler
# from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from functools import partial
from language.t5 import T5Embedder
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from condition.lineart import LineArt
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import Dataset
# from selector import ImageSelector  # 确保 ImageSelector 被正确导入

class T2IControlCode(Dataset):
    def __init__(self, args):
        self.text_embedding_dir = args.text_embedding_dir
        self.get_image = args.get_image
        self.get_prompt = args.get_prompt
        self.get_label = args.get_label
        self.control_type = args.condition_type
        if self.control_type == 'canny' or self.control_type == 'canny_base':
            self.get_control = CannyDetector()
        
        self.code_path = args.code_path
        code_file_path = os.path.join(self.code_path, 'code')
        file_num = len(os.listdir(code_file_path))
        self.code_files = [os.path.join(code_file_path, f"{i}.npy") for i in range(file_num)]
        
        if args.code_path2 is not None:
            self.code_path2 = args.code_path2
            code_file_path2 = os.path.join(self.code_path2, 'code')
            file_num2 = len(os.listdir(code_file_path2))
            self.code_files2 = [os.path.join(code_file_path2, f"{i}.npy") for i in range(file_num2)]
            self.code_files = self.code_files + self.code_files2

        self.image_size = args.image_size
        latent_size = args.image_size // args.downsample_size
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len

    def __len__(self):
        return len(self.code_files)

    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = 0
        return img, t5_feat_padding, attn_mask, valid

    def collate_fn(self, examples):
        
        code = torch.stack([example["code"] for example in examples])
        control =  torch.stack([example["control"] for example in examples])
        if self.control_type == 'canny' or self.control_type == 'canny_base':
            control = control.unsqueeze(1).repeat(1,3,1,1)
        caption_emb =  torch.stack([example["caption_emb"] for example in examples])
        attn_mask = torch.stack([example["attn_mask"] for example in examples])
        valid = torch.stack([example["valid"] for example in examples])
        if self.get_image:
            image = torch.stack([example["image"] for example in examples])
        if self.get_prompt:
            prompt = [example["prompt"] for example in examples]
        if self.control_type == "seg":
            label = torch.stack([example["label"] for example in examples])
            
        output = {}
        output['code'] = code
        output['control'] = control
        output['caption_emb'] = caption_emb
        output['attn_mask'] = attn_mask
        output['valid'] = valid
        if self.get_image:
            output['image'] = image
        if self.get_prompt:
            output['prompt'] = prompt
        if self.control_type == "seg":
            output['label'] = label
        return output

    def __getitem__(self, index):
        code_path = self.code_files[index]
        if self.control_type == 'seg':
            control_path = code_path.replace('code', 'control').replace('npy', 'png') 
            control = np.array(Image.open(control_path))/255
            control = 2*(control - 0.5)
        elif self.control_type == 'depth':
            control_path = code_path.replace('code', 'control_depth').replace('npy', 'png')
            control = np.array(Image.open(control_path))/255
            control = 2*(control - 0.5)
        caption_dir = os.path.join(self.text_embedding_dir,os.path.basename(code_path).replace('.npy', ''))
        image_path = code_path.replace('code', 'image').replace('npy', 'png')
        label_path = code_path.replace('code', 'label').replace('npy', 'png') 
        
        code = np.load(code_path)
        image = np.array(Image.open(image_path))
        
    
        caption_paths = sorted([os.path.join(caption_dir, each_path) for each_path in os.listdir(caption_dir) if ".npz" in each_path])
        captions = [np.load(caption_path) for caption_path in caption_paths]
        t5_feats = [torch.from_numpy(caption['caption_emb']) for caption in captions]
        prompt = [caption['prompt'] for caption in captions]
        t5_feat_lens = [t5_feat.shape[1] for t5_feat in t5_feats]
        feat_len = [min(self.t5_feature_max_len, t5_feat_len) for t5_feat_len in t5_feat_lens]
        t5_feat_padding = torch.zeros((len(t5_feat_lens), self.t5_feature_max_len, self.t5_feature_dim))
        emb_mask = torch.zeros((len(t5_feat_lens),self.t5_feature_max_len))
        attn_mask = torch.tril(torch.ones(len(t5_feat_lens),self.max_seq_length, self.max_seq_length))
        for i in range(len(t5_feat_lens)):
            t5_feat_padding[i, -feat_len[i]:] = t5_feats[i][:, :feat_len[i]]
            emb_mask[i, -feat_len[i]:] = 1
            T = self.t5_feature_max_len
            attn_mask[i, :, :T] = attn_mask[i, :, :T] * emb_mask[i].unsqueeze(0)
        eye_matrix = torch.eye(self.max_seq_length).unsqueeze(0).repeat(len(t5_feat_lens), 1, 1)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.to(torch.bool)
        valid = 1
        
        output = {}
        output['code'] = torch.from_numpy(code)
        if self.control_type == 'canny' or self.control_type == 'canny_base':
            output['control'] = torch.from_numpy(2*(self.get_control(image)/255 - 0.5))
        elif self.control_type == "seg":
            output['control'] = torch.from_numpy(control.transpose(2,0,1))
        elif self.control_type == "depth":
            output['control'] = torch.from_numpy(control.transpose(2,0,1))
        elif self.control_type == 'hed':
            output['control'] = torch.from_numpy(image.transpose(2,0,1))
        elif self.control_type == 'lineart':
            output['control'] = torch.from_numpy(image.transpose(2,0,1))
        output['caption_emb'] = t5_feat_padding
        output['attn_mask'] = attn_mask
        output['valid'] = torch.tensor(valid)
        output['image'] = torch.from_numpy(image.transpose(2,0,1))
        if self.get_prompt:
            output['prompt'] = prompt
        if self.control_type == "seg":
            output['label'] = torch.from_numpy(np.array(Image.open(label_path)))
        return output


def build_t2i_control_code(args):
    dataset = T2IControlCode(args)
    return dataset

def resize_image_to_16_multiple(image_path, condition_type='seg'):
    image = Image.open(image_path)
    width, height = image.size
    
    if condition_type == 'depth':  # The depth model requires a side length that is a multiple of 32
        new_width = (width + 31) // 32 * 32
        new_height = (height + 31) // 32 * 32
    else:
        new_width = (width + 15) // 16 * 16
        new_height = (height + 15) // 16 * 16

    resized_image = image.resize((new_width, new_height))
    return resized_image

def main(args):
    # Setup PyTorch:
    device = "cuda:0"
    device2 = "cuda:1"
    # selector = ImageSelector(device=device2, device_map={"": device2}, pretrained=args.reward_model_path)

    assert torch.cuda.is_available(), "This script requires at least one GPU."
    torch.set_grad_enabled(False)
    # 非分布式设置：直接使用单机的设备（如果有多 GPU，可通过 DataParallel 实现多卡并行）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 如果 device 类型为 cuda 但没有指定索引，则默认使用 0 号 GPU
    if device.type == 'cuda' and device.index is None:
        device = torch.device("cuda:0")
    torch.manual_seed(args.global_seed)
    print(f"Using device: {device}, seed: {args.global_seed}, world_size: 1.")
    
    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"Image tokenizer is loaded")

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        adapter_size=args.adapter_size,
        condition_type=args.condition_type,
    ).to(device=device, dtype=precision)
        
    _, file_extension = os.path.splitext(args.gpt_ckpt)
    if file_extension.lower() == '.safetensors':
        from safetensors.torch import load_file
        model_weight = load_file(args.gpt_ckpt)
        gpt_model.load_state_dict(model_weight, strict=False)
        gpt_model.eval()
    else:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        if "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "module" in checkpoint: # deepspeed
            model_weight = checkpoint["module"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception("please check model weight")
        gpt_model.load_state_dict(model_weight, strict=False)
        gpt_model.eval()
        del checkpoint
    print(f"GPT model is loaded")
    
    t5_model = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = build_t2i_control_code(args)
    # subset_dataset = Subset(dataset, list(range(1000)))  # 只取前 1000 个样本
    subset_dataset = dataset
    # 非分布式模式，不使用 DistributedSampler
    loader = DataLoader(
        subset_dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )
    
    if args.compile:
        print(f"Compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    
    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    date = os.path.split(os.path.dirname(os.path.dirname(os.path.dirname(args.gpt_ckpt))))[-1]
    folder_name = f"{model_string_name}-{date}-{ckpt_string_name}-size-{args.image_size}-{args.vq_model}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}"
    # 由于非分布式运行，直接由主进程创建文件夹
    if args.save_image:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(f"{args.sample_dir}/visualization", exist_ok=True)
        os.makedirs(f"{args.sample_dir}/annotations", exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    
    n = args.per_proc_batch_size
    total = 0  # 用于记录已处理 batch 数

    if args.condition_type == 'hed':
        get_condition = HEDdetector().to(device).eval()
    elif args.condition_type == 'canny' or args.condition_type == 'canny_base':
        get_condition = CannyDetector()
    elif args.condition_type == 'lineart':
        get_condition = LineArt()
        get_condition.load_state_dict(torch.load('condition/ckpts/lineart_model.pth', map_location=torch.device('cpu')))
        get_condition.to(device)

    for batch in tqdm(loader):
        batch_size = len(batch['prompt'])
        batch_indices = list(range(total, total + batch_size))

        if all(os.path.exists(f"{args.sample_dir}/intermediate/sample{idx}_9.png") for idx in batch_indices):
            total += batch_size
            print(f"Skipping batch {total - batch_size} to {total}")
            continue

        prompts = batch['prompt']  # shape: [batch_size, 10]
        condition_imgs = batch['control'].to(device)

        if args.condition_type in ['hed', 'lineart']:
            with torch.no_grad():
                condition_imgs = get_condition(condition_imgs.float())
                if args.condition_type == 'hed':
                    condition_imgs = condition_imgs.unsqueeze(1) / 255
                if args.condition_type == 'lineart':
                    condition_imgs = 1 - condition_imgs
                condition_imgs = condition_imgs.repeat(1, 3, 1, 1)
                condition_imgs = 2 * (condition_imgs - 0.5)

        if args.condition_type == 'seg':
            labels = batch['label']

        for iteration_count in range(len(prompts[0])):  # 改为 prompts 的第二维长度
            current_prompts = [prompts[i][iteration_count] for i in range(batch_size)]
            print(f"Current prompts[{iteration_count}]: {current_prompts}")

            caption_embs, emb_masks = t5_model.get_text_embeddings(current_prompts)

            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for caption_emb, emb_mask in zip(caption_embs, emb_masks):
                valid_num = int(emb_mask.sum().item())
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)

            new_caption_embs = torch.stack(new_caption_embs)
            c_indices = new_caption_embs * new_emb_masks[:, :, None]
            c_emb_masks = new_emb_masks

            qzshape = [len(c_indices), args.codebook_embed_dim, args.image_H // args.downsample_size, args.image_W // args.downsample_size]

            start_time = time.time()
            c_indices_copy = c_indices.clone().detach().to(device)
            c_emb_masks_copy = c_emb_masks.clone().detach().to(device)
            condition_imgs_copy = condition_imgs.clone().detach().to(device)

            with torch.no_grad():
                index_sample = generate(
                    gpt_model, c_indices_copy, (args.image_H // args.downsample_size) * (args.image_W // args.downsample_size),
                    c_emb_masks_copy, condition=condition_imgs_copy.to(precision),
                    cfg_scale=args.cfg_scale, cfg_interval=args.cfg_interval,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, sample_logits=True,
                )
                print(index_sample.shape)

            samples = vq_model.decode_code(index_sample, qzshape)

            # 保存中间结果
            for i, index in enumerate(batch_indices):
                if args.save_image:
                    image_filename = f"sample{index}_{iteration_count}.png"
                    image_folder = os.path.join(f"{args.sample_dir}/intermediate/")
                    os.makedirs(image_folder, exist_ok=True)
                    image_path = os.path.join(image_folder, image_filename)
                    save_image(samples[i], image_path, nrow=1, normalize=True, value_range=(-1, 1))

            end_time = time.time()
            print(f"Iteration {iteration_count} took {end_time - start_time}s")

        # 保存condition images
        for i, index in enumerate(batch_indices):
            annotation_folder = f"{args.sample_dir}/annotations/"
            os.makedirs(annotation_folder, exist_ok=True)
            if args.condition_type == 'seg':
                Image.fromarray(labels[i].numpy().astype('uint8'), mode='L').save(f"{annotation_folder}/{index}.png")
            else:
                save_image(condition_imgs[i, 0], f"{annotation_folder}/{index}.png", nrow=1, normalize=True, value_range=(-1, 1))

        total += batch_size




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # text_embedding
    parser.add_argument("--text_embedding_dir", type=str, required=True)
    
    # 模型相关参数
    parser.add_argument("--gpt_model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt_ckpt", type=str, default=None)
    parser.add_argument("--gpt_type", type=str, choices=['c2i', 't2i'], default="t2i", help="class-conditional or text-conditional")
    parser.add_argument("--from_fsdp", action='store_true')
    
    # 生成参数
    parser.add_argument("--cls_token_num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=False)

    # VQ 模型参数
    parser.add_argument("--vq_model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq_ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook_size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook_embed_dim", type=int, default=8, help="codebook dimension for vector quantization")

    # 图像参数
    parser.add_argument("--image_size", type=int, choices=[256, 384, 512, 768], default=512)
    parser.add_argument("--image_H", type=int, choices=[256, 320, 384, 400, 448, 512, 576, 640, 704, 768, 832, 960, 1024], default=512)
    parser.add_argument("--image_W", type=int, choices=[256, 320, 384, 400, 448, 512, 576, 640, 704, 768, 832, 960, 1024], default=512)
    parser.add_argument("--downsample_size", type=int, choices=[8, 16], default=16)

    # 采样参数
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=4)
    parser.add_argument("--cfg_interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top_p", type=float, default=1.0, help="top-p value to sample with")

    # 其他设置
    parser.add_argument("--condition", type=str, default='hed', choices=['canny', 'hed'])
    parser.add_argument("--per_proc_batch_size", type=int, default=16)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco', 'imagenet_code'], default='imagenet_code')
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--sample_dir", type=str, default="samples")
    parser.add_argument("--num_fid_samples", type=int, default=2000)
    parser.add_argument("--save_image", type=bool, default=True)

    # T5 相关参数
    parser.add_argument("--t5_path", type=str, required=True)
    parser.add_argument("--t5_model_type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5_feature_max_len", type=int, default=120)
    parser.add_argument("--t5_feature_dim", type=int, default=2048)

    # 代码路径
    parser.add_argument("--code_path", type=str, default="code")
    parser.add_argument("--code_path2", type=str, default=None)

    # 生成选项
    parser.add_argument("--get_image", type=bool, default=False)
    parser.add_argument("--get_prompt", type=bool, default=True)
    parser.add_argument("--get_label", type=bool, default=False)

    # 条件控制类型
    parser.add_argument("--condition_type", type=str, choices=['seg', 'canny', 'hed', 'lineart', 'depth', 'canny_base'], default="canny")

    # 适配器
    parser.add_argument("--adapter_size", type=str, default="small")

    # ORM
    parser.add_argument("--eval_num", type=int, default=1)
    parser.add_argument("--search_num", type=int, default=10)
    parser.add_argument("--reward_model", type=str, choices=['orm_zs', 'orm_ft', 'parm'], default='orm_zs')

    # 解析参数
    args = parser.parse_args()
    if args.reward_model == 'orm_zs':
        args.reward_model_path = 'lmms-lab/llava-onevision-qwen2-7b-ov'
        print("Running Zero-shot ORM...")
    elif args.reward_model == 'orm_ft':
        args.reward_model_path = 'ckpts/orm_ft'
        print("Running Fine-tuned ORM...")
    elif args.reward_model == 'parm':
        args.reward_model_path = 'ckpts/parm'
        print("Running PARM...")
    elif args.reward_model == '':
        args.reward_model_path = ''
        print("Running without Reward Model...")
    else:
        raise ValueError(f'Reward model: {args.reward_model} is not supported yet...')
    main(args)