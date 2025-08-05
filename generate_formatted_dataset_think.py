import json
import os
import argparse
from tqdm import tqdm

def build_sample(image_dir, image_name, prompt_text, condition_type):
    img_path = os.path.join(image_dir, image_name)  # Path of Images
    user_txt = (
        f"<image>\nOriginal Prompt: {prompt_text}\n\n"
        f"Please analyze and rewrite the above image-generation prompt based on the provided {condition_type}-based control image and the original prompt. "
    )

    system_message = {
        "role": "system",
        "content": (
            f"You are a professional prompt engineer tasked with rewriting image-generation prompts based on a provided \"{condition_type}-based control image\" and an original prompt.\n"
            f"The user will provide an original prompt and a {condition_type}-based control image which is for the conditional image generation. Your task is to analyze the {condition_type}-based control image and the original prompt, then expand it into a new prompt that can closely describe the image to generate.\n"
            "First, output your reasoning process within <think></think> tags. Then provide the expanded prompt within <answer></answer> tags."
        )
    }

    return {
        "messages": [
            system_message,
            {"role": "user",      "content": user_txt},
            # {"role": "assistant", "content": ""}  
        ],
        "images": [img_path]
    }

def main(input_json, image_dir, output_jsonl, condition_type, repeat):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out = []
    for idx, (image_name, prompt_text) in enumerate(tqdm(data.items(), desc="Processing samples")):
        # if idx >= 100:  # Only process the first 100 entries
        #     break
        sample = build_sample(image_dir, image_name, prompt_text, condition_type)
        for i in range(repeat):
            out.append(sample)

    with open(output_jsonl, "w", encoding='utf-8') as f:
        for s in out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Conversion complete! Total samples written: {len(out)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert single JSON to ms-swift standard format")
    parser.add_argument("--condition_type", required=True, choices=["depth", "hed", "segmentation", "lineart", "canny"], help="Type of condition for the image.")
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--repeat", default=1, type=int, help="Number of times to repeat each sample")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_jsonl", required=True, help="Output file path for standard ms-swift format")

    args = parser.parse_args()

    main(args.input_json, args.image_dir, args.output_jsonl, args.condition_type, args.repeat)
