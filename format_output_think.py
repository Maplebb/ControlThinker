import json
import re
import argparse

data = []
parser = argparse.ArgumentParser(description="Process and format JSON data.")
parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file.")
parser.add_argument('--output_json', type=str, required=True, help="Path to the output JSON file.")
args = parser.parse_args()

input_json = args.input_json
output_json = args.output_json

# Parse the input JSON file
with open(input_json, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  
            # print(line)
            data.append(json.loads(line))

output_dict = {}

for entry in data:
    user_content = entry['messages'][1]['content']
    original_prompt_match = re.search(r'Original Prompt: (.*?)\n\nPlease analyze and rewrite', user_content, re.DOTALL)
    original_prompt = original_prompt_match.group(1).strip() if original_prompt_match else ""

    image_path = entry['images'][0]['path']
    image_name = image_path.split('/')[-1]

    response = entry['response']
    response_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    response = response_match.group(1).strip().strip('"') if response_match else ""

    if image_name not in output_dict:
        output_dict[image_name] = {
            "original_prompt": original_prompt,
            "prompts": [response],
            "prompt_count": 1
        }
    else:
        output_dict[image_name]["prompts"].append(response)
        output_dict[image_name]["prompt_count"] += 1

sorted_output_dict = dict(sorted(output_dict.items(), key=lambda item: int(item[0].split('.')[0])))

# save the output
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(sorted_output_dict, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_json}")
