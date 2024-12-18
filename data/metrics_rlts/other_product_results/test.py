import json
import re

def extract_data_from_txt(txt_file_path):
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    matches = re.findall(r'(?:^|=====)(.*?)(?======|$)', content, re.DOTALL)
    matches = [match.strip() for match in matches if match.strip()]
    import pdb;pdb.set_trace()
    if len(matches) != len(other_product_datas):
        raise ValueError("The number of extracted matches does not match the number of items in the JSON file.")
    return matches

with open("gpt_4o_sample100.json", 'r', encoding='utf-8') as f:
    other_product_datas = json.load(f)

with open("/mnt/workspace/wangbaoxin_wkspace/project/Agent/model_rlts/three_few_shot/Qwen2.5-7B-Instruct-128k-bing.json", 'r', encoding='utf-8') as f:
    our_product_datas = json.load(f)

new_data = []
for other in other_product_datas:
    for our in our_product_datas:
        if our['question'] == other['question']:
            new_data.append(our)

with open("our_product_sample100_7B.json", 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)