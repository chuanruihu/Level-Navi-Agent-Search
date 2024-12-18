import os
import re
import sys
import json
import string

import jieba
import difflib
import argparse

from tqdm import tqdm
from typing import List
from rouge import Rouge
from fuzzywuzzy import fuzz
from collections import Counter

current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, ".."))
parent_of_project_root = os.path.abspath(os.path.join(project_root, ".."))
sys.path.append(project_root)

from datasets import Dataset,load_dataset,Value
import pandas as pd
from io import StringIO


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

def recall_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    recall = 1.0 * num_same / len(ground_truth)
    return recall

def qa_recall_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return recall_score(prediction_tokens, ground_truth_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PlanningAgent with VllmServer")
    parser.add_argument('--eval_folder_path', required=True, type=str, help="Base path for eval data")
    parser.add_argument('--eval_name', required=True, type=str, help="File name for eval data")
    return parser.parse_args()

def scorer(dataset,eval_funcs):
    def compute_score(example):
        scores = {}
        prediction = example['response']
        ground_truth = example['answer']
        for func in eval_funcs:
            scores[func.__name__] = func(prediction, ground_truth)
        return scores
        
    scored_datas = dataset.map(compute_score,num_proc=4)
    return scored_datas

def save_scores_to_markdown(file_name: str, eval_funcs: List, scored_datas: Dataset, output_path: str):
    avg_scores = []

    headers = ["Model Name"] + [func.__name__ for func in eval_funcs]

    for func in eval_funcs:
        total_score = sum(num for resp, num in zip(scored_datas["response"], scored_datas[func.__name__]) if resp != "错误")
        total_acc_num =  sum(1 for resp in scored_datas["response"] if resp != "错误")
        avg_score = round(total_score / total_acc_num, 2)
        print(f"{func.__name__} avg score: {avg_score}")
        avg_scores.append(avg_score)

    required_fields = {"search_nums", "search_function"}
    if required_fields.issubset(scored_datas.column_names):
        headers = headers + ["agent_count","search_count","pass_rate"]
        avg_search_nums = round(sum(scored_datas["search_nums"]) / len(scored_datas), 2)
        avg_search_function = round(sum(scored_datas["search_function"]) / len(scored_datas), 2)
        pass_rate = round((len(scored_datas) - scored_datas['response'].count("错误")) / len(scored_datas), 2)
        avg_scores.append(avg_search_nums)
        avg_scores.append(avg_search_function)
        avg_scores.append(pass_rate)
        print(f"agent_count avg score: {avg_search_nums}")
        print(f"search_count avg score: {avg_search_function}")
        print(f"pass rate: {pass_rate}")

    data_row = [file_name] + avg_scores

    try:
        with open(output_path, "r") as file:
            lines = file.readlines()
            if not lines or not lines[0].strip().startswith("| Model Name"):
                raise FileNotFoundError  
    except FileNotFoundError:
        with open(output_path, "w") as file:
            file.write("| " + " | ".join(headers) + " |\n")
            file.write("|" + " | ".join(["---"] * len(headers)) + "|\n")

    with open(output_path, "a") as file:
        file.write("| " + " | ".join(map(str, data_row)) + " |\n")

    print(f"Results saved to {output_path}")

def main():
    args = parse_args()
    eval_funcs = [rouge_zh_score,qa_f1_zh_score,qa_recall_zh_score]

    eval_path = os.path.join(args.eval_folder_path, args.eval_name)
    
    if args.eval_name.endswith(('.json', '.jsonl')):
        try:
            dataset = load_dataset('json', data_files=eval_path, split='train')
        except Exception as e:
            with open(eval_path,'r') as f:
                datas = json.load(f)
            datas = [{k: v for k, v in data.items() if k != 'search'} for data in datas]
            dataset = Dataset.from_list(datas)
    else:
        dataset = load_dataset(eval_path, split='train')
    
    scored_datas = scorer(dataset, eval_funcs)
    model_name = args.eval_name.rpartition('.')[0]
    mk_name = args.eval_folder_path.split("/")[-1] + "_llm_eval_benchmark.md"
    output_path = os.path.join(parent_of_project_root,mk_name)
    save_scores_to_markdown(model_name,eval_funcs,scored_datas,output_path)
    
    store_data_dir = args.eval_folder_path + '/eval_rlts'
    store_name = model_name + "-token-eval.json"
    store_file_path = os.path.join(store_data_dir, store_name)
    df = scored_datas.to_pandas()
    df.to_json(store_file_path, orient="records", force_ascii=False, indent=4)

    
if __name__ == '__main__':
    main()
    
        

    