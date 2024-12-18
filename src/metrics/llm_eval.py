import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, ".."))
parent_of_project_root = os.path.abspath(os.path.join(project_root, ".."))
sys.path.append(project_root)

from ai_search import Operation_Utils
from serve import VllmServer

from datasets import Dataset, concatenate_datasets, load_dataset, Value
import pandas as pd
from io import StringIO

from typing import Dict, List, Optional, Union

import multiprocessing
from termcolor import colored


RELEVANCE_PROMPT = """请根据以下提供的答案，构建与其内容相符的 5 个问题。在生成问题之前，请先分析答案的主题和关键信息，并明确可能的提问方向，以确保问题与提供的答案高度相关。

### 输入信息
- **答案**：{answer}

### 任务步骤
1. **答案分析**：详细分析提供的答案，提取其中的核心主题和关键信息。
2. **问题推导**：基于关键信息，推导可能的问题方向，涵盖不同的侧重点。
3. **问题生成**：生成与提供答案内容匹配的 5 个问题，确保问题逻辑清晰，语言准确。

### 输出格式
请返回一个 JSON 对象，格式如下：

{{"thought": "详细描述你的分析过程，即问题生成的思路。","question": ["问题1", "问题2", "问题3", "问题4", "问题5"]}}

#### 字段说明
1. **thought**：清晰描述你的分析过程，包括划分子问题的思路。
2. **question**：根据提供的答案生成的问题，每个问题都应该具备完整的信息。

### 示例
- **答案**："地球是太阳系的第三颗行星，拥有适合生命生存的环境，包括大气层和液态水。"

输出结果：
{{
   "thought": "通过分析提供的信息，核心主题是地球的特性，关键信息包括地球是太阳系的第三颗行星、地球拥有适合生命生存的环境、地球的大气层和液态水。由此推导出可能的问题方向，例如地球在太阳系中的位置、地球为什么适合生命生存、地球环境的独特性。",
   "question": [
      "地球在太阳系中的位置是什么？",
      "地球有哪些特性使其适合生命生存？",
      "地球上的大气层对生命的意义是什么？",
      "液态水在地球上的分布及其重要性是什么？",
      "地球环境的独特性有哪些体现？"
   ]
}}
"""

CORRECTNESS_PROMPT = """请根据以下提供的信息，判断预测答案是否与标准答案的含义一致，且是否能够回答问题。根据一致性和回答问题的准确性，对预测答案打分，分数范围为 0-10 分。

### 输入信息
- **问题**：{question}
- **标准答案**：{standard_answer}
- **预测答案**：{predicted_answer}

### 评分规则
1. **完全一致**：
   - 如果预测答案与标准答案的含义完全一致，并准确回答问题，评分为 10 分。
2. **部分一致**：
   - 如果预测答案与标准答案的含义部分一致，或者能够回答问题但不完全准确，评分为 5-9 分，具体分数根据一致性程度确定。
3. **完全不一致**：
   - 如果预测答案与标准答案的含义完全不一致，且无法回答问题，评分为 0-4 分，具体分数根据不一致程度确定。

### 输出要求
请返回一个 JSON 对象，格式如下：
{{"score": "判断分数 (0-10)，以int的类型返回","reason": "简要说明判断结果的理由。"}}

### 示例
#### 示例 1
- **问题**："地球是太阳系的第几颗行星？"
- **标准答案**："地球是太阳系的第三颗行星。"
- **预测答案**："地球是太阳系的第四颗行星。"

输出结果：
{{"score": 3,"reason": "预测答案的内容与标准答案不一致，且回答问题错误。"}}

#### 示例 2
- **问题**："太阳系中有哪些适合生命生存的星球？"
- **标准答案**："地球是太阳系中唯一已知适合生命生存的星球。"
- **预测答案**："火星是太阳系中唯一适合生命生存的星球。"

输出结果：
{{"score": 2,"reason": "预测答案与标准答案语义冲突，且无法正确回答问题。"}}

#### 示例 3
- **问题**："地球上有哪些适合生命生存的条件？"
- **标准答案**："地球拥有液态水和适宜的大气层，这些条件适合生命生存。"
- **预测答案**："地球上有液态水，这对生命生存很重要。"

输出结果：
{{"score": 8,"reason": "预测答案部分回答了标准答案的内容，但未完整提及所有条件。"}}
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Run PlanningAgent with VllmServer")
    parser.add_argument('--num_processes', type=int, default=1, help="Server num processes (must be > 0)")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct', help="Model name to use for saving results")
    parser.add_argument('--api_key', type=str, default="EMPTY", help="API KEY of the VllmServer API")
    parser.add_argument('--api_base', nargs='+', type=str, default=["http://localhost:8000/v1"], help="Base URL of the VllmServer API")
    parser.add_argument('--embedding_path', required=True, type=str, help="Path to the embedding model")
    parser.add_argument('--eval_folder_path', required=True, type=str, help="Base path for eval data")
    parser.add_argument('--eval_name', required=True, type=str, help="File name for eval data")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()

def Semantic_Similarity(embedding_model,sentences_1: List[str],sentences_2: List[str], debug, **kwargs):
    assert embedding_model is not None
    embeddings_1 = embedding_model.encode(sentences_1, normalize_embeddings=True)
    embeddings_2 = embedding_model.encode(sentences_2, normalize_embeddings=True)
    similarity_scores = embeddings_1 @ embeddings_2.T
    return similarity_scores

def Semantic_Relevance(eval_model,embedding_model,question: str, prediction: str, debug, **kwargs):
    assert embedding_model is not None
    if prediction == "错误":
        score = 0.0
        return score
    else:
        messages = [{"role": "user", "content": RELEVANCE_PROMPT.format(answer=prediction)}]
        result = Operation_Utils.Json_parser(eval_model.chat(messages=messages))
        if debug:
            print(colored(f"Semantic Relevance：{result}", 'blue'))
        score = Semantic_Similarity(embedding_model,[question], result['question'], debug)
        return float(score.max())

def Factual_Correctness(eval_model,question: str, ground_truth: str, prediction: str, debug, **kwargs):
    if prediction == "错误":
        score = 0.0
        return score 
    else:
        messages = [{"role": "user", "content": CORRECTNESS_PROMPT.format(question=question, standard_answer=ground_truth,predicted_answer=prediction)}]
        result = eval_model.chat(messages=messages)
        result = Operation_Utils.Json_parser(eval_model.chat(messages=messages))
        if debug:
            print(colored(f"Factual Correctness：{result}", 'blue'))
        score = float(result['score'] / 10)
        return score

def split_dataset(dataset,num_process):
    chunk_size = len(dataset) // num_process
    chunks = [dataset.select(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_process - 1)]
    chunks.append(dataset.select(range((num_process - 1) * chunk_size, len(dataset))))
    return chunks

def scorer(llm, embedding, dataset, eval_funcs, debug):
    def compute_score(example):
        scores = {}
        for func in eval_funcs:
            question = example['question']
            ground_truth = example['answer']
            prediction = example['response']
            try:
                if func.__name__ == "Semantic_Similarity":
                    scores[func.__name__] = func(embedding, ground_truth, prediction, debug)
                elif func.__name__ == "Semantic_Relevance":
                    scores[func.__name__] = func(llm, embedding, question, prediction, debug)
                elif func.__name__ == "Factual_Correctness":
                    scores[func.__name__] = func(llm, question, ground_truth, prediction, debug)
                else:
                    # TODO warning have no this eval func
                    scores[func.__name__] = 0.0
            except Exception as e:
                print(colored(e,'red'))
                scores[func.__name__] = 0.0001
        return scores
        
    scored_dataset = dataset.map(compute_score,num_proc=1)
    return scored_dataset

def store_eval_rlts(eval_folder_path, eval_model_name, all_results):
    model_name = eval_model_name + '-llm-eval.json'
    store_data_dir = eval_folder_path + '/eval_rlts'
    store_file_path = os.path.join(store_data_dir, model_name) 
    df = all_results.to_pandas()
    df.to_json(store_file_path, orient="records", force_ascii=False, indent=4)

def save_scores_to_markdown(eval_folder_path, eval_model_name, eval_funcs, all_results):
    avg_scores = []
    headers = ["Model Name"] + [func.__name__ for func in eval_funcs]

    for func in eval_funcs:
        total_score = sum(num for resp, num in zip(all_results["response"], all_results[func.__name__]) if resp != "错误")
        total_acc_num =  sum(1 for resp in all_results["response"] if resp != "错误")
        avg_score = round(total_score / total_acc_num, 2)
        avg_scores.append(avg_score)
        print(f"{func.__name__} avg score: {avg_score}")

    required_fields = {"search_nums", "search_function"}
    if required_fields.issubset(all_results.column_names):
        headers = headers + ["agent_count","search_count","pass_rate"]
        avg_search_nums = round(sum(all_results["search_nums"]) / len(all_results), 2)
        avg_search_function = round(sum(all_results["search_function"]) / len(all_results), 2)
        pass_rate = round((len(all_results) - all_results['response'].count("错误")) / len(all_results), 2)
        avg_scores.append(avg_search_nums)
        avg_scores.append(avg_search_function)
        avg_scores.append(pass_rate)
        print(f"agent_count avg score: {avg_search_nums}")
        print(f"search_count avg score: {avg_search_function}")
        print(f"pass rate: {pass_rate}")

    data_row = [eval_model_name] + avg_scores
    mk_name = eval_folder_path.split("/")[-1] + "_llm_eval_benchmark.md"
    output_path = os.path.join(parent_of_project_root,mk_name)
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

def run_eval(api_key,api_base,model_name,embedding_path,dataset,eval_funcs, debug, progress_queue):
    llm = VllmServer(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
    )
    embedding = SentenceTransformer(embedding_path)
    with tqdm(total=len(dataset), desc=f"Progress on {api_base}", position=progress_queue.get()) as pbar:
        scored_dataset = scorer(llm, embedding, dataset, eval_funcs, debug)
    return scored_dataset

def main():
    args = parse_args()
    if args.num_processes <= 0:
        parser.error("--num_processes must be greater than 0")

    eval_funcs = [Semantic_Similarity,Semantic_Relevance,Factual_Correctness]
    
    eval_path = os.path.join(args.eval_folder_path,args.eval_name)
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

    splited_testset = split_dataset(dataset,args.num_processes)
    progress_queue = multiprocessing.Manager().Queue()
    for i in range(args.num_processes):
        progress_queue.put(i)

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        results = pool.starmap(
            run_eval,
            [
                (args.api_key,args.api_base[i],args.model_name,args.embedding_path,splited_testset[i],eval_funcs, args.debug, progress_queue)
                for i in range(args.num_processes)
            ]
        )
    for dset in results:
        for func in eval_funcs:
            dset = dset.cast_column(func.__name__, Value("float32"))
    all_results = concatenate_datasets(results)

    eval_model_name = args.eval_name.rpartition('.')[0]

    store_eval_rlts(args.eval_folder_path,eval_model_name,all_results)

    save_scores_to_markdown(args.eval_folder_path, eval_model_name, eval_funcs, all_results)

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    main()
    
    
