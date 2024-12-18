import os
import sys
import json
import argparse
from tqdm import tqdm

import multiprocessing

from io import StringIO
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from util import ResultSaves
from serve import VllmServer
from plugins import QihooWebSearch, BingSearch
from actions import ActionExecutor, SearchAction, SelectAction
from component import PlanningAgent, SearcherAgent, SearchDistributor

def parse_args():
    parser = argparse.ArgumentParser(description="Run PlanningAgent with VllmServer")
    parser.add_argument('--num_processes', type=int, default=1, help="Server num processes (must be > 0)")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct', help="Model name to use for saving results")
    parser.add_argument('--api_key', type=str, default="EMPTY", help="API KEY of the VllmServer API")
    parser.add_argument('--api_base', nargs='+', type=str, default=["http://localhost:8000/v1"], help="Base URL of the VllmServer API")
    parser.add_argument('--input_path', required=True, type=str, help="Path to the input data file")
    parser.add_argument('--save_path', required=True, type=str, help="Base path for saving results")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()


def split_dataset(dataset, num_process):
    chunk_size = len(dataset) // num_process
    chunks = [dataset.select(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_process - 1)]
    chunks.append(dataset.select(range((num_process - 1) * chunk_size, len(dataset))))
    return chunks

def store_agent_result(agent_result, dataset, save_path):
    """
    Process a single agent_result, compare with dataset, and save the result to a JSONL file.
    
    Parameters:
        agent_result (dict): The result from an agent, to be processed.
        dataset (Dataset): The dataset to compare with.
        save_path (str): The file path to save the processed result.
    """
    agent_result = ResultSaves.to_dict(agent_result)
    inner_steps = agent_result.get('inner_steps', [])
    query = next((step.get('content', None) for step in inner_steps if step.get('role') == 'user'), None)
    
    if not query:
        print("No query found in agent_result.")
        return

    matching_test = next((raw for raw in dataset if raw['question'] == query), None)

    if not matching_test:
        print(f"No matching test found for query: {query}")
        return

    processed_data = {
        "question": query,
        "answer": matching_test.get('answer', None),
        "response": agent_result.get('response', None),
        "search": agent_result.get('search', []),
        "thought_depth": agent_result.get('thought_depth', 0),
        "search_nums": agent_result.get('search_nums', 0),
        "search_function": agent_result.get('search_function', None)
    }

    with open(save_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(processed_data, ensure_ascii=False) + '\n')

    

def run_agent_instance(model_name, api_key, api_base, dataset, save_path, debug, progress_queue):
    llm = VllmServer(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
    )
    tool_info, tool_map = ActionExecutor.get_tool_info(SearchAction, SelectAction)

    agent = PlanningAgent(
        llm,
        SearchDistributor(
            searcher_type=SearcherAgent,
            llm=llm,
            # searcher_class=BingSearch,
            # searcher_class=QihooWebSearch(api_key=os.getenv("BING_API")),
            tool_info=tool_info,
            tool_map=tool_map,
        ),
        max_turn=4,
        debug=debug
    )

    try:
        with tqdm(total=len(dataset), desc=f"Progress on {api_base}", position=progress_queue.get()) as pbar:
            for query in dataset['question']:
                agent_result = None
                try:
                    for agent_result in agent.stream_chat(query):
                        pass
                    store_agent_result(agent_result, dataset, save_path)
                except Exception as e:
                    print(f"Error during query processing: {e}")                
                pbar.update(1)
        print(f"Saved processed result to {save_path}")
    except Exception as e:
        print(f"Processing failed with exception: {e}")

def main():
    args = parse_args()
    if args.num_processes <= 0:
        parser.error("--num_processes must be greater than 0")

    if args.input_path.endswith(('.json', '.jsonl')):
        dataset = load_dataset('json', data_files=args.input_path, split='train')
    else:
        dataset = load_dataset(args.input_path, split='train')    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    # Checkpoint
    if os.path.exists(args.save_path) and os.path.getsize(args.save_path) > 0:
        checkpoint_dataset = load_dataset('json', data_files=args.save_path, split='train')
        checkpoint_questions = set(checkpoint_dataset['question'])
        dataset = dataset.filter(lambda x: x['question'] not in checkpoint_questions)
    
    if len(dataset) == 0:  # 或者 if not dataset:
        print("No new data to process.")
    else:
        # Split the dataset among workers
        if args.num_processes > 1:
            splited_dataset = split_dataset(dataset, args.num_processes)
        else:
            splited_dataset = [dataset]

        progress_queue = multiprocessing.Manager().Queue()
        for i in range(args.num_processes):
            progress_queue.put(i)

        with multiprocessing.Pool(processes=args.num_processes) as pool:
            pool.starmap(
                run_agent_instance,
                [
                    (args.model_name, args.api_key, args.api_base[i], splited_dataset[i], args.save_path, args.debug, progress_queue)
                    for i in range(args.num_processes)
                ],
            )
    

if __name__ == '__main__':
    main()
    