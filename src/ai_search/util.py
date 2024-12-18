import os
import re
import sys
import json
from loguru import logger

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from termcolor import colored

current_dir = os.path.dirname(__file__)  
sys.path.append(current_dir)

from prompts import (
    THOUGHT_PROMPT_CN,
    THOUGHT_FEW_SHOT_1_CN,
    THOUGHT_FEW_SHOT_3_CN,
    SUMMARY_PROMPT_CN,
    SEARCH_TOOL_PROMPT_CN,
    SEARCH_TOOL_FEW_SHOT_1_CN, 
    SEARCH_TOOL_FEW_SHOT_3_CN,
    SELECT_TOOL_PROMPT_CN,
    SELECT_TOOL_FEW_SHOT_1_CN,
    SELECT_TOOL_FEW_SHOT_3_CN,
    SEARCHER_THOUGHT_PROMPT_CN, 
    SEARCHER_THOUGHT_FEW_SHOT_1_CN,
    SEARCHER_THOUGHT_FEW_SHOT_3_CN
)


@dataclass
class ResultSaves:
    """
    Class to store the result of an agent's operations.
    """
    response: str = ''
    inner_steps: List = field(default_factory=list)
    errmsg: Optional[str] = None
    thought_depth: int = 0
    search: List[str] = field(default_factory=list)
    search_nums: int = 0
    search_function: int = 0

    def add_search(self, new_search: list) -> None:
        """
        Add new search terms to the search history.
        Updates the search count and appends the new items.
        """
        self.search_nums += len(new_search)
        self.search.extend(new_search)

    @staticmethod
    def to_dict(obj: Union["ResultSaves", dict]) -> Union[dict, object]:
        """
        Convert a ResultSaves object or nested structure to a dictionary.
        Handles nested dict structures containing ResultSaves instances.
        """
        if isinstance(obj, ResultSaves):
            return {
                'response': obj.response,
                'inner_steps': obj.inner_steps,
                'errmsg': obj.errmsg,
                'thought_depth': obj.thought_depth,
                'search': obj.search,
                'search_nums': obj.search_nums,
                'search_function': obj.search_function
            }
        if isinstance(obj, dict):
            return {k: ResultSaves.to_dict(v) for k, v in obj.items()}
        return obj


class CustomLogger:
    def __init__(self, debug=False):
        """
        初始化日志配置
        :param debug: 是否启用调试模式
        """
        self.debug = debug

        # 配置 Loguru
        logger.remove()  # 移除默认处理器
        log_level = "DEBUG" if debug else "INFO"
        # 添加终端处理器
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{message}</cyan>",
        )

        # 可选：添加日志文件处理器
        # logger.add(
        #     "application.log",
        #     level=log_level,
        #     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        #     rotation="1 MB",  # 每 1 MB 生成一个新日志文件
        #     retention="7 days",  # 保留 7 天的日志文件
        # )

    def log(self, message, level="info"):
        """
        自定义日志方法，支持不同日志级别的颜色
        :param message: 日志消息
        :param level: 日志级别 (debug, info, warning, error, critical)
        """
        level_colors = {
            "debug": "blue",
            "info": "green",
            "warning": "yellow",
            "error": "red",
            "critical": "magenta"
        }
        color = level_colors.get(level, "white")
        colored_message = colored(message, color)

        # 使用 loguru 记录日志
        if level == "debug":
            logger.debug(colored_message)
        elif level == "info":
            logger.info(colored_message)
        elif level == "warning":
            logger.warning(colored_message)
        elif level == "error":
            logger.error(colored_message)
        elif level == "critical":
            logger.critical(colored_message)
     


class Prompt:
    @staticmethod
    def _add_thought(message: List[Dict], few_shot: bool = True) -> List[Dict]:
        """
        Add a system thought message with the current date and a predefined prompt.

        :param message: List of message dictionaries to format.
        :return: Formatted list of messages including the thought prompt.
        """
        formatted = []
        # Add current date as a system message
        timestamp = datetime.now().strftime('The current date is %Y-%m-%d.')
        formatted.append(dict(role='system', content=timestamp))
        # Add the thought prompt
        tought_prompt = THOUGHT_PROMPT_CN + '\n' + THOUGHT_FEW_SHOT_3_CN if few_shot else THOUGHT_PROMPT_CN
        formatted.append(dict(role='system', content=tought_prompt))
        # Append user-provided messages
        formatted += message
        return formatted

    @staticmethod
    def _get_web_search_prompt(message: List[Dict], tool_info_string: str, few_shot: bool = True) -> List[Dict]:
        """
        Add a web thought system prompt with the current date.

        :param message: List of message dictionaries to format.
        :param tool_info_string: tool_info_string to format.
        :return: Formatted list of messages including the searcher thought prompt.
        """
        formatted = []
        # Generate formatted date
        meta_time = "当前日期: %Y-%m-%d."
        meta_prompt = datetime.now().strftime(meta_time)

        # Format the searcher prompt with dynamic data
        search_prompt = SEARCH_TOOL_PROMPT_CN.format(
            current_date=meta_prompt,
            tool_info=tool_info_string
        )
        
        search_prompt = search_prompt + '\n' + SEARCH_TOOL_FEW_SHOT_3_CN if few_shot else search_prompt
        # Add the searcher system prompt
        formatted.append(dict(role='system', content=search_prompt))
        # Append user-provided messages
        formatted += message
        return formatted

    @staticmethod
    def _get_searcher_thought_prompt(message: List[Dict], few_shot: bool = True) -> List[Dict]:
        """
        Append a user searcher thought prompt.

        :param message: List of message dictionaries to format.
        :return: Formatted list of messages including the searcher thought prompt.
        """
        formatted = []
        # Append user-provided messages
        formatted += message
        search_thought_prompt = SEARCHER_THOUGHT_PROMPT_CN + '\n' + SEARCHER_THOUGHT_FEW_SHOT_3_CN if few_shot else SEARCHER_THOUGHT_PROMPT_CN
        # Add the searcher thought user prompt
        formatted.append(dict(role='user', content=search_thought_prompt))
        return formatted

    @staticmethod
    def _get_web_select_prompt(message: List[Dict], tool_info_string: str, few_shot: bool = True) -> List[Dict]:
        """
        Add a web select prompt and remove the second last message.

        :param message: List of message dictionaries to format.
        :param tool_info_string: tool_info_string to format.
        :return: Formatted list of messages including the searcher select prompt.
        """
        formatted = []
        # Format the searcher prompt with dynamic data
        select_prompt = SELECT_TOOL_PROMPT_CN.format(
            tool_info=tool_info_string
        )
        select_prompt = select_prompt + '\n' + SELECT_TOOL_FEW_SHOT_3_CN if few_shot else select_prompt
        # Add the searcher select system prompt
        formatted.append(dict(role='system', content=select_prompt))
        # Append user-provided messages
        formatted += message
        # Remove the second last message
        formatted.pop(-2)
        return formatted

    @staticmethod
    def _get_summary_prompt(message: List[Dict]) -> List[Dict]:
        """
        Add a summary prompt and format the assistant's content.

        :param message: List of message dictionaries to format.
        :return: Formatted list of messages including the summary prompt.
        """
        formatted = []
        # Add the summary system prompt
        formatted.append(dict(role='system', content=SUMMARY_PROMPT_CN))
        # Append user-provided messages
        formatted += message

        # # Process assistant messages to extract chain content
        # for item in formatted:
        #     if item['role'] == 'assistant':
        #         item['content'] = json.loads(item['content'])['chain']
        return formatted

class Operation_Utils:
    @staticmethod
    def Json_parser(response: str) -> dict:
        """
        从字符串中解析 JSON 数据。
        
        :param response: 包含 JSON 数据的字符串
        :return: 解析后的 JSON 数据（字典）；如果解析失败，返回空字典 {}
        """
        result = {}
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            matches = re.findall(r'\{.*\}', response)
            if matches:
                try:
                    result = json.loads(matches[0])
                    return result
                except json.JSONDecodeError:
                    pass
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    json_content = response[start:end].strip().replace("\n", "").replace("\r", "")
                    json_content = json_content.replace("“", "\"").replace("”", "\"")
                    result = json.loads(json_content)
                    return result
            except json.JSONDecodeError:
                print(colored(f"Error: 提取到的 JSON 无法解析\n{json_content}", 'red'))

        except Exception as e:
            print(colored(f"Error 未找到有效的 JSON 数据: {e}", 'red'))

        return result

def align_json_answers(test_set_path, eval_folder_path):
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)
    
    for file_name in os.listdir(eval_folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(eval_folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                eval_file_data = json.load(f)
            
            # 对齐数据
            for item in eval_file_data:
                for test in test_set:
                    if item['question'] == test['q']:
                        item['answer'] = test['a']  # 对齐答案
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(eval_file_data, f, ensure_ascii=False, indent=4)


if __name__=="__main__":
    
    # test_set_path = "/mnt/workspace/xieshichong/benchmark/data/qa_zh_data_v1.json"
    # eval_folder_path = "/mnt/workspace/wangbaoxin_wkspace/project/Agent/model_rlts"
    # align_json_answers(test_set_path,eval_folder_path)
    test = """{"name": "web_search", "parameters": {"query": ""2024年第三季度国内电影市场总票房" 的最新统计数据"}}                                                                        """
    print(Operation_Utils.Json_parser(test))
    
    

    