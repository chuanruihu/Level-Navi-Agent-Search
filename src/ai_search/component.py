import re
import json
import traceback

from copy import deepcopy
from datetime import datetime
from termcolor import colored
from typing import Dict, List, Optional, Generator, Type
from concurrent.futures import ThreadPoolExecutor, as_completed

from serve import VllmServer
from util import ResultSaves, CustomLogger, Prompt, Operation_Utils
from actions import ActionExecutor, SearchAction, SelectAction

import multiprocessing
from termcolor import colored



class SearcherAgent:
    def __init__(
        self, 
        llm: VllmServer = None, 
        max_turn: int = 3,
        topk: int = 6,
        searcher_class: Type = None,
        tool_info: List[Dict] = [],
        tool_map: Dict = {},
        debug: bool = False,
        **kwargs
    ) -> None:
        self.llm = llm
        self.max_turn = max_turn
        self.topk = topk
        self.searcher_class=searcher_class
        self.tool_map = tool_map
        self.tool_info = tool_info
        self.debug = debug
        self.logger = CustomLogger(debug=self.debug)

    def _execute_tool_call(self, func_calls: Dict) -> str:
        """
        Simulate the execution of tool calls.
        """
        available_tools = self.tool_map
        call_function = available_tools[func_calls['name']]
        parameters = func_calls['parameters']
        func = call_function(self.topk,self.searcher_class)
        result = func(parameters)
        result_str = json.dumps(result,ensure_ascii=False)
        return result_str

    def _information_sufficient(self, inner_history: List) -> bool:
        thought = Prompt._get_searcher_thought_prompt(inner_history, few_shot=True)        
        response = self._stream_chat(thought,'light_green')
        try:
            is_sufficient = Operation_Utils.Json_parser(response)
            return "False" in str(is_sufficient.get('action'))
        except json.JSONDecodeError:
            return False

    def _stream_chat(self, message: list, color: str) -> str:
        """直接从模型获取响应。"""
        response = ''.join(
            chunk.choices[0].delta.content
            for chunk in self.llm.stream_chat(message)
            if chunk.choices[0].delta.content
        )
        self.logger.log(f"Response: {response}", "debug")
        return response

    def get_response(self, query: str, agent_result: ResultSaves) -> str:
        query = f"## 当前问题\n{query}"
        message = [{'role': 'user', 'content': query}]
        inner_history = message[:]
        max_turn = 3
        is_sufficient = self._information_sufficient(inner_history)
        func_params = {"name": None}
        for _ in range(max_turn):
            if is_sufficient or _ == max_turn-1:
                return self._stream_chat(inner_history,'red')
            else:
                if not func_params['name'] or func_params['name'] == 'web_select':
                    self.logger.log("==========调用 web search 工具==========", "debug")
                    search_tool_info_string = json.dumps(self.tool_info[0], ensure_ascii=False)
                    search_tool_prompt = Prompt._get_web_search_prompt(inner_history,search_tool_info_string,few_shot=True)
                    response = self._stream_chat(search_tool_prompt,'red')
                    func_params = Operation_Utils.Json_parser(response)
                    inner_history.append({"role": "assistant", "content": response})
                    if func_params.get('parameters', {}) == {} or func_params.get('parameters', {}).get('query', []) == []:
                        break
                    search_observation = self._execute_tool_call(func_params)
                    self.logger.log(f"web search 返回结果\n{search_observation}", "debug")
                    inner_history.append({"role": "user", "content": search_observation})
                    agent_result.search_function += 1 
                elif func_params['name'] == 'web_search':
                    self.logger.log("==========调用 web select 工具==========", "debug")
                    select_tool_info_string = json.dumps(self.tool_info[1], ensure_ascii=False)
                    select_prompt = Prompt._get_web_select_prompt(inner_history,select_tool_info_string,few_shot=True)
                    response = self._stream_chat(select_prompt,'red')
                    func_params = Operation_Utils.Json_parser(response)
                    inner_history.append({"role": "assistant", "content": response})    
                    func_params['parameters']['search_results'] = json.loads(search_observation)
                    select_observation = self._execute_tool_call(func_params)
                    self.logger.log(f"web select 返回结果\n{select_observation}", "debug")
                    inner_history.append({"role": "user", "content": select_observation})
                else:
                    # TODO feedback
                    self.logger.log("Unknown function call or feedback required", "error")
                    break
            is_sufficient = self._information_sufficient(inner_history)

class SearchDistributor:
    def __init__(
        self, 
        searcher_type: Type,
        llm: VllmServer = None,
        max_turn: int = 3,
        topk: int = 6,
        searcher_class: Type = None,
        tool_info: List[Dict] = [],
        tool_map: Dict = {}
    ):
        self.llm = llm
        self.topk = topk
        self.searcher_class = searcher_class
        self.max_turn = max_turn
        self.searcher_type = searcher_type
        self.tool_info = tool_info
        self.tool_map = tool_map

    def distribute_searches(self, queries, agent_saves, debug):

        self.searchers = [self.searcher_type(self.llm, self.max_turn, self.topk, self.searcher_class, self.tool_info, self.tool_map, debug) for _ in range(len(queries))]
        # 创建一个字典来保存Future对象和对应的query
        future_to_query = {}

        with ThreadPoolExecutor(max_workers=len(self.searchers)) as executor:
            for query, searcher in zip(queries, self.searchers):
                future = executor.submit(searcher.get_response, query, agent_saves)
                future_to_query[future] = query
            content = ''
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = '##当前问题:' + query + '\n' + future.result() + '\n'
                    content += result
                    #print(f"Query: {query} => {result}")
                except Exception as exc:
                    print(f"Query: {query} generated an exception: {exc}")

            return content

class PlanningAgent:
    def __init__(
        self,
        llm: VllmServer = None,
        searcher: SearchDistributor =None,
        max_turn: int = 3,
        debug: bool = False,
    ) -> None:
        self.llm = llm
        self.max_turn = max_turn
        self.searchagent = searcher
        self.debug = debug
        self.logger = CustomLogger(debug=self.debug)
    

    def stream_chat(self, messages: List[Dict]) -> Generator:
        if isinstance(messages, str):
            messages = [{'role': 'user', 'content': messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        self.logger.log(f"Messages received: {messages}", "debug")
        inner_history = messages[:]
        # code_history = get_code_prompt([])
        agent_saves = ResultSaves()
        for turn in range(self.max_turn):
            self.logger.log(f"----------第{turn}轮思考----------","debug")
            thought_prompt = Prompt._add_thought(inner_history,few_shot=True)
            response = ''.join(
                chunk.choices[0].delta.content
                for chunk in self.llm.stream_chat(thought_prompt)
                if chunk.choices[0].delta.content
            )
            self.logger.log(f"Response: {response}", "debug")
            check = Operation_Utils.Json_parser(response)
            if check == {}:
                agent_saves.response = response
                if turn == self.max_turn - 1:
                    if not response:
                        response = "错误"
                    inner_history.append({"role": "assistant", "content": response})
                    agent_saves.inner_steps = inner_history
                    yield deepcopy(agent_saves)
                    return
            elif check['search'] == [] or (turn == self.max_turn - 1):
                summary = Prompt._get_summary_prompt(inner_history)
                response = ''.join(
                    chunk.choices[0].delta.content
                    for chunk in self.llm.stream_chat(summary)
                    if chunk.choices[0].delta.content
                )
                self.logger.log(f"==========总结答案==========\n{response}", "debug")
                agent_saves.response = response
                yield deepcopy(agent_saves)
                return
            else:
                agent_saves.thought_depth += 1
                inner_history.append({"role": "assistant", "content": response})
                agent_saves.inner_steps = inner_history
                agent_saves.add_search(check['search'])
                result = self.searchagent.distribute_searches(check['search'],agent_saves,self.debug)
                if result:
                    inner_history.append({"role": "user", "content": result})
        yield deepcopy(agent_saves)
