import os
import sys
import requests
import warnings
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached
from typing import List, Optional, Tuple, Type, Union

from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import colored

import re
import json5
import urllib.parse

import itertools

current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from plugins import QihooWebSearch, BingSearch

class ContentFetcher:

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def fetch(self, url: str) -> Tuple[bool, str]:
        try:
            response = requests.get(
                url, 
                timeout=self.timeout,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            html = response.content
        except requests.RequestException as e:
            return False, str(e)

        text = BeautifulSoup(html, 'html.parser').get_text()
        cleaned_text = re.sub(r'\n+', '\n', text)
        return True, cleaned_text


class SearchAction:
    name = 'web_search'
    description = '调用浏览器 API，根据用户提出的问题搜索相关信息并返回搜索结果。'
    parameters = [{
        'name': 'query',
        'type': 'str',
        'description': '用户提出的问题或需要搜索的关键词。',
        'required': True
    }]
    def __init__(
        self,
        topk: str = 3,
        searcher_class: Type = None,
    ) -> None:
        self.topk = topk
        black_list = ['enoN','youtube.com','bilibili.com','researchgate.net']
        if searcher_class is None:
            searcher_class = QihooWebSearch
        self.searcher = searcher_class(black_list=black_list, topk=self.topk)
    def call(self, arguments: dict) -> dict:
        if isinstance(arguments['query'], list):
            queries = arguments['query'] 
        if isinstance(arguments['query'], str):
            queries = [arguments['query']]
        search_results = {}

        with ThreadPoolExecutor() as executor:
            future_to_query = {
                executor.submit(self.searcher.search, q): q
                for q in queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                except Exception as exc:
                    warnings.warn(f'{query} generated an exception: {exc}')
                else:
                    for result in results.values():
                        if result['url'] not in search_results:
                            search_results[result['url']] = result
                        else:
                            search_results[
                                result['url']]['summ'] += f"\n{result['summ']}"

        observation = {
            idx: result
            for idx, result in itertools.islice(enumerate(search_results.values()), 6) #前6个
        }
        return observation
    def __call__(self, arguments: dict) -> dict:
        return self.call(arguments)



class SelectAction:
    name = 'web_select'
    description = '从搜索结果中选择相关网页，便于进一步阅读或处理。'
    parameters = [{
        'name': 'select_ids',
        'type': 'list[str]',
        'description': '选择相关网页索引列表，用于指定需要阅读的网页。',
        'required': True
    }]
    def __init__(
        self,
        topk: str = 3,
        searcher_class: Type = None,
        **kwargs
    ) -> None:
        timeout = 5
        self.fetcher = ContentFetcher(timeout=timeout)
    def call(self, arguments: dict) -> dict:
        select_ids = arguments['select_ids']
        search_results = arguments['search_results']
        if not search_results:
            raise ValueError('No search results to select from.')

        new_search_results = {}
        with ThreadPoolExecutor() as executor:
            future_to_id = {
                executor.submit(self.fetcher.fetch,
                                search_results[select_id]['url']):
                select_id
                for select_id in select_ids if select_id in search_results
            }

            for future in as_completed(future_to_id):
                select_id = future_to_id[future]
                try:
                    web_success, web_content = future.result()
                except Exception as exc:
                    warnings.warn(f'{select_id} generated an exception: {exc}')
                else:
                    if web_success:
                        search_results[select_id][
                            'content'] = web_content[:2048]
                        new_search_results[select_id] = search_results[
                            select_id].copy()
                        new_search_results[select_id].pop('summ')

        return new_search_results
    def __call__(self, arguments: dict) -> dict:
        return self.call(arguments)


if __name__ == "__main__":
    arguments =  {"query": "2024 TGA 年度最佳游戏 提名作品"}
    func = SearchAction(topk=3,searcher_class=BingSearch)
    result = func(arguments)
    print(result)