import asyncio
import json
import logging
import random
import re
import os
import time
import warnings

from typing import List, Optional, Tuple, Type, Union

import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached


import hashlib



class BaseSearch:

    def __init__(self, topk: int = 3, black_list: List[str] = None):
        self.topk = topk
        self.black_list = black_list

    def _filter_results(self, results: List[tuple]) -> dict:
        filtered_results = {}
        count = 0
        for url, snippet, title in results:
            if all(domain not in url
                   for domain in self.black_list) and not url.endswith('.pdf'):
                filtered_results[count] = {
                    'url': url,
                    'summ': json.dumps(snippet, ensure_ascii=False)[1:-1],
                    'title': title
                }
                count += 1
                if count >= self.topk:
                    break
        return filtered_results


class QihooWebSearch(BaseSearch):
    def __init__(self, 
                 topk: int=5,
                 api_key: str='',
                 key: str= "597ff7ec4a69ec",
                 url:str="https://api.360.cn/v2/mwebsearch",
                 cid: str="saas_360zhengqi",
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 **kwargs):
        # self.api_key = api_key
        self.api_key = os.getenv("BING_API")
        self.key = key
        self.url = url
        self.cid = cid
        super().__init__(topk, black_list)

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self.call_qihoo_web(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from DuckDuckGo after retries.')
    def call_qihoo_web(self, query: str, **kwargs) -> dict:
        unix_timestamp = int(time.time())
        params = {
            'q': query,
            't': unix_timestamp,
            'ref_prom': '360so-s4',
            'cid': self.cid,
            'm': self.calculate_md5_string(f'{self.cid}{query}{self.key}{unix_timestamp}')
        }
        # 构建请求头部
        headers = {
            'Authorization': f'Bearer {self.api_key}',  # 根据 API 平台的要求，可能是其他形式，例如 Basic Auth
            'Content-Type': 'application/json'  # 根据 API 的要求设置其他头部信息
        }
        response = requests.get(self.url, params=params, headers=headers)
        return response.json().get('items')

    def calculate_md5_string(self, input_string):
        # print(f"Input string: {input_string}")
        # print(f"Encoded bytes: {input_string.encode('utf-8')}")
        # 创建一个 MD5 hash 对象
        md5_hash = hashlib.md5()

        # 更新 hash 对象的内容，需要将字符串编码为字节型数据
        md5_hash.update(input_string.encode('utf-8'))

        # 获取 MD5 哈希值的十六进制表示
        md5_hex = md5_hash.hexdigest()

        return md5_hex[:16]

    def _parse_response(self, document):
        final_results = []
        for page in document:
            page_content = None
            if page["type"] == "weather":
                if "realtime" in page["data"]:
                    page_content = "实时天气: " + json.dumps(page["data"]["realtime"], ensure_ascii=False)
                if "weather" in page["data"]:
                    week_weather = "最近两周天气: " + json.dumps(page["data"]["weather"], ensure_ascii=False)
                    if page_content == None:
                        page_content = week_weather
                    else:
                        page_content += "\n" + week_weather
            elif page["type"] == "regional" or page["type"] == "engine":
                if "content_large" in page and page["content_large"] != "":
                    page_content = page["content_large"]
                elif "summary" in page and page["summary"] != "":
                    page_content = page["summary"]
            if page_content is None:
                continue
            url = page['url'] if "url" in page else ''
            title = page['title'] if "title" in page else ''
            final_results.append((url, page_content, title))
        return self._filter_results(final_results)

class BingSearch(BaseSearch):

    def __init__(self,
                 api_key: str = "4ebc523f056e49719f241404bd53448c",
                 region: str = 'zh-CN',
                 topk: int = 3,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 **kwargs):
        self.api_key = api_key
        self.market = region
        self.proxy = kwargs.get('proxy')
        super().__init__(topk, black_list)

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self._call_bing_api(query)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(
                    f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception(
            'Failed to get search results from Bing Search after retries.')

    def _call_bing_api(self, query: str) -> dict:
        endpoint = 'https://api.bing.microsoft.com/v7.0/search'
        params = {'q': query, 'mkt': self.market, 'count': f'{self.topk * 2}'}
        headers = {'Ocp-Apim-Subscription-Key': self.api_key}
        response = requests.get(
            endpoint, headers=headers, params=params, proxies=self.proxy)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response: dict) -> dict:
        webpages = {
            w['id']: w
            for w in response.get('webPages', {}).get('value', [])
        }
        raw_results = []

        for item in response.get('rankingResponse',
                                 {}).get('mainline', {}).get('items', []):
            if item['answerType'] == 'WebPages':
                webpage = webpages.get(item['value']['id'])
                if webpage:
                    raw_results.append(
                        (webpage['url'], webpage['snippet'], webpage['name']))
            elif item['answerType'] == 'News' and item['value'][
                    'id'] == response.get('news', {}).get('id'):
                for news in response.get('news', {}).get('value', []):
                    raw_results.append(
                        (news['url'], news['description'], news['name']))

        return self._filter_results(raw_results)

    