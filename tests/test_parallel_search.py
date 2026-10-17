"""Exercise the real MCP client and SearchAction against an in-memory HTTP server."""

import asyncio
import builtins
import json
import os
import unittest
import warnings
from unittest.mock import patch

import httpx

from src.actions import SearchAction
from src.plugins import BingSearch, ParallelSearch, YouSearch
from plugins import QihooWebSearch as DefaultQihooWebSearch


class MCPFixture:
    def __init__(self, result=None, mode='normal'):
        self.result = result if result is not None else {
            'content': [],
            'structuredContent': {'results': [
                {'url': 'https://youtube.com/skip', 'title': 'Blocked', 'excerpts': ['skip']},
                {'url': 'https://example.org/skip.pdf', 'excerpts': ['skip']},
                {'url': 'https://example.org/moon', 'title': '月球', 'excerpts': ['第一段', '第二段']},
                {'url': 'https://example.org/landing', 'title': None, 'excerpts': ['Landing date']},
                {'url': 'https://example.org/extra', 'title': 'Extra', 'excerpts': []},
            ]},
        }
        self.mode = mode
        self.requests = []
        self.clients = []
        self.started = None
        self.client_type = httpx.AsyncClient

    async def handle(self, request):
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request, body))
        method = body.get('method')
        if method == 'initialize':
            result = {
                'protocolVersion': body['params']['protocolVersion'],
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'fixture', 'version': '1'},
            }
        elif method == 'notifications/initialized':
            return httpx.Response(202)
        elif method == 'tools/list':
            result = {'tools': [{
                'name': 'web_search', 'inputSchema': {'type': 'object'},
            }]}
        elif method == 'tools/call':
            if self.started is not None:
                self.started.set()
            if self.mode == 'slow':
                await asyncio.sleep(10)
            if self.mode == 'http_error':
                return httpx.Response(503, text='Service unavailable')
            result = self.result
        else:
            raise AssertionError(f'Unexpected MCP request: {request.method} {body}')
        return httpx.Response(200, json={
            'jsonrpc': '2.0', 'id': body['id'], 'result': result,
        })

    def client(self, *args, **kwargs):
        client = self.client_type(*args, transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(client)
        return client

    def tool_calls(self):
        return [body for _, body in self.requests if body.get('method') == 'tools/call']


class ParallelSearchTests(unittest.TestCase):
    def test_search_action_consumes_anonymous_mcp_results(self):
        fixture = MCPFixture()
        with patch.dict(os.environ, {
            'PARALLEL_API_KEY': 'must-not-be-sent', 'BING_API': 'must-not-be-sent',
            'YDC_API_KEY': 'must-not-be-sent',
        }), patch('httpx.AsyncClient', side_effect=fixture.client):
            action = SearchAction(topk=2, searcher_class=ParallelSearch)
            results = action({'query': ['月球 样品 返回 时间', '月球 样品 着陆 地点']})

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['url'], 'https://example.org/moon')
        self.assertEqual(results[0]['title'], '月球')
        # SearchAction merges matching URLs from both queries.
        self.assertEqual(results[0]['summ'], '第一段\\n第二段\n第一段\\n第二段')
        self.assertEqual(results[1]['title'], 'https://example.org/landing')
        self.assertEqual(len(fixture.tool_calls()), 2)
        for call in fixture.tool_calls():
            self.assertEqual(call['params']['name'], 'web_search')
            arguments = call['params']['arguments']
            self.assertEqual(arguments['search_queries'], [arguments['objective']])
        for request, _ in fixture.requests:
            self.assertEqual(str(request.url), 'https://search.parallel.ai/mcp')
            self.assertEqual(request.headers['User-Agent'], ParallelSearch.user_agent)
            self.assertNotIn('authorization', request.headers)
            self.assertNotIn('x-api-key', request.headers)
            self.assertNotIn('ocp-apim-subscription-key', request.headers)
        self.assertTrue(all(client.is_closed for client in fixture.clients))

    def test_text_json_response_and_empty_results(self):
        for payload in ({'results': []}, {'results': [
            {'url': 'https://example.org', 'title': 'Example', 'excerpts': ['Useful']},
        ]}):
            with self.subTest(payload=payload):
                fixture = MCPFixture({'content': [{'type': 'text', 'text': json.dumps(payload)}]})
                with patch('httpx.AsyncClient', side_effect=fixture.client):
                    result = ParallelSearch().search('example source information')
                self.assertEqual(len(result), len(payload['results']))

    def test_tool_error_reaches_existing_action_warning_without_retry(self):
        fixture = MCPFixture({
            'isError': True, 'content': [{'type': 'text', 'text': 'Quota exceeded'}],
        })
        with patch('httpx.AsyncClient', side_effect=fixture.client), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = SearchAction(searcher_class=ParallelSearch)({'query': 'query'})
        self.assertEqual(result, {})
        self.assertTrue(any('Quota exceeded' in str(item.message) for item in caught))
        self.assertEqual(len(fixture.tool_calls()), 1)

    def test_malformed_response_and_http_failure_are_errors(self):
        cases = [
            MCPFixture({'content': [], 'structuredContent': {'unexpected': []}}),
            MCPFixture({'content': [{'type': 'text', 'text': 'not JSON'}]}),
            MCPFixture(mode='http_error'),
        ]
        for fixture in cases:
            with self.subTest(mode=fixture.mode), patch('httpx.AsyncClient', side_effect=fixture.client):
                with self.assertRaises(Exception):
                    ParallelSearch().search('query')
            self.assertEqual(len(fixture.tool_calls()), 1)
            self.assertTrue(all(client.is_closed for client in fixture.clients))

    def test_timeout_closes_client(self):
        fixture = MCPFixture(mode='slow')
        with patch('httpx.AsyncClient', side_effect=fixture.client):
            with self.assertRaises(TimeoutError):
                ParallelSearch(timeout=0.1).search('query')
        self.assertEqual(len(fixture.tool_calls()), 1)
        self.assertTrue(all(client.is_closed for client in fixture.clients))

    def test_cancellation_propagates_and_closes_client(self):
        fixture = MCPFixture(mode='slow')

        async def cancel():
            fixture.started = asyncio.Event()
            task = asyncio.create_task(ParallelSearch()._search('query'))
            await asyncio.wait_for(fixture.started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with patch('httpx.AsyncClient', side_effect=fixture.client):
            asyncio.run(cancel())
        self.assertTrue(all(client.is_closed for client in fixture.clients))

    def test_optional_dependency_and_existing_default(self):
        original_import = builtins.__import__

        def without_mcp(name, *args, **kwargs):
            if name == 'mcp' or name.startswith('mcp.'):
                raise ImportError('MCP intentionally unavailable')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=without_mcp):
            self.assertIsInstance(SearchAction().searcher, DefaultQihooWebSearch)
            self.assertIsInstance(SearchAction(searcher_class=BingSearch).searcher, BingSearch)
            self.assertIsInstance(SearchAction(searcher_class=YouSearch).searcher, YouSearch)
            with self.assertRaisesRegex(ImportError, 'requirements-parallel.txt'):
                ParallelSearch().search('query')

    def test_invalid_query_and_limits_fail_before_network(self):
        for kwargs in ({'topk': 0}, {'timeout': 0}):
            with self.assertRaises(ValueError):
                ParallelSearch(**kwargs)
        for query in ('', '  ', None):
            with self.assertRaises(ValueError):
                ParallelSearch().search(query)


if __name__ == '__main__':
    unittest.main()
