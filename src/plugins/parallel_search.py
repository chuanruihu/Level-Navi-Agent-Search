"""Optional anonymous Parallel Search MCP provider."""

import asyncio
import json
from datetime import timedelta

from .web_search import BaseSearch


class ParallelSearch(BaseSearch):
    """Adapt Parallel's web_search results to the existing SearchAction format.

    The optional MCP dependencies are loaded only when a search is requested.
    This provider makes one attempt per query and never reads API credentials.
    """

    endpoint = 'https://search.parallel.ai/mcp'
    user_agent = 'Level-Navi-Agent-Search/ParallelSearch'

    def __init__(self, topk=3, black_list=None, timeout=30, **kwargs):
        if topk <= 0 or timeout <= 0:
            raise ValueError('topk and timeout must be positive')
        super().__init__(topk, list(black_list or []))
        self.timeout = timeout

    def search(self, query: str) -> dict:
        """Synchronous entry point used by SearchAction's worker threads."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be a nonempty string')
        return asyncio.run(self._search(query))

    async def _search(self, query):
        try:
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise ImportError(
                'ParallelSearch requires: pip install -r requirements-parallel.txt'
            ) from exc

        # Bound initialization, tool execution and cleanup together. Cancellation
        # propagates through the SDK context managers and closes the HTTP client.
        async with asyncio.timeout(self.timeout):
            async with httpx.AsyncClient(
                headers={'User-Agent': self.user_agent}, timeout=self.timeout,
            ) as client:
                async with streamable_http_client(
                    self.endpoint, http_client=client,
                ) as (read, write, _):
                    async with ClientSession(
                        read, write,
                        read_timeout_seconds=timedelta(seconds=self.timeout),
                    ) as session:
                        await session.initialize()
                        result = await session.call_tool('web_search', {
                            'objective': query,
                            'search_queries': [query],
                        })

        if result.isError:
            message = '\n'.join(
                item.text for item in result.content if item.type == 'text'
            )
            raise RuntimeError(f'Parallel search failed: {message}')

        payload = result.structuredContent
        if payload is None:
            text = '\n'.join(
                item.text for item in result.content if item.type == 'text'
            )
            payload = json.loads(text)
        if not isinstance(payload, dict) or not isinstance(payload.get('results'), list):
            raise ValueError('Parallel search response must contain a results list')

        raw_results = []
        for item in payload['results']:
            url = item['url']
            if not url.startswith(('https://', 'http://')):
                continue
            raw_results.append((
                url, '\n'.join(item.get('excerpts', [])), item.get('title') or url,
            ))
        return self._filter_results(raw_results)
