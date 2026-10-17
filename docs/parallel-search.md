# Optional Parallel search

[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
provides free web search without an account or API key. Anonymous searches use
Fast mode and have usage limits. `ParallelSearch` is an optional provider for
Level-Navi's existing `SearchAction`; it returns the same `url`, `summ` and
`title` fields as the other providers.

## Try a search without an LLM

From the repository root, use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-parallel.txt
python example/parallel_search.py "嫦娥六号 月球样品 返回时间"
```

This lightweight example calls `SearchAction` directly and prints up to three
results. It does not start an LLM or run the full planning agent. The optional
requirements include everything needed for this example, without GPU packages.
An empty result exits with status 1; provider errors are reported by
`SearchAction` as warnings.

## Use with the agent

For the full agent, follow the normal installation and model configuration in
the [README](../README.md), then install `requirements-parallel.txt` in that
environment. In `example/terminal.py`, add the provider import:

```python
from src.plugins import ParallelSearch
```

In the existing `SearchDistributor(...)` call, replace
`searcher_class=QihooWebSearch` with:

```python
searcher_class=ParallelSearch,
```

Run `python example/terminal.py` from the repository root. Your configured model
is still required. The existing defaults, Bing and you.com providers, and
benchmark CLI choices are unchanged; `--search_provider parallel` is not a CLI
option. Page selection still uses the existing `ContentFetcher`; this provider
only adds web search and does not route page fetching through Parallel.

## Requests and failures

The provider uses MCP Streamable HTTP at `https://search.parallel.ai/mcp`, with a
project User-Agent. It does not read or send search API credentials. Excerpts
become result summaries; the existing blacklist, PDF filter and `topk` limit
apply before `SearchAction` combines results. There is one attempt per query,
with a 30-second timeout. Network, protocol and service errors propagate to
`SearchAction`'s existing warning handling rather than triggering a fallback to
another provider. The MCP dependency is loaded only when Parallel is called.

Run deterministic provider and caller tests without network access:

```bash
python -m unittest discover -s tests -v
```
