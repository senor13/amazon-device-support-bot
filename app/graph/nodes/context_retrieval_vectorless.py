import asyncio
import aiobreaker
import motor.motor_asyncio
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.graph.state import SupportBotState
from app.observability.logging import get_logger
from app.resilience.breakers import pageindex_breaker
from app.config import settings

_mongo_client = None

#Creates a MongoDB client once and reuses it (singleton pattern). Returns the support_bot database.
#Called every time retrieval runs but only creates the connection on the first call.
def get_mongo():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=3000,
        )
    return _mongo_client.support_bot #db


class TreeSearchResult(BaseModel):
    node_list: list[str]

#temp = 0 for consistency, same nodeids every time for same query
_tree_search_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(TreeSearchResult)

_TREE_SEARCH_PROMPT = """
You are given a question and a tree structure of an Amazon device support document.
Each node has a node_id, title, and summary.

Return the node_ids of the 1 to 3 nodes most directly relevant to the question.
Only include nodes whose content would specifically help answer the question.
Do not include general overview, introduction, or table of contents nodes unless the question is explicitly about them.
Prefer specific subsection nodes over broad chapter nodes.

Question: {query}

Document tree:
{tree_json}
"""

# Takes the full tree from MongoDB and removes the text field from every node recursively. Keeps only node_id, title, summary.
# This slimmed-down tree is what gets sent to the LLM — sending full text of every node would be thousands of tokens and expensive.
def _strip_text(tree: list) -> list:
    result = []
    for node in tree:
        n = {k: v for k, v in node.items() if k not in ("text",)}
        if "nodes" in node:
            n["nodes"] = _strip_text(node["nodes"])
        result.append(n)
    return result

#Flattens the tree into a simple dictionary: {node_id → full_node}.
#Makes it easy to look up a node by its ID in O(1) instead of searching the tree. Runs recursively to catch nested child nodes too.
def _build_node_map(tree: list, node_map: dict | None = None) -> dict:
    if node_map is None:
        node_map = {}
    for node in tree:
        node_map[node["node_id"]] = node
        if "nodes" in node:
            _build_node_map(node["nodes"], node_map)
    return node_map


def _build_parent_map(tree: list, parent_id: str | None = None, parent_map: dict | None = None) -> dict:
    if parent_map is None:
        parent_map = {}
    for node in tree:
        parent_map[node["node_id"]] = parent_id
        if "nodes" in node:
            _build_parent_map(node["nodes"], node["node_id"], parent_map)
    return parent_map


def _drop_ancestors(node_ids: list[str], parent_map: dict) -> list[str]:
    node_set = set(node_ids)
    return [nid for nid in node_ids if parent_map.get(nid) not in node_set]


async def _search_tree(tree: list, query: str) -> list[dict]:
    import json
    stripped = _strip_text(tree)
    prompt = _TREE_SEARCH_PROMPT.format(
        query=query,
        tree_json=json.dumps(stripped, indent=2),
    )
    result: TreeSearchResult = await _tree_search_llm.ainvoke(prompt)
    parent_map = _build_parent_map(tree)
    deduped = _drop_ancestors(result.node_list, parent_map)[:3]
    node_map = _build_node_map(tree)
    return [node_map[nid] for nid in deduped if nid in node_map]


#Fetches matching documents from MongoDB based on relevant_docs from state,
#then calls _search_tree on each doc in parallel via asyncio.gather.
#Flattens all results and returns just the text fields as a flat list of strings.
async def _fetch_and_search(scrubbed_query: str, doc_ids: list[str]) -> list[str]:
    db = get_mongo()
    docs = await db.document_trees.find(
        {"doc_id": {"$in": doc_ids}}
    ).to_list(length=len(doc_ids))

    if not docs:
        return []
    # This creates coroutine objects but doesn't execute them
    search_tasks = [_search_tree(doc["tree"], scrubbed_query) for doc in docs]
    # This actually runs all of them in parallel
    results = await asyncio.gather(*search_tasks)
    all_nodes = [node for nodes in results for node in nodes]
    return [node.get("text", "") for node in all_nodes if node.get("text")]


async def context_retrieval_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="context_retrieval")

    relevant_docs = state.get("relevant_docs", [])

    # out of scope — skip retrieval entirely
    if not relevant_docs:
        log.info("retrieval_skipped", reason="out_of_scope")
        return {"retrieved_context": []}

    try:
        context = await pageindex_breaker.call(_fetch_and_search, state["scrubbed_query"], relevant_docs)
        if not context:
            log.warning("no_document_trees_found", doc_ids=relevant_docs)
        else:
            log.info("retrieval_complete", num_nodes=len(context), doc_ids=relevant_docs)
        return {"retrieved_context": context}

    except aiobreaker.CircuitBreakerError:
        log.warning("pageindex_circuit_open")
        return {"retrieved_context": []}
    except Exception as exc:
        log.warning("retrieval_failed", error=str(exc))
        return {"retrieved_context": []}
