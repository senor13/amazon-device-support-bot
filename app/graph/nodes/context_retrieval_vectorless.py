import json
import asyncio
import aiobreaker
import motor.motor_asyncio
from langchain_openai import ChatOpenAI

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
    return _mongo_client.support_bot


_tree_search_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_TREE_SEARCH_PROMPT = """
You are given a question and a tree structure of an Amazon device support document.
Each node has a node_id, title, and summary.
Find all nodes likely to contain the answer.

Question: {query}

Document tree:
{tree_json}

Reply ONLY with JSON: {{"node_list": ["node_id_1", "node_id_2"]}}
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


async def _search_tree(tree: list, query: str) -> list[dict]:
    #Calls _strip_text to get the lightweight tree
    stripped = _strip_text(tree)
    #Sends it to gpt-4o-mini: "which node_ids are relevant to this query?"
    prompt = _TREE_SEARCH_PROMPT.format(
        query=query,
        tree_json=json.dumps(stripped, indent=2),
    )
    result = await _tree_search_llm.ainvoke(prompt)
    #Gets back a list of node_ids
    node_ids = json.loads(result.content)["node_list"]
    # Calls _build_node_map to get a lookup dict
    node_map = _build_node_map(tree) 
    #Returns the full nodes (with text) for the matched IDs
    return [node_map[nid] for nid in node_ids if nid in node_map]
  


_DOC_IDS = ["kindle"]

#Fetches all matching documents from MongoDB (currently just ["kindle"]), then calls _search_tree on each doc in parallel via asyncio.gather. 
#Flattens all results and returns just the text fields as a flat list of strings.
async def _fetch_and_search(scrubbed_query: str) -> list[str]:
    db = get_mongo()
    docs = await db.document_trees.find(
        {"doc_id": {"$in": _DOC_IDS}}
    ).to_list(length=len(_DOC_IDS))

    if not docs:
        return []

    search_tasks = [_search_tree(doc["tree"], scrubbed_query) for doc in docs]
    results = await asyncio.gather(*search_tasks)
    all_nodes = [node for nodes in results for node in nodes]
    return [node.get("text", "") for node in all_nodes if node.get("text")]


async def context_retrieval_node(state: SupportBotState) -> dict:
    log = get_logger(state["request_id"], node="context_retrieval")

    try:
        context = await pageindex_breaker.call(_fetch_and_search, state["scrubbed_query"])
        if not context:
            log.warning("no_document_trees_found")
        else:
            log.info("retrieval_complete", num_nodes=len(context))
        return {"retrieved_context": context}

    except aiobreaker.CircuitBreakerError:
        log.warning("pageindex_circuit_open")
        return {"retrieved_context": []}
    except Exception as exc:
        log.warning("retrieval_failed", error=str(exc))
        return {"retrieved_context": []}
