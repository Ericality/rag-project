"""
Demo 2: Knowledge Graph Construction & Concept Hierarchy Query

Core approach:
- Extract (subject, relation, object) triples from legal text via LLM
- Build a directed knowledge graph using NetworkX
- search_graph() performs BFS traversal to answer hierarchy queries
  that vector retrieval cannot (e.g., "What are the subcategories of X?")
"""

import os
import json
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import networkx as nx

# ---- Configuration ----
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "./data/中华人民共和国个人信息保护法样例.pdf")

LLM = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
    max_tokens=4096,
    temperature=0,
)

# ---- Triple Extraction ----


def extract_triples_from_chunk(chunk_text: str) -> list:
    """Use LLM to extract (subject, relation, object) triples from a text chunk."""
    prompt = f"""Extract knowledge triples from the following legal text.

Format each triple as (subject, relation, object).

Requirements:
1. Subjects and objects should be legal concepts, entities, or actions
   (e.g., "personal information", "consent", "data processor")
2. Relation should be concise and use one of:
   - structure: "includes", "belongs to", "defined as", "applies to"
   - obligation: "requires", "prohibits", "may", "shall"
   - exemplification: "includes", "excludes", "for example"
   - Custom relations are allowed when none of the above fit
3. Extract all meaningful entity-relation pairs; do not omit any
4. Return ONLY a JSON array, nothing else

Text:
{chunk_text}

Return format:
[["subject1", "relation1", "object1"], ["subject2", "relation2", "object2"], ...]"""

    response = LLM.invoke(prompt)
    content = response.content.strip()

    try:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            triples = json.loads(match.group(0))
            valid_triples = []
            for t in triples:
                if len(t) == 3 and all(
                    isinstance(x, str) and len(x.strip()) > 0 for x in t
                ):
                    valid_triples.append(tuple(t))
            return valid_triples
    except (json.JSONDecodeError, Exception) as e:
        print(f"  [WARN] Failed to parse LLM response: {e}")

    return []


# ---- Graph Construction ----


def build_knowledge_graph(triples: list) -> nx.DiGraph:
    """Build a directed graph from a list of (subject, relation, object) triples."""
    G = nx.DiGraph()

    for subj, pred, obj in triples:
        subj = subj.strip()
        pred = pred.strip()
        obj = obj.strip()
        G.add_edge(subj, obj, relation=pred)

    return G


# ---- Graph Query ----


def search_graph(
    G: nx.DiGraph,
    concept: str,
    direction: str = "both",
    max_depth: int = 3,
    relation_filter: str = None,
) -> dict:
    """Query the graph for hierarchical relationships around a concept.

    Args:
        G: The knowledge graph.
        concept: Concept to query (supports substring matching).
        direction: "up" (parents), "down" (children), or "both".
        max_depth: Maximum traversal depth (1 = direct, 2+ = indirect).
        relation_filter: If set, only return edges matching this relation type.

    Returns:
        A dict with direct_parents, direct_children, indirect_relations,
        and statistics.
    """
    results = {
        "concept": concept,
        "direct_parents": [],
        "direct_children": [],
        "indirect_relations": [],
        "statistics": {},
    }

    matched_nodes = [n for n in G.nodes() if concept in n]

    if not matched_nodes:
        return {
            "concept": concept,
            "error": f"No node matching '{concept}' found.",
            "suggestion": list(G.nodes())[:20],
        }

    main_node = matched_nodes[0]

    # --- Direct relations (depth = 1) ---
    if direction in ("up", "both"):
        for source, _, data in G.in_edges(main_node, data=True):
            rel = data.get("relation", "")
            if relation_filter and relation_filter not in rel:
                continue
            results["direct_parents"].append(
                {"concept": source, "relation": rel, "depth": 1}
            )

    if direction in ("down", "both"):
        for _, target, data in G.out_edges(main_node, data=True):
            rel = data.get("relation", "")
            if relation_filter and relation_filter not in rel:
                continue
            results["direct_children"].append(
                {"concept": target, "relation": rel, "depth": 1}
            )

    # --- Indirect relations (depth > 1) ---
    if max_depth > 1:
        if direction in ("down", "both"):
            for depth in range(2, max_depth + 1):
                try:
                    descendants = nx.descendants_at_distance(G, main_node, depth)
                    for d in descendants:
                        try:
                            for path in nx.all_shortest_paths(G, main_node, d):
                                if len(path) - 1 == depth:
                                    steps = []
                                    for i in range(len(path) - 1):
                                        edge = G.get_edge_data(path[i], path[i + 1])
                                        rel = edge.get("relation", "?") if edge else "?"
                                        steps.append(f"{path[i]} -[{rel}]-> {path[i + 1]}")
                                    results["indirect_relations"].append(
                                        {"concept": d, "depth": depth, "path": steps}
                                    )
                                    break
                        except nx.NetworkXNoPath:
                            pass
                except nx.NetworkXError:
                    pass

        if direction in ("up", "both"):
            for depth in range(2, max_depth + 1):
                try:
                    ancestors = nx.ancestors(G, main_node)
                    for a in ancestors:
                        try:
                            for path in nx.all_shortest_paths(G, a, main_node):
                                if len(path) - 1 == depth:
                                    steps = []
                                    for i in range(len(path) - 1):
                                        edge = G.get_edge_data(path[i], path[i + 1])
                                        rel = edge.get("relation", "?") if edge else "?"
                                        steps.append(f"{path[i]} -[{rel}]-> {path[i + 1]}")
                                    if not any(
                                        ir["concept"] == a
                                        for ir in results["indirect_relations"]
                                    ):
                                        results["indirect_relations"].append(
                                            {"concept": a, "depth": depth, "path": steps}
                                        )
                                    break
                        except nx.NetworkXNoPath:
                            pass
                except nx.NetworkXError:
                    pass

    results["statistics"] = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "matched_nodes": matched_nodes,
    }
    return results


def search_graph_by_relation(G: nx.DiGraph, relation: str) -> list:
    """Return all triples in the graph that use the specified relation type."""
    results = []
    for source, target, data in G.edges(data=True):
        if relation == data.get("relation"):
            results.append((source, data["relation"], target))
    return results


# ---- Formatting Utilities ----


def format_graph_results(results: dict) -> str:
    """Convert search_graph() output into a readable string for LLM context."""
    if "error" in results:
        return f"[Graph query failed] {results['error']}"

    lines = [f"Concept: {results['concept']}"]
    stats = results.get("statistics", {})

    if results["direct_parents"]:
        lines.append("\nParents (pointing to this concept):")
        for item in results["direct_parents"]:
            lines.append(
                f"  {item['concept']} -[{item['relation']}]-> {results['concept']}"
            )

    if results["direct_children"]:
        lines.append("\nChildren (this concept points to):")
        for item in results["direct_children"]:
            lines.append(
                f"  {results['concept']} -[{item['relation']}]-> {item['concept']}"
            )

    if results["indirect_relations"]:
        lines.append(f"\nIndirect relations (depth > 1):")
        for item in results["indirect_relations"]:
            lines.append(f"  [{item['concept']}] (depth={item['depth']})")
            for step in item["path"]:
                lines.append(f"    {step}")

    if stats.get("matched_nodes"):
        lines.append(
            f"\nMatched nodes: {stats['matched_nodes'][:5]}..."
            if len(stats["matched_nodes"]) > 5
            else f"\nMatched nodes: {stats['matched_nodes']}"
        )

    return "\n".join(lines)


# ---- Demo ----
if __name__ == "__main__":
    print("=" * 60)
    print("Demo 2: Knowledge Graph Construction & Query")
    print("=" * 60)

    # 1. Load & split PDF
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"\nLoaded {len(docs)} page(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s)")

    # 2. Extract triples (limit to first 10 chunks to control API cost)
    all_triples = []
    for i, chunk in enumerate(chunks[:10]):
        text = chunk.page_content.strip()
        if len(text) < 30:
            continue
        triples = extract_triples_from_chunk(text)
        all_triples.extend(triples)
        print(f"  Chunk {i + 1}: extracted {len(triples)} triple(s)")

    unique_triples = list(set(all_triples))
    print(f"\nTotal: {len(all_triples)} raw / {len(unique_triples)} unique triples")

    # 3. Build graph
    G = build_knowledge_graph(unique_triples)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Show relation type distribution
    relation_counts = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "unknown")
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    print("Relation distribution (top 10):")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {rel}: {count}")

    # 4. Query examples
    print("\n" + "=" * 60)
    print("Concept Hierarchy Queries")
    print("=" * 60)

    for concept in ["个人信息", "同意", "未成年人"]:
        results = search_graph(G, concept, direction="both", max_depth=3)
        print(format_graph_results(results))
        print()

    # 5. Filter by relation type
    print("=" * 60)
    print("Filter by Relation Type")
    print("=" * 60)
    for rel_name in ["requires", "prohibits"]:
        triples = search_graph_by_relation(G, rel_name)
        print(f"\n  Relation '{rel_name}': {len(triples)} triple(s)")
        for s, p, o in triples[:5]:
            print(f"    ({s}) -[{p}]-> ({o})")

    print("\nDone.")