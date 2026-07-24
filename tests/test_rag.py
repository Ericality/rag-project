"""
Test suite for rag-project core functions.

Covers:
- Vector retrieval (Demo 1)
- Knowledge graph construction & query (Demo 2)
- search_graph_by_relation() filtering
"""

import sys
from pathlib import Path

# Add project root to path so demo modules are importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "demo1_vector_rag"))
sys.path.insert(0, str(PROJECT_ROOT / "demo2_knowledge_graph"))
sys.path.insert(0, str(PROJECT_ROOT / "demo3_hybrid_agent"))

import pytest
from knowledge_graph import (
    build_knowledge_graph,
    search_graph,
    search_graph_by_relation,
    format_graph_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_triples():
    """A minimal set of triples that exercises all major relation types."""
    return [
        ("personal information", "includes", "biometric data"),
        ("personal information", "includes", "health data"),
        ("consent", "requires", "explicit notification"),
        ("processing", "prohibits", "coercion"),
        ("minor data", "requires", "parental consent"),
    ]


@pytest.fixture
def sample_graph(sample_triples):
    """A pre-built DiGraph from sample triples."""
    return build_knowledge_graph(sample_triples)


# ---------------------------------------------------------------------------
# Vector retrieval tests (skip when ChromaDB/data is not available)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires ChromaDB index & LLM API — disable for unit test")
def test_vector_retrieval_returns_results():
    pass  # placeholder: would load vectordb and assert len(docs) > 0


@pytest.mark.skip(reason="Requires ChromaDB index & LLM API — disable for unit test")
def test_vector_retrieval_irrelevant_query():
    pass  # placeholder: would test irrelevant query returns empty/graceful


# ---------------------------------------------------------------------------
# Knowledge graph construction
# ---------------------------------------------------------------------------

def test_build_graph_node_count(sample_graph, sample_triples):
    """Graph should contain one node per unique subject/object."""
    expected_nodes = set()
    for s, p, o in sample_triples:
        expected_nodes.add(s)
        expected_nodes.add(o)
    assert sample_graph.number_of_nodes() == len(expected_nodes)


def test_build_graph_edge_count(sample_graph, sample_triples):
    """Graph should contain one edge per triple."""
    assert sample_graph.number_of_edges() == len(sample_triples)


def test_build_graph_edge_relation_stored(sample_graph):
    """Each edge must carry the 'relation' attribute matching the triple."""
    for _, _, data in sample_graph.edges(data=True):
        assert "relation" in data
        assert isinstance(data["relation"], str)
        assert len(data["relation"]) > 0


# ---------------------------------------------------------------------------
# search_graph() — hierarchy query
# ---------------------------------------------------------------------------

def test_search_graph_known_direct_children(sample_graph):
    """Querying 'personal information' should return its direct children."""
    results = search_graph(sample_graph, "personal information", direction="down")
    children = [item["concept"] for item in results["direct_children"]]
    assert "biometric data" in children
    assert "health data" in children
    assert len(children) == 2


def test_search_graph_known_direct_parent(sample_graph):
    """Querying 'biometric data' should show its parent concept."""
    results = search_graph(sample_graph, "biometric data", direction="up")
    parents = [item["concept"] for item in results["direct_parents"]]
    assert "personal information" in parents
    assert len(parents) == 1


def test_search_graph_unknown_concept(sample_graph):
    """Querying a concept not in the graph should return an error dict."""
    results = search_graph(sample_graph, "nonexistent_concept_xyz")
    assert "error" in results


def test_search_graph_both_directions(sample_graph):
    """Direction 'both' should populate both parents and children."""
    results = search_graph(sample_graph, "consent", direction="both")
    assert len(results["direct_parents"]) >= 0
    assert len(results["direct_children"]) >= 1
    assert results["direct_children"][0]["concept"] == "explicit notification"


def test_search_graph_statistics(sample_graph):
    """Result dict must include statistics."""
    results = search_graph(sample_graph, "personal information")
    stats = results.get("statistics", {})
    assert stats.get("total_nodes") == sample_graph.number_of_nodes()
    assert stats.get("total_edges") == sample_graph.number_of_edges()


# ---------------------------------------------------------------------------
# search_graph_by_relation()
# ---------------------------------------------------------------------------

def test_filter_requires_relation(sample_graph):
    """Filtering by 'requires' should return 2 triples."""
    triples = search_graph_by_relation(sample_graph, "requires")
    assert len(triples) == 2
    assert ("consent", "requires", "explicit notification") in triples
    assert ("minor data", "requires", "parental consent") in triples


def test_filter_unknown_relation(sample_graph):
    """Filtering by a relation that does not exist should return an empty list."""
    triples = search_graph_by_relation(sample_graph, "allows")
    assert triples == []


# ---------------------------------------------------------------------------
# format_graph_results()
# ---------------------------------------------------------------------------

def test_format_error_results():
    """format_graph_results should handle error dicts gracefully."""
    error_result = {"error": "No node found"}
    output = format_graph_results(error_result)
    assert "failed" in output.lower() or "error" in output.lower()


def test_format_success_results(sample_graph):
    """format_graph_results should include the queried concept name."""
    results = search_graph(sample_graph, "consent", direction="down")
    output = format_graph_results(results)
    assert "consent" in output
    assert "explicit notification" in output