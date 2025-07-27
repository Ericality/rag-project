# RAG Project: Legal Knowledge Retrieval

A dual-approach Retrieval-Augmented Generation (RAG) project demonstrating two
complementary strategies for legal information retrieval, using China's
Personal Information Protection Law as the test corpus.

## Architecture

| Demo | Approach | Core Technology | Best For |
|------|----------|----------------|----------|
| **Demo 1** | Vector Retrieval | Embedding + ChromaDB + LangChain Agent | "What does this paragraph say?" |
| **Demo 2** | Knowledge Graph | Triple Extraction + NetworkX BFS Traversal | "How are these concepts related?" |
| **Demo 3** | Hybrid Agent | Dual-tool Agent (vector + graph) + LLM routing | "Answer complex legal questions" |
| **Demo 4** | Streamlit Web UI | Streamlit + Docker + Demo Mode | One-click deploy, no API key needed |

## Project Structure

```
rag-project/
├── data/
│   └── 个人信息保护法样例.pdf         # Test corpus
├── demo1_vector_rag/
│   └── rag_qa.py                      # Demo 1: Vector-based RAG Agent
├── demo2_knowledge_graph/
│   └── knowledge_graph.py             # Demo 2: Knowledge Graph & Query
├── demo3_hybrid_agent/
│   └── agent_graphrag.py              # Demo 3: Hybrid Agent (vector + graph)
├── demo4_web_ui/
│   └── app.py                         # Demo 4: Streamlit Web UI
├── leetcode/                          # Algorithm practice
├── tests/
│   └── test_rag.py                    # pytest test suite
├── .github/workflows/
│   ├── test.yml                       # CI: auto-run pytest
│   └── docker-build.yml               # CI: auto-build Docker image
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### Option 1: Try the Live Demo (No Installation Required)

Open **[https://rag-project-demo.streamlit.app](https://rag-project-demo.streamlit.app)** in your browser.  
The demo runs in **Demo Mode** with pre-canned answers — no API key needed.

### Option 2: Run Locally

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your DeepSeek API key
```

#### 3. Run Demos

```bash
# Demo 1: Vector-based legal Q&A
python demo1_vector_rag/rag_qa.py

# Demo 2: Knowledge graph construction & hierarchy queries
python demo2_knowledge_graph/knowledge_graph.py

# Demo 3: Hybrid Agent
python demo3_hybrid_agent/agent_graphrag.py

# Demo 4: Streamlit Web UI
streamlit run demo4_web_ui/app.py
```

### Option 3: Run with Docker

```bash
docker-compose up -d
# Open http://localhost:8501
```

Pull the pre-built image:

```bash
docker pull ericality/rag-project:latest
```

## Core Concepts

### Demo 1 — Vector Retrieval
- Splits legal text into overlapping chunks (200 chars, 30 overlap)
- Embeds chunks with `all-MiniLM-L6-v2` and stores in ChromaDB
- Wraps retrieval in a LangChain Agent tool, allowing multi-turn reasoning

### Demo 2 — Knowledge Graph
- Extracts `(subject, relation, object)` triples from text via LLM
- Builds a directed graph using NetworkX
- `search_graph()` performs BFS traversal to answer hierarchy queries
  that vector retrieval cannot handle (e.g., "What are the subcategories of X?")

### Demo 3 — Hybrid Agent
- Combines vector retrieval and knowledge graph into a single LangChain Agent
- LLM routes queries to the appropriate tool based on tool descriptions
- Supports iterative multi‑round reasoning — retry with alternative tool if initial result insufficient

### Demo 4 — Streamlit Web UI
- One‑click deployment on [Streamlit Community Cloud](https://rag-project-demo.streamlit.app)
- **Demo Mode** with pre‑canned answers for zero‑cost instant trial
- Supports uploading custom PDF documents for general‑domain use
- Language toggle (EN / 中文) and quick‑select example questions

## Key Differentiators

| Capability | Vector Retrieval | Knowledge Graph | Hybrid Agent |
|------------|:---:|:---:|:---:|
| Semantic similarity search | Yes | No | Yes |
| Hierarchy / containment queries | No | Yes | Yes |
| Causal reasoning | Indirect | Explicit | Explicit |
| Fuzzy natural-language queries | Yes | Limited | Yes |
| Hallucination risk | Medium | Low | Medium (mitigated by tool routing) |

## Future Roadmap

- [x] Hybrid Agent (Demo 3) — dual-tool agent with LLM routing
- [x] Streamlit Web UI (Demo 4) — online demo & custom document upload
- [x] pytest test suite + GitHub Actions CI
- [x] Docker image auto-build & push to Docker Hub
- [ ] Jaccard-based triple quality scoring
- [ ] Cross-validation of triple extraction stability
- [ ] Entity normalization (e.g., "personal data" → "personal information")
- [ ] Evaluation metrics (MRR, precision@k) for retrieval quality

## License

MIT