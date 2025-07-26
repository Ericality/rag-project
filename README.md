# RAG Project: Legal Knowledge Retrieval

A dual-approach Retrieval-Augmented Generation (RAG) project demonstrating two
complementary strategies for legal information retrieval, using China's
Personal Information Protection Law as the test corpus.

## Architecture

| Demo | Approach | Core Technology | Best For |
|------|----------|----------------|----------|
| **Demo 1** | Vector Retrieval | Embedding + ChromaDB + LangChain Agent | "What does this paragraph say?" |
| **Demo 2** | Knowledge Graph | Triple Extraction + NetworkX BFS Traversal | "How are these concepts related?" |

## Project Structure

```
rag-project/
├── data/
│   └── 个人信息保护法样例.pdf      # Test corpus
├── demo1_vector_rag/
│   └── rag_qa.py                   # Demo 1: Vector-based RAG Agent
├── demo2_knowledge_graph/
│   └── knowledge_graph.py          # Demo 2: Knowledge Graph Construction & Query
├── .env.example                    # Environment template
├── .gitignore
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

## Key Differentiators

| Capability | Vector Retrieval | Knowledge Graph |
|------------|:---:|:---:|
| Semantic similarity search | Yes | No |
| Hierarchy / containment queries | No | Yes |
| Causal reasoning | Indirect | Explicit |
| Fuzzy natural-language queries | Yes | Limited |
| Hallucination risk | Medium | Low (explicit relations) |

## Future Roadmap

- [ ] Jaccard-based triple quality scoring
- [ ] Cross-validation of triple extraction stability
- [ ] GraphRAG: unified agent with both vector & graph tools
- [ ] Entity normalization (e.g., "personal data" → "personal information")

## License

MIT