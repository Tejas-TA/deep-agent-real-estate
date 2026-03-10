# Deep Agent: AI-Powered Real Estate Orchestrator

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat)
![AI Framework](https://img.shields.io/badge/AI-LangGraph%20%7C%20MCP-purple?style=flat)
![ML Framework](https://img.shields.io/badge/ML-XGBoost%20%7C%20Optuna-orange?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)

Welcome to the **Deep Agent Real Estate Orchestrator** – a cutting-edge implementation of modern artificial intelligence and machine learning engineering. This project demonstrates a production-grade, multi-agent system designed to perform complex reasoning, valuation, and analysis in the real estate domain (specifically the Orlando market), showcasing the absolute forefront of AI.

---

## 🚀 Key Innovations & Architecture

This project is built around the **ReAct (Reasoning and Acting)** framework, managed by a supervisor agent that intelligently routes queries to specialized sub-agents. It leverages the emerging **Model Context Protocol (MCP)** and **Agent to Agent protocol (A2A)** to provide standardized, highly-performant tools to the AI via a FastAPI backend.

### 1. Multi-Agent Orchestration (LangGraph)
- **Hierarchical Routing:** A supervisor agent synthesizes user requests and conceptually delegates tasks to specialized domain agents (Zoning, Vision, Valuation, Market Expert).
- **LangSmith Observability:** Full tracing of thought processes, tool invocations, and agent latency for rapid debugging and MLOps monitoring.

### 2. MCP and A2A Integration
- **Standardized Tool Access:** Built a modular API layer using bare-metal `FastAPI` to expose programmatic tools to the AI.
- **Pydantic Validation:** Strict input/output schema validation ensuring agents never hallucinate tool parameters and execute deterministic backend code.
- **Intent Parsing & Semantic Routing:** The top-level Supervisor Node intercepts raw user queries and semantically determines which specialized sub-agent (or combination of agents) is required to fulfill the request.
- **Specialized Worker Nodes:** The Supervisor dispatches precise context payloads to isolated worker nodes, including:
  - **Zoning Expert:** Dedicated to querying the RAG Pinecone index.
  - **Vision Assessor:** Dedicated to processing LMM image classification.
  - **Valuation Analyst:** Dedicated to querying the XGBoost pipeline.
  - **Market Guide:** Dedicated to querying the fine-tuned LLM.
- **State Graph Orchestration:** LangGraph maintains a strict `State` object that flows between nodes, ensuring that an agent can request more information from another agent before synthesizing a final, comprehensive response back to the user. This enforces separation of concerns and dramatically reduces hallucinations at scale.

### 3. Advanced RAG & Vector Search
- **Pinecone Vector Database:** High-dimensional semantic search over heavy, unstructured local zoning laws (ingested, chunked, and vectorized).
- **Contextual Chunking:** Optimized retrieval augmented generation (RAG) using `text-embedding-3-small` and `RecursiveCharacterTextSplitter`.

### 4. Large Multimodal Models (LMM) in Production
- **Automated Damage Assessment:** Integrating GPT-4o-mini's visual capabilities to process raw property imagery, resizing and standardizing on the fly.
- **Metadata Extraction:** Extracting severity metrics and generating multimodal embeddings upserted directly into the vector DB.

### 5. Classical ML Meets Agentic AI
- **Fair Market Value (FMV) Modeling:** Building a highly robust `XGBoost` Regressor, bridging the gap between Generative AI and Classical ML.
- **Optuna Hyperparameter Tuning:** Automated, intelligent grid searching with Early Stopping to optimize RMSE and R² scores.
- **Explainable AI (XAI):** Utilizing `SHAP` (SHapley Additive exPlanations) and Permutation Importance so the Agent can confidently *explain* its valuation logic to users.

### 6. Geospatial & Amenity Intelligence
- **OpenStreetMap (OSM) API:** Live geographic querying using Haversine distance formulas to calculate neighborhood amenity metrics.
- **Algorithmic Walkability:** Creating dynamic walkability scores based on radius thresholds to augment the FMV model and Agent Market Reports.

### 7. Domain-Specific Fine-Tuning
- **Hyper-Local Expertise:** Curating specialized Orlando market datasets to fine-tune `GPT-4o-mini`, drastically reducing hallucinations and giving the Market Expert agent an unparalleled understanding of hyper-local real estate dynamics.

---

## 🛠️ Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Agentic Frameworks** | LangChain, LangGraph, OpenAI Assistant API |
| **Foundation Models** | GPT-4o, GPT-4o-mini, `text-embedding-3-small` |
| **Machine Learning** | XGBoost, Scikit-Learn, Optuna, SHAP |
| **Vector DB / Search** | Pinecone |
| **Backend & APIs** | Model Context Protocol (MCP), FastAPI, Pydantic |
| **Data Pipeline** | Pandas, Numpy, Pillow (PIL), OpenStreetMap |
| **DevOps / MLOps** | `uv` (Dependency Management), LangSmith, `.env` config |

---

## 📂 Project Structure

```text
sturdy-barnacle/
├── src/
│   └── deep_agent/                 # Placeholder for productionized multi-agent core
├── notebooks/                      # R&D, Model Training, & Agent Prototyping
│   ├── agent.py                    # Core ReAct agent implementation
│   ├── mcp_server.py               # MCP FastAPI entrypoint
│   ├── tools.py                    # Tool definitions and Pydantic schemas
│   ├── agent2agent.ipynb           # Multi-agent orchestrator logic
│   ├── xgboost-fmv-model-v2.ipynb  # Advanced XGBoost modeling & SHAP explainability
│   ├── ingest_vectorize_zoning.ipynb # RAG ingestion pipeline
│   ├── open_street_maps_api_test.ipynb # Geospatial data engineering
│   ├── mcp_layer.ipynb             # Server-client protocol testing
│   └── resize_metadata_tag_prop_images.ipynb # Vision processing pipeline
├── data/                           # RAG source files, Housing data, Images
├── tests/                          # Test suite (Integration, Unit)
├── .env                            # Environment Variables (OpenAI, Pinecone, LangSmith)
├── pyproject.toml                  # Modern uv-managed project dependencies
└── README.md                       # Project documentation
```

---

## 💻 Getting Started

### Prerequisites
- **Python 3.12+**
- **uv** (Rapid Python Package Installer & Resolver)
- Accounts for: **OpenAI**, **Pinecone**, **LangSmith**

### 1. Installation
Clone the repository and install dependencies using `uv` for hyper-fast resolution:

```bash
git clone https://github.com/Tejas-TA/sturdy-barnacle.git
cd sturdy-barnacle
uv sync
```

### 2. Environment Configuration
Create a `.env` file in the root directory (never commit this) with the following keys:
```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENV=your_pinecone_env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=orlando_real_estate
```

### 3. Exploring the AI Pipeline
The `notebooks/` directory contains the bleeding-edge R&D workflows. I recommend stepping through them in this order:
1. **RAG Vectorization:** `ingest_vectorize_zoning.ipynb`
2. **Machine Learning:** `xgboost-fmv-model-v2.ipynb`
3. **Computer Vision:** `resize_metadata_tag_prop_images.ipynb`
4. **Agent Orchestration:** `agent2agent.ipynb`

### 4. Running the MCP Server
To expose the agentic tools to your environment locally:
```bash
uvicorn notebooks.mcp_server:app --reload --port 8000
```

---

## 🌟 Future Roadmap
- [ ] Finalize migration of notebook prototypes into the `src/` modular structure.
- [ ] Increase Test-Driven Development (TDD) coverage in `tests/` directory to harden MCP tools.
- [ ] Seamless GUI integration for human-in-the-loop (HITL) approval on Fair Market Valuation predictions.
- [ ] Containerize architecture using Docker / Kubernetes for one-click Kubernetes deployment.

---

*This project exemplifies the seamless integration of Agentic Orchestration, Predictive Modeling, and Multimodal Vision AI-setting a benchmark for intelligent automation.*
