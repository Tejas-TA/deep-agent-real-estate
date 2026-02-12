# sturdy-barnacle

Deep real-estate valuation agent combining:
- a **retrieval-augmented generation (RAG)** layer for property documents and structured metadata,
- a **large multimodal model (LMM)** to understand property photos,
- and a **classical ML model (e.g. logistic regression)** that estimates fair market value.

This repo is intentionally structured like a production-style Python project so you can show it to potential employers.

## Project structure

- `pyproject.toml` – project metadata and dependencies, managed with **uv**.
- `src/deep_agent/` – Python package for your core code (models, training, pipelines, etc.).
- `notebooks/` – exploratory work and experiments (e.g. `ingest_vectorize_zoning.ipynb`).
- `data/` – data directory (typically `raw/`, `processed/`, etc. under here; large files should not be committed).
- `tests/` – automated tests (currently a simple smoke test).
- `.env.example` – template for required environment variables (copy to `.env` locally).
- `.gitignore` – standard Python/uv/data ignore rules.

## Environment and tooling (uv)

This project is set up to use [`uv`](https://github.com/astral-sh/uv) instead of raw `pip`.

1. **Install uv** (if you don't have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

2. **Create and sync the environment**:

```bash
uv sync
```

This will create a virtual environment and install dependencies from `pyproject.toml`.

3. **Activate and run**:

```bash
uv run python -m deep_agent.main
```

Or, to launch Jupyter for your notebooks:

```bash
uv run jupyter lab
```

## Environment variables

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

Example keys:

- `OPENAI_API_KEY` – key for the LMM provider.
- `VISION_MODEL_ENDPOINT` – endpoint for vision inference if you self-host or use a gateway.
- `RAG_INDEX_PATH` – path to your pre-built vector index (under `data/processed/` by default).

## Next steps for you

- Implement your RAG pipeline and deep model code under `src/deep_agent/`.
- Keep training scripts, data loaders, and evaluation utilities in `src/` instead of notebooks once they stabilize.
- Add real tests in `tests/` to show employers you can write production-quality, tested ML code.
