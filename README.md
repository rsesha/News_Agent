# Introducing News Agent: Three Powerful Search Engines, One Final Answer

![News Agent](docs/news_agent.png)

## Overview
This is a **high-performance AI search agent** that orchestrates multiple search engines to deliver synthesized, accurate answers. Think of it as a "Google AI Overviews" engine built for developers, optimized for speed and zero hallucination.

You can plug this agent into:
- RAG Retrieval Pipelines to get accurate summarized information from the web
- Real world agents that need fast accurate information about products and news from internet
---

## Architecture

```
News_Agent/
├── Backend (Python/FastAPI)
│   ├── agent/ - Core AI research engine
│   │   ├── research_agent.py - Orchestrates search → synthesis pipeline
│   │   ├── local_llm.py - Llama-Swap & Gemini integration
│   │   ├── app.py - FastAPI server (port 2024)
│   │   ├── prompts.py - LLM prompt templates
│   │   └── graph.py - LangGraph state management
│   └── search_engines/ - Modular engine integrations
│       ├── brightdata.py - Social/video search
│       ├── tavily.py - Research transcripts
│       └── duckduckgo.py - Direct web search
│
├── Frontend (React/Vite/TypeScript)
│   ├── components/ - UI components
│   │   ├── InputForm.tsx - Query input
│   │   ├── ChatMessagesView.tsx - Results display
│   │   └── ActivityTimeline.tsx - Search progress
│   └── hooks/useStream.ts - WebSocket streaming
│
└── Infrastructure
    ├── Dockerfile + docker-compose.yml
    └── main.py - CLI entry point
```

---

## Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | Python 3.11, FastAPI | Research orchestration, LLM integration |
| **Frontend** | React, Vite, TypeScript | Chat interface, real-time streaming |
| **LLM Layer** | Gemini API + Local (Llama-Swap/Ollama) | Query generation, synthesis |
| **Search Engines** | Brightdata, Tavily, DuckDuckGo | Multi-source information retrieval |

---

## How It Works

1. **Query Generation**: LLM breaks down user question into 3 targeted sub-queries
2. **Parallel Search**: All 3 search engines execute simultaneously on each query
3. **Snippet Synthesis**: High-quality snippets extracted (not full-page scraping for speed)
4. **Final Answer**: Single synthesis pass produces cited answer with source links

---

## Interface Preview

![News Agent UI](docs/screenshot.png)

---

## Key Files

| File | Description |
|------|-------------|
| `backend/agent/research_agent.py` | Core research pipeline (query → search → answer) |
| `backend/agent/app.py` | FastAPI server with streaming endpoint |
| `frontend/src/App.tsx` | Main React application |
| `main.py` | CLI entry point for search pipeline |
| `docker-compose.yml` | Containerized deployment |

---

## Dependencies

**Backend:**
- `fastapi`, `uvicorn` - Web framework
- `langchain`, `langgraph` - LLM orchestration
- `google-genai` - Gemini API client
- `duckduckgo-search`, `requests` - Search engines

**Frontend:**
- `react`, `react-dom`
- `tailwindcss` - Styling
- `shadcn/ui` - Component library

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_key
BRIGHTDATA_API_KEY=your_key
TAVILY_API_KEY=your_key
# DuckDuckGo doesn't need an API key!

# Optional
LOCAL_MODEL_PORT=8080  # use 8080 if running Llama-Swap, 11434 for Ollama
```

### 3. Launch
```bash
# Start Backend
python backend/agent/app.py

# Start Frontend
cd frontend
npm run dev
```
Open **`http://localhost:5173/app/`** to start searching.

---

## Engine Optimization

| Engine | Role | Why it's here |
|---|---|---|
| **Brightdata** (API Key needed) | Social & Video | Best for primary sources, transcripts, and viral trends |
| **Tavily** (API Key needed) | Transcripts & Research | Specialized in finding high-density information for LLMs |
| **DuckDuckGo** (No API key) | Direct Web | Free, no rate-limit delays |

---

## Notable Features

- ⚡ **5x faster** than traditional search via parallel execution
- 🔄 **Zero-config** - Just add API keys to `.env`
- 🎯 **Cited answers** - Every claim linked to source
- 📱 **Real-time streaming** - Watch research progress live
- 🐳 **Docker-ready** - Full containerization support

---

## Advanced Usage

Each search engine can still be tested individually via CLI:
```bash
python search_engines/brightdata.py --search "query" --max 3
python search_engines/tavily.py --search "query" --max 3
python search_engines/duckduckgo.py --search "query" --max 3
```

Raw search result URLs and scoring metrics are always exported to `search_results.tsv` for manual review.