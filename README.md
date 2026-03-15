# News_Summarizer_Agent: High-Speed AI Research Engine ⚡

News_Summarizer_Agent is a high-performance research tool that orchestrates multiple search APIs in parallel. It intelligently scores, consolidates, and synthesizes information using a "Snippet-First" architecture designed for maximum speed and zero UI noise.

Whether using local LLMs via Llama-Swap or Gemini-2.5-Flash, this agent delivers deep research results in a fraction of the time compared to traditional RAG pipelines.

---

## 🚀 Key Enhancements (v2.0)

| Feature | Enhancement | Benefit |
|---|---|---|
| **Ultra-Fast Search** | Removed slow engines (DDG); Optimized for Brightdata & Tavily. | **5x faster** search execution. |
| **Snippet Synthesis** | Replaced full-page scraping with high-quality snippet analysis. | Instant answers without scraper hangups. |
| **Silent UI** | Aggressively filtered milestones (Reflection & Web Research silenced). | Zero-noise, distraction-free research timeline. |
| **Parallel Execution** | Multi-query generation with simultaneous engine calls. | Comprehensive coverage in a single "swoop". |
| **Unified Synthesis** | Single final synthesis pass instead of looping summaries. | Highly coherent, cited answers with no redundancy. |

---

## 📸 Interface Preview

![News_Summarizer_Agent UI](docs/screenshot.png)

---

## ✨ Core Features

*   **Smart Query Generation**: Breaks down complex topics into targeted sub-queries.
*   **Result Scoring**: Composite ranking based on source count, domain authority (Reuters, BBC, etc.), and recency signals.
*   **Zero-Config UI**: Modern Vite+React+TS frontend with real-time milestone tracking.
*   **Flexible Backend**: Pure Python FastAPI backend compatible with any OpenAI-style API (local or cloud).
*   **Rich TSV Export**: All raw research data saved to `search_results.tsv` for auditability.

---

## 🗂 Project Structure

```
Deep_Search/
├── backend/
│   └── agent/
│       ├── app.py           # FastAPI Server (listening on :2024)
│       ├── research_agent.py # Core logic: query → parallel search → synthesis
│       └── local_llm.py     # Llama-Swap & Gemini integration
├── frontend/                # Vite + React + Tailwind UI
├── main.py                  # Optimized search pipeline runner
├── scraper.py               # (Legacy/Optional) Full content extraction
└── search_engines/          # Modular engine integrations (Tavily, Brightdata)
```

---

## 📋 Quick Start

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
```

### 3. Launch
```powershell
# Start Backend
python backend/agent/app.py

# Start Frontend
cd frontend
npm run dev
```
Open **`http://localhost:5173/app/`** to start searching.

---

## 📊 Engine Optimization

| Engine | Role | Why it's here |
|---|---|---|
| **Brightdata** | Social & Video | Best for primary sources, transcripts, and viral trends. |
| **Tavily** | Transcripts & Research | Specialized in finding high-density information for LLMs. |
| **DuckDuckGo** | *Direct* | No API key needed but we use it to avoid rate-limit delays. |

---

## 🛠 Advanced Usage

Each search engine can still be tested individually via CLI:
```powershell
python search_engines/brightdata.py --search "query" --max 3
python search_engines/tavily.py --search "query" --max 3
```

Raw search logs and scoring metrics are always exported to `search_results.tsv` for manual review.
