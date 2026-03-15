# Introducing News_Summarizer_Agent, your One-Stop AI Mode Agent

News_Summarizer_Agent searches multiple search API's in parallel, intelligently scores and consolidates results, scrapes full article content, and exports everything into a multimodal AI model which could be a local model or a Gemini model. You just need to set the API key in the .env file or give it the port number and start Ollama or llama.cpp. It will check those ports automatically. The front UI is based on Vite+React+TS. The backend is pure python and uses the news_search(https://github.com/AutoViML/news_search) tool which I created in AutoViML

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-Engine Search** | DuckDuckGo, Brightdata SERP, Tavily (active); Google/Gemini (available, excluded from default rotation) |
| **Smart Result Scoring** | Composite score: source count + domain credibility tier + recency signals |
| **Per-Engine CLI** | Run any engine individually with `--search "query"` |
| **TSV Export** | All raw results saved to `search_results.tsv` (overwrites each run) |
| **Full Content Scraping** | Crawlee `BeautifulSoupCrawler` extracts article text + images |
| **Markdown Report** | `search_text.md` — rich structured output ready for multimodal LLMs |
| **Robust Error Handling** | Retries for Brightdata timeouts, DuckDuckGo rate limits |
| **Windows Optimized** | Console encoding sanitization for emojis and special characters |

---

## 🗂 Project Structure

```
Deep_Search/
├── main.py                  # Core runner: search → score → export → scrape
├── scraper.py               # Crawlee-based scraper + markdown/TSV exporter
├── requirements.txt
├── .env                     # API keys (not committed)
├── search_results.tsv       # ← Generated: all raw results (tab-separated)
├── search_text.md           # ← Generated: scraped full content for LLM input
└── search_engines/
    ├── google.py            # Gemini 2.0 Flash grounding search (available, not in rotation)
    ├── duckduckgo.py        # ddgs library — no API key required
    ├── brightdata.py        # Brightdata SERP API (zone: serp_api1)
    └── tavily.py            # Tavily Research API
```

---

## 📋 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
BRIGHTDATA_API_KEY=your_brightdata_api_key
TAVILY_API_KEY=your_tavily_api_key
```

> **Note:** DuckDuckGo requires no API key. Google (Gemini grounding) requires `GEMINI_API_KEY` but is excluded from the default search rotation — it's available as a standalone CLI call.

---

## 🚀 Usage

### Run the full pipeline (search + score + export + scrape)

```powershell
python main.py
```

This will:
1. Search DuckDuckGo, Brightdata, and Tavily for the configured query
2. Deduplicate and score all results
3. Print results per engine + top 3 consolidated (with score breakdown)
4. Export **`search_results.tsv`** — all raw results, all engines
5. Scrape article pages and fetch video metadata → **`search_text.md`**

---

### Run any individual engine directly

Each search engine module has a consistent `--search` CLI:

```powershell
python search_engines/duckduckgo.py  --search "your query here" --max 5
python search_engines/brightdata.py  --search "your query here" --max 5
python search_engines/tavily.py      --search "your query here" --max 5
python search_engines/google.py      --search "your query here" --max 5
```

`--max` is optional (default: 5).

---

## 📊 Scoring System

Results are ranked using a **composite score** (higher = better):

| Component | Points | Criteria |
|---|---|---|
| **Source Count** | 10 pts each | How many engines independently found the result |
| **Domain Credibility** | 10 pts | Tier-1: Reuters, BBC, NYT, YouTube, etc. |
| | 5 pts | Tier-2: Facebook, Reddit, Substack, etc. |
| | 2 pts | Unknown/obscure domains |
| **Recency** | 0–10 pts | URL date patterns (`/2026/03/`) or snippet age ("21 hours ago", "today") |

**Example output:**
```
1. [34 pts] Jiang Xueqin: The Iran War — Singjupost.com
   Sources: DuckDuckGo, Brightdata
   Snippet: 3 days ago — PROFESSOR JIANG: Right, so we are in World War Three...
```

---

## 📄 Output Files

Both files are **overwritten** on each run — no timestamped versions are kept.

### `search_results.tsv`
Tab-separated file with all raw results from all active engines:

| Column | Content |
|---|---|
| `query` | The search query |
| `engine` | Which engine found it |
| `rank` | Rank within that engine's results |
| `title` | Result title |
| `url` | Result URL |
| `snippet` | First 300 chars of snippet |

### `search_text.md`
Structured markdown document for feeding to multimodal LLMs (Gemini, GPT-4o, Claude):

- **Article pages**: Full scraped text (up to 8,000 chars) + embedded images
- **YouTube videos**: Thumbnail image link + creator name via YouTube oEmbed API
- **Facebook/Instagram/Twitter**: URL + snippet (content behind login walls — not scraped)

---

## 🔧 Engine Notes

| Engine | Key Required | Strengths | Notes |
|---|---|---|---|
| **DuckDuckGo** | ❌ None | News articles, analysis pieces | Rate-limited; retry logic built-in |
| **Brightdata** | ✅ `BRIGHTDATA_API_KEY` | Primary source videos, social media | 60s timeout, 2 retries |
| **Tavily** | ✅ `TAVILY_API_KEY` | Research-quality sources, transcripts | Best for deep research |
| **Google** | ✅ `GEMINI_API_KEY` | Broad current events | Excluded from rotation; use CLI directly |

---

## ⚙️ Benchmark: *"What is the latest from Prof Jiang on Iran War?"*

| Engine | Result Quality |
|---|---|
| **DuckDuckGo** | ⭐⭐⭐⭐⭐ Best for mainstream news articles & analysis |
| **Brightdata** | ⭐⭐⭐⭐⭐ Best for primary source videos & transcripts |
| **Tavily** | ⭐⭐⭐⭐ Good research content, overlaps with Brightdata |
| **Google** | ⭐⭐ Returns general Iran war news, misses persona-specific content |
