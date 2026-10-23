"""
app.py
======
FastAPI Web Application for AI Text Summarizer.
Provides both an interactive browser UI and a REST API endpoint.
Deployable to Render, Railway, Hugging Face, or local browser.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import the core summarization pipeline from main.py
from main import summarize_text, SummaryResponse, DEFAULT_GEMINI_MODEL

load_dotenv()

app = FastAPI(
    title="AI Text Summarizer",
    description="Modern browser interface for AI Text Summarizer powered by Google Gemini.",
    version="2.0.0",
)


class SummarizeRequest(BaseModel):
    text: str
    style: str = "short"
    mode: str = "structured"
    use_langchain: bool = False
    use_chunk: bool = False


class SummarizeWebResponse(BaseModel):
    raw_json: str
    parsed: SummaryResponse
    mode_used: str
    model: str


@app.post("/api/summarize", response_model=SummarizeWebResponse)
async def api_summarize(payload: SummarizeRequest):
    """
    Summarize endpoint processing request through Gemini.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()

    if not gemini_key:
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY is missing in your .env file."
        )

    try:
        raw_json_str, result = summarize_text(
            text=payload.text,
            style=payload.style,
            mode=payload.mode,
            use_langchain=payload.use_langchain,
            use_chunk=payload.use_chunk,
            gemini_key=gemini_key,
            gemini_model=gemini_model,
        )
        return SummarizeWebResponse(
            raw_json=raw_json_str,
            parsed=result,
            mode_used="langchain" if payload.use_langchain else payload.mode,
            model=gemini_model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample")
async def get_sample():
    """
    Returns the built-in sample.txt content for easy testing.
    """
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.txt")
    if os.path.isfile(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            return {"sample": f.read().strip()}
    return {"sample": "Artificial intelligence is advancing rapidly..."}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serves the modern, responsive web application interface.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Text Summarizer | Gemini Powered</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-glow: rgba(59, 130, 246, 0.35);
      --accent: #8b5cf6;
      --cyan: #06b6d4;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --success: #10b981;
      --font-main: 'Plus Jakarta Sans', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-main);
      min-height: 100vh;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.12) 0%, transparent 40%);
      padding: 2rem 1rem;
    }
    .container {
      max-width: 1080px;
      margin: 0 auto;
    }
    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(59, 130, 246, 0.12);
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      color: #60a5fa;
      margin-bottom: 0.75rem;
    }
    h1 {
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 1rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      backdrop-filter: blur(16px);
      border-radius: 16px;
      padding: 1.5rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .options-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }
    label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.35rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    select, textarea {
      width: 100%;
      background: rgba(10, 15, 26, 0.8);
      border: 1px solid var(--card-border);
      color: var(--text);
      font-family: var(--font-main);
      border-radius: 10px;
      padding: 0.75rem;
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    select:focus, textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px var(--primary-glow);
    }
    textarea {
      min-height: 220px;
      resize: vertical;
      line-height: 1.6;
    }
    .actions-bar {
      display: flex;
      gap: 0.75rem;
      margin-top: 1rem;
    }
    button {
      cursor: pointer;
      font-family: var(--font-main);
      font-weight: 600;
      font-size: 0.95rem;
      border-radius: 10px;
      padding: 0.75rem 1.25rem;
      border: none;
      transition: transform 0.15s, box-shadow 0.2s, background 0.2s;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: white;
      flex: 1;
      box-shadow: 0 4px 14px var(--primary-glow);
    }
    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      border: 1px solid var(--card-border);
    }
    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
    }
    .output-section {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    .result-box {
      background: rgba(10, 15, 26, 0.6);
      border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .result-heading {
      font-size: 0.85rem;
      font-weight: 700;
      color: #60a5fa;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.5rem;
    }
    .summary-text {
      line-height: 1.65;
      font-size: 0.95rem;
      color: #e2e8f0;
    }
    .key-points-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
      margin-top: 0.5rem;
    }
    .key-points-list li {
      position: relative;
      padding-left: 1.5rem;
      font-size: 0.9rem;
      color: #cbd5e1;
      line-height: 1.5;
    }
    .key-points-list li::before {
      content: "•";
      position: absolute;
      left: 0.5rem;
      color: var(--cyan);
      font-size: 1.2rem;
      line-height: 1;
    }
    .tags-container {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.5rem;
    }
    .tag-badge {
      background: rgba(6, 182, 212, 0.12);
      border: 1px solid rgba(6, 182, 212, 0.3);
      color: #38bdf8;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 600;
    }
    pre.raw-json {
      background: #050811;
      color: #a5f3fc;
      font-family: var(--font-mono);
      font-size: 0.8rem;
      padding: 1rem;
      border-radius: 8px;
      overflow-x: auto;
      max-height: 240px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .placeholder-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 350px;
      color: var(--text-muted);
      text-align: center;
      gap: 0.75rem;
    }
    .spinner {
      width: 24px;
      height: 24px;
      border: 3px solid rgba(255, 255, 255, 0.1);
      border-radius: 50%;
      border-top-color: var(--primary);
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">⚡ Powered by Google Gemini 3.8 Flash</div>
      <h1>AI Text Summarizer</h1>
      <p class="subtitle">Structured Outputs • Function Calling • LangChain • Map-Reduce</p>
    </header>

    <div class="grid">
      <!-- Input Column -->
      <div class="card">
        <div class="card-title">
          <span>Source Input</span>
          <button class="btn-secondary" style="padding: 0.35rem 0.75rem; font-size: 0.75rem;" onclick="loadSample()">Load Sample</button>
        </div>

        <div class="options-row">
          <div>
            <label>Summary Style</label>
            <select id="styleSelect">
              <option value="short">Short (2-3 sentences)</option>
              <option value="detailed">Detailed (In-depth)</option>
              <option value="bullet_points">Bullet Points</option>
            </select>
          </div>
          <div>
            <label>Mode</label>
            <select id="modeSelect">
              <option value="structured">Structured Output</option>
              <option value="function">Function Calling</option>
              <option value="langchain">LangChain LCEL</option>
              <option value="chunk">Map-Reduce Chunking</option>
            </select>
          </div>
        </div>

        <label>Document or Article Text</label>
        <textarea id="inputText" placeholder="Paste your article or notes here..."></textarea>

        <div class="actions-bar">
          <button class="btn-primary" id="btnSubmit" onclick="runSummarize()">Generate Summary</button>
          <button class="btn-secondary" onclick="clearAll()">Clear</button>
        </div>
      </div>

      <!-- Output Column -->
      <div class="card" id="outputCard">
        <div class="card-title">
          <span>Summary Result</span>
          <span id="statusIndicator" style="font-size: 0.8rem; color: var(--text-muted);">Ready</span>
        </div>

        <div id="placeholderState" class="placeholder-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.4;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
          <p>Enter text on the left and click <strong>Generate Summary</strong> to see structured results.</p>
        </div>

        <div id="resultsContent" class="output-section" style="display: none;">
          <div class="result-box">
            <div class="result-heading">Executive Summary</div>
            <p id="outSummary" class="summary-text"></p>
          </div>

          <div class="result-box">
            <div class="result-heading">Key Takeaways</div>
            <ul id="outKeyPoints" class="key-points-list"></ul>
          </div>

          <div class="result-box">
            <div class="result-heading">Keywords & Tags</div>
            <div id="outKeywords" class="tags-container"></div>
          </div>

          <details style="margin-top: 0.5rem; cursor: pointer;">
            <summary style="font-size: 0.8rem; color: var(--text-muted); font-weight: 600;">[Step 1] View Raw JSON Response</summary>
            <pre id="outRawJson" class="raw-json" style="margin-top: 0.5rem;"></pre>
          </details>
        </div>
      </div>
    </div>
  </div>

  <script>
    async function loadSample() {
      try {
        const res = await fetch('/api/sample');
        const data = await res.json();
        document.getElementById('inputText').value = data.sample;
      } catch (err) {
        alert('Failed to load sample text.');
      }
    }

    function clearAll() {
      document.getElementById('inputText').value = '';
      document.getElementById('placeholderState').style.display = 'flex';
      document.getElementById('resultsContent').style.display = 'none';
      document.getElementById('statusIndicator').innerText = 'Ready';
    }

    async function runSummarize() {
      const text = document.getElementById('inputText').value.trim();
      if (!text) {
        alert('Please enter or paste text to summarize.');
        return;
      }

      const style = document.getElementById('styleSelect').value;
      const modeVal = document.getElementById('modeSelect').value;
      const btn = document.getElementById('btnSubmit');
      const status = document.getElementById('statusIndicator');

      const isLangchain = modeVal === 'langchain';
      const isChunk = modeVal === 'chunk';
      const mode = (isLangchain || isChunk) ? 'structured' : modeVal;

      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Processing...';
      status.innerText = 'Analyzing with Gemini...';

      try {
        const response = await fetch('/api/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: text,
            style: style,
            mode: mode,
            use_langchain: isLangchain,
            use_chunk: isChunk
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Summarization failed.');
        }

        // Render Result
        document.getElementById('outSummary').innerText = data.parsed.summary;
        
        const kpList = document.getElementById('outKeyPoints');
        kpList.innerHTML = '';
        data.parsed.key_points.forEach(pt => {
          const li = document.createElement('li');
          li.innerText = pt;
          kpList.appendChild(li);
        });

        const kwBox = document.getElementById('outKeywords');
        kwBox.innerHTML = '';
        data.parsed.keywords.forEach(kw => {
          const span = document.createElement('span');
          span.className = 'tag-badge';
          span.innerText = '#' + kw.replace(/^#/, '');
          kwBox.appendChild(span);
        });

        document.getElementById('outRawJson').innerText = data.raw_json;

        document.getElementById('placeholderState').style.display = 'none';
        document.getElementById('resultsContent').style.display = 'flex';
        status.innerText = 'Completed (' + data.mode_used.toUpperCase() + ')';

      } catch (err) {
        alert('Error: ' + err.message);
        status.innerText = 'Error';
      } finally {
        btn.disabled = false;
        btn.innerHTML = 'Generate Summary';
      }
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
