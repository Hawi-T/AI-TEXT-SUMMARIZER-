# 📝 AI Text Summarizer (Google Gemini Edition)

A beginner-friendly, production-ready AI Text Summarizer built in **Python 3** using **ONLY the Google Gemini API** (`google-genai` SDK).

Includes full support for **Structured Outputs**, **Function Calling**, **Prompt Engineering best practices**, **LangChain LCEL Chains**, **Map-Reduce Document Chunking**, and a **modern Browser Web App interface** ready for cloud deployment!

---

## 🌟 Key Features

1. **Gemini Only**: Uses the official `google-genai` SDK with `gemini-3.8-flash`.
2. **Native Structured Output**: Employs Pydantic schemas with Gemini's `response_schema` to guarantee 100% valid JSON without markdown fences.
3. **Gemini Function Calling (`--mode function`)**: Demonstrates tool use by letting Gemini call `save_summary()` with structured arguments.
4. **Prompt Engineering**: Applies all core Prompting Guide techniques:
   - **Role Definition** (Executive Research Assistant)
   - **Explicit Instructions** (Short, Detailed, or Bullet Points)
   - **One Few-Shot Demonstration** (Concrete anchor example)
   - **Anti-Hallucination Guardrail** ("Do not invent facts")
5. **LangChain Integration (`--langchain`)**: Clean, separate LCEL pipeline (`summarizer_langchain.py`) chaining `PromptTemplate | ChatGoogleGenerativeAI | with_structured_output`.
6. **Cookbook Map-Reduce Chunking (`--chunk`)**: Handles long texts exceeding 35,000 characters by splitting into chunks, summarizing sections (Map), and synthesizing the final summary (Reduce).
7. **Browser Web App Interface (`--web`)**: Built with FastAPI and a modern UI to run and test directly in your browser or deploy to Render/Railway.
8. **Flexible Inputs**: Paste text directly in the terminal, provide a `.txt` file, or run a quick test with `sample.txt`.

---

## 🧠 Core AI Concepts Explained Simply

### 1. What is an LLM API?
A **Large Language Model (LLM)** like Gemini 3.8 Flash is a neural network trained to understand and generate human language. An **API (Application Programming Interface)** allows your Python code to send text across the internet to Google's data centers and receive AI-generated responses.

### 2. What are Tokens?
LLMs process language in numerical chunks called **tokens**:
- **Rule of thumb**: `1 token ≈ 4 characters` or `0.75 words` in English.
- The phrase *"Summarize this text"* is 4 words and ~5 tokens.
- APIs bill based on prompt input tokens and generated response tokens.

### 3. System Prompt vs. User Prompt
- **System Prompt**: Defines the model's identity, guidelines, constraints, and behavioral guardrails (e.g., *"You are an executive research assistant. Do not invent facts."*).
- **User Prompt**: The specific task and data payload to process (e.g., *"Summarize this text according to the short style: [Text here]"*).

### 4. What is Temperature?
Temperature controls randomness:
- **Low Temperature (0.0 – 0.3)**: Predictable, focused, and strictly factual. Ideal for summarization.
- **High Temperature (0.7 – 1.0+)**: Creative, varied, and imaginative.
- Newer Gemini models automatically optimize sampling, so we omit manual temperature overrides to prevent schema interference.

### 5. Structured Output vs. Function Calling (What's the difference?)
Both guarantee structured data, but serve different purposes:
- **Structured Output (`response_schema`)**: The model's *text reply itself* is mathematically constrained to conform to a JSON schema. Best for pure data extraction and summarization.
- **Function Calling (`tools=[save_summary]`)**: The model acts as an **agent**. It decides *to invoke a Python function* and passes structured arguments to your code. This is how AI agents take real actions (e.g., querying databases, calling APIs, sending emails).

### 6. What is a LangChain Chain?
A **Chain** in LangChain Expression Language (LCEL) links multiple steps using the pipe operator (`|`):
```text
[Input Variables] -> [Prompt Template] -> [Gemini Chat Model] -> [Structured Output] -> [SummaryResponse]
```
Each component's output feeds automatically into the next.

### 7. What is Map-Reduce Chunking?
When an article is too long to fit comfortably in a single prompt:
- **Map Step**: Split the document into sections and summarize each section individually.
- **Reduce Step**: Combine all section summaries and synthesize them into the final, cohesive summary.

---

## 📂 Project Structure

```text
ai-text-summarizer/
│
├── .env                  # Private API keys & settings (NEVER committed to Git!)
├── .env.example          # Safe template showing required configuration
├── .gitignore            # Tells Git to exclude secret keys and cache files
├── requirements.txt      # Python dependencies (google-genai, langchain, rich, fastapi, uvicorn)
├── README.md             # Comprehensive documentation and tutorial
├── sample.txt            # Ready-to-use sample text for quick testing
├── main.py               # Core CLI application with clean, modular functions
├── summarizer_langchain.py # Standalone LangChain LCEL summarization module
└── app.py                # FastAPI web server and browser interface
```

---

## 🚀 Setup & Installation

### Step 1: Navigate to the Project Folder
```powershell
cd "c:\Users\hp\Desktop\AI Text Summarizer"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🔑 Getting Your Free Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API key** -> **Create API key in new project**.
4. Copy your key.

---

## ⚙️ Configuration (`.env`)

Edit your [.env](file:///c:/Users/hp/Desktop/AI%20Text%20Summarizer/.env) file:
```ini
# Google Gemini Configuration
GEMINI_MODEL=gemini-3.8-flash
GEMINI_API_KEY=AIzaSyYourCopiedKeyHere
```

> **Security Note:** `.env` is listed in `.gitignore` so your private API key is never pushed to GitHub.

---

## 💻 How to Run Every Mode

### 1. Default Mode (Native Structured Output)
```powershell
# Interactive menu:
python main.py

# One-liner with sample.txt:
python main.py --file sample.txt --style short
```

### 2. Function Calling Mode (`--mode function`)
```powershell
python main.py --file sample.txt --style bullet_points --mode function
```

### 3. LangChain Mode (`--langchain`)
```powershell
python main.py --file sample.txt --style detailed --langchain
```

### 4. Map-Reduce Chunking Mode (`--chunk`)
```powershell
# Splits long texts, summarizes chunks, and reduces to final output:
python main.py --file sample.txt --style short --chunk
```

### 5. Browser Web Application Mode (`--web`)
```powershell
python main.py --web
# OR
python app.py
```
Open your browser to: **[http://127.0.0.1:8001](http://127.0.0.1:8001)** to use the browser interface!

---

## 🖥️ Example Output

```text
=================================================================
   AI TEXT SUMMARIZER - Google Gemini Edition
=================================================================
 Provider       : Google Gemini
 Model          : gemini-3.8-flash
 Mode           : STRUCTURED
 Chunking       : Disabled
 API Key Loaded : Yes
=================================================================

Processing text with Gemini (structured mode, short style)... Please wait...

=================================================================
 [STEP 1] RAW JSON RESPONSE FROM MODEL (Mode: STRUCTURED)
=================================================================
Notice the raw JSON data received from Gemini:

{
  "summary": "Modern artificial intelligence is advancing rapidly through foundation models and Transformer architectures. While revolutionizing healthcare and science, issues of safety, bias, and ethics remain pivotal.",
  "key_points": [
    "Transformer self-attention enables massive training parallelization.",
    "Foundation models accelerate medical diagnostics and drug discovery.",
    "Major deployment challenges include algorithmic bias and hallucination risks.",
    "Global governance is prioritizing explainability and alignment."
  ],
  "keywords": [
    "Artificial Intelligence",
    "Transformer Architecture",
    "Healthcare Innovation",
    "AI Safety"
  ]
}

=================================================================
 [STEP 2] PARSED & BEAUTIFULLY FORMATTED OUTPUT
=================================================================
╭── Executive Summary (SHORT - Mode: STRUCTURED) ──────────────────╮
│ Modern artificial intelligence is advancing rapidly through       │
│ foundation models and Transformer architectures...               │
╰──────────────────────────────────────────────────────────────────╯

                       Key Takeaways
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #  ┃ Point                                                       ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1  │ Transformer self-attention enables massive training...      │
│ 2  │ Foundation models accelerate medical diagnostics...         │
│ 3  │ Major deployment challenges include algorithmic bias...     │
│ 4  │ Global governance is prioritizing explainability...         │
└────┴─────────────────────────────────────────────────────────────┘

Keywords & Tags:
 #Artificial Intelligence   #Transformer Architecture   #Healthcare Innovation   #AI Safety
```

---

## 🐙 Pushing to GitHub

```powershell
# 1. Check Git status (ensure .env is NOT tracked)
git status

# 2. Add all files
git add .

# 3. Commit
git commit -m "feat: complete Gemini AI Text Summarizer with structured outputs, function calling, langchain, chunking, and web app"

# 4. Set branch to main
git branch -M main

# 5. Link to your GitHub repo (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ai-text-summarizer.git

# 6. Push to GitHub
git push -u origin main
```

---

## 📜 License
MIT License. Free to use, adapt, and deploy!
