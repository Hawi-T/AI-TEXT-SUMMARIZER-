# 📝 AI Text Summarizer (CLI)

A clean, beginner-friendly Command-Line AI Text Summarizer built in **Python 3**. It converts long articles, essays, or pasted text into crisp summaries, key takeaways, and keyword tags using **OpenAI** or **Google Gemini** with guaranteed **Structured Outputs**.

---

## 🌟 Features

- **Flexible Input**: Paste text directly into your terminal, provide a `.txt` file path, or run a quick test using the built-in `sample.txt`.
- **Guaranteed Structured Output**: Powered by Pydantic and the official OpenAI / Gemini APIs so the model returns strictly valid JSON without formatting glitches.
- **Dual Provider Support**: Switch effortlessly between **OpenAI** (`gpt-4o-mini`) and **Google Gemini** (`gemini-2.5-flash`) by changing one variable in `.env`.
- **Customizable Summary Styles**:
  - `short`: 2–3 concise sentences focusing on the core message.
  - `detailed`: Comprehensive 1–2 paragraph breakdown.
  - `bullet_points`: Digestible key takeaways formatted as bullet points.
- **Two-Step Display**: Inspect the **raw JSON response** first to learn how LLMs speak, followed by a **beautifully styled terminal view** with panels, tables, and tag badges.
- **Safe & Robust**: Full error handling for missing/invalid API keys, offline network conditions, empty text, and overly long inputs.

---

## 📚 Core AI Concepts (Beginner's Guide)

### 1. What is an LLM API?
A **Large Language Model (LLM)** like GPT-4o or Gemini is trained on billions of texts to predict the most probable next words. An **API (Application Programming Interface)** allows your Python code to send text to the AI model over the internet and receive a response without needing a supercomputer to run the model locally.

### 2. What are Tokens?
LLMs don't read words or letters directly; they process chunks of characters called **tokens**.
- As a general rule of thumb: `1 token ≈ 4 characters` or `0.75 words` in English.
- For example, `"Summarize this text"` is 4 words and approximately 5 tokens.
- APIs charge per token (both for your prompt input tokens and the generated output tokens).

### 3. System Prompt vs. User Prompt
- **System Prompt**: Gives the model its "identity", persona, constraints, and behavioral rules (e.g., *"You are an executive research assistant. Do not hallucinate. Output strictly in JSON format."*).
- **User Prompt**: The actual content or task to process (e.g., *"Summarize this article..."*).

### 4. What is Temperature?
Temperature is a floating-point number (typically between `0.0` and `1.0` or `2.0`) that controls how "creative" or "random" the model's word choices are:
- **Low Temperature (0.0 – 0.3)**: Highly deterministic, focused, and fact-focused. Best for summarization, code, and structured data extraction.
- **High Temperature (0.7 – 1.0+)**: More varied and imaginative. Best for brainstorming, creative writing, or poetry.
- In this project, we set `temperature=0.2` to ensure consistent, accurate summaries.

### 5. What is Structured Output?
In early AI development, developers would ask the model to *"Return your answer in JSON"*, but the model would often include markdown fences (````json ... ````), conversational chatter, or miss required fields, causing Python's `json.loads()` to crash!

**Structured Outputs** solve this permanently:
- We define a strict schema using Python's `pydantic` library (`summary`, `key_points`, `keywords`).
- The AI provider mathematically forces the model to adhere 100% to this schema during token generation.
- The result is guaranteed, valid JSON that parses cleanly every single time.

---

## 📂 Project Structure

```text
ai-text-summarizer/
│
├── .env                  # Your private API keys & settings (NEVER commit to Git!)
├── .env.example          # Safe template showing required configuration
├── .gitignore            # Tells Git to exclude secret keys and cache files
├── requirements.txt      # List of required Python packages
├── README.md             # Documentation and beginner tutorial
├── sample.txt            # Ready-to-use sample text for quick testing
└── main.py               # Main CLI script with modular, well-commented functions
```

---

## 🚀 Setup & Installation

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd "c:\Users\hp\Desktop\AI Text Summarizer"
```

### Step 2: (Optional but Recommended) Create a Virtual Environment
A virtual environment keeps your project packages isolated from your system Python.
```powershell
# Create virtual environment
python -m venv venv

# Activate it on Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# (If PowerShell blocks scripts, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
```

### Step 3: Install Required Libraries
```powershell
pip install -r requirements.txt
```

---

## 🔑 Getting Your API Keys

You can use either **OpenAI** or **Google Gemini** (or both!).

### Option A: Google Gemini (Free Tier Available)
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API Key**.
4. Copy your key.

### Option B: OpenAI
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys).
2. Sign in or create an account.
3. Click **Create new secret key**.
4. Copy your key.

---

## ⚙️ Configuration (.env)

Open `.env` (or copy `.env.example` to `.env` if not already present) and fill in your keys:

```ini
# Choose which LLM provider to use: 'openai' or 'gemini'
PROVIDER=openai

# OpenAI Settings
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini

# Google Gemini Settings (Free tier available)
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_MODEL=gemini-2.5-flash
```

> **Security Note:** `.env` is listed in `.gitignore` so your secret keys will never be accidentally committed or published to GitHub.

---

## 💻 How to Run

### 1. Interactive Mode (Recommended for Beginners)
Simply run the script with no flags:
```powershell
python main.py
```
You will be prompted to:
1. Choose how to provide text:
   - `1`: Paste text directly in terminal
   - `2`: Provide a `.txt` file path
   - `3`: Use the built-in `sample.txt` (Quick Test)
2. Choose your summary style:
   - `1`: Short (2–3 sentences)
   - `2`: Detailed (in-depth paragraph)
   - `3`: Bullet points

### 2. Command-Line Arguments Mode
You can also run summaries with one-liner terminal commands:

```powershell
# Summarize a text file with short style
python main.py --file sample.txt --style short

# Summarize with bullet points using Google Gemini
python main.py --file sample.txt --style bullet_points --provider gemini

# Summarize raw text directly
python main.py --text "Python was created by Guido van Rossum and released in 1991." --style short
```

---

## 🖥️ Example Output

```text
=================================================================
   AI TEXT SUMMARIZER (CLI) - Structured Outputs Edition
=================================================================
 Active Provider : OPENAI
 Active Model    : gpt-4o-mini
 API Key Loaded  : Yes
=================================================================

Processing text with OPENAI (short style)... Please wait...

=================================================================
 [STEP 1] RAW JSON RESPONSE FROM MODEL
=================================================================
Notice how the LLM returned exact, validated JSON without markdown fences:

{
  "summary": "Modern artificial intelligence is experiencing rapid advancement fueled by foundation models and the self-attention mechanisms of the Transformer architecture. These innovations are revolutionizing healthcare and diverse scientific domains, though critical hurdles like algorithmic bias, privacy, and safety remain.",
  "key_points": [
    "The 2017 Transformer architecture unlocked parallel training on massive internet datasets via self-attention.",
    "Foundation models are accelerating healthcare diagnostics and pharmaceutical protein structure prediction.",
    "Major deployment challenges include algorithmic bias, carbon footprint, privacy, and hallucination risks.",
    "Global regulatory efforts and research are pivoting toward AI safety, explainability, and governance."
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
╭── Executive Summary (SHORT - Provider: OPENAI) ──────────────────╮
│ Modern artificial intelligence is experiencing rapid advancement │
│ fueled by foundation models and the self-attention mechanisms    │
│ of the Transformer architecture. These innovations are          │
│ revolutionizing healthcare and diverse scientific domains,       │
│ though critical hurdles like algorithmic bias, privacy, and     │
│ safety remain.                                                  │
╰──────────────────────────────────────────────────────────────────╯

                       Key Takeaways
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #  ┃ Point                                                       ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1  │ • The 2017 Transformer architecture unlocked parallel      │
│    │ training on massive internet datasets via self-attention.  │
│ 2  │ • Foundation models are accelerating healthcare diagnostics │
│    │ and pharmaceutical protein structure prediction.            │
│ 3  │ • Major deployment challenges include algorithmic bias,     │
│    │ carbon footprint, privacy, and hallucination risks.         │
│ 4  │ • Global regulatory efforts and research are pivoting      │
│    │ toward AI safety, explainability, and governance.           │
└────┴─────────────────────────────────────────────────────────────┘

Keywords & Tags:
 #Artificial Intelligence   #Transformer Architecture   #Healthcare Innovation   #AI Safety 
```

---

## 🛡️ Error Handling Explained

The application guards against real-world errors:
| Error Condition | Behavior |
| :--- | :--- |
| **Missing API Key** | Displays helpful message pointing to `.env` and where to get a free key |
| **Invalid API Key** | Traps authentication error and informs user |
| **No Internet Connection** | Gracefully catches connection timeouts without an unhandled stack trace |
| **Empty or Whitespace Input** | Rejects input and asks user for valid text |
| **Input Too Long** | Warns if text exceeds safe token thresholds (35k chars) to avoid unexpected API bills |
| **Malformed JSON** | Pydantic schema validation ensures structure; catches and reports any deviations |

---

## 🐙 Pushing to GitHub

Follow these steps to create a Git repository and push it to GitHub:

### 1. Initialize Git in the project directory
```powershell
git init
```

### 2. Check status to ensure `.env` is ignored
```powershell
git status
```
*(Make sure `.env` does **NOT** show up in the untracked files list! Only `.gitignore`, `.env.example`, `requirements.txt`, `README.md`, `sample.txt`, and `main.py` should appear).*

### 3. Add and commit all files
```powershell
git add .
git commit -m "feat: initial commit of AI Text Summarizer with OpenAI and Gemini support"
```

### 4. Create a new repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. Repository name: `ai-text-summarizer`
3. Description: `Command-line AI Text Summarizer in Python with Structured Outputs (OpenAI & Gemini)`
4. Set to **Public** or **Private**.
5. Leave "Add a README", ".gitignore", and "license" unchecked (we already created them).
6. Click **Create repository**.

### 5. Link local repo and push
```powershell
# Rename default branch to main
git branch -M main

# Link to your remote GitHub repo (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ai-text-summarizer.git

# Push code to GitHub
git push -u origin main
```

---

## 📜 License
MIT License. Free to use, modify, and build upon for educational and commercial projects!
