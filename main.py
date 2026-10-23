"""
AI Text Summarizer (CLI & Web)
==============================
A beginner-friendly AI tool using ONLY the Google Gemini API (google-genai SDK).
Demonstrates:
  1. Native Structured Outputs (response_schema)
  2. Gemini Function Calling (--mode function)
  3. Prompt Engineering (Role, Constraints, Few-Shot, Anti-Hallucination)
  4. LangChain LCEL Mode (--langchain)
  5. Chunking / Map-Reduce for long documents (--chunk)
  6. Web deployment support (--web)

Language: Python 3
SDK: google-genai
"""

import os
import sys
import json
import argparse
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Rich terminal styling (with clean fallback)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Google GenAI SDK
from google import genai
from google.genai import types


# =====================================================================
# DEFAULT CONFIGURATION
# =====================================================================
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


# =====================================================================
# 1. STRUCTURED OUTPUT SCHEMA (Pydantic Model)
# =====================================================================
# What is Structured Output?
# Traditional LLMs output free-form text. Structured Outputs force the
# model to adhere strictly to a JSON schema, ensuring every required
# field and data type is returned exactly as specified.
# =====================================================================

class SummaryResponse(BaseModel):
    """
    Pydantic schema defining the structure of the LLM response.
    Gemini uses this schema to shape its output in structured mode.
    """
    summary: str = Field(
        description="The summary of the input text matching the requested style."
    )
    key_points: list[str] = Field(
        description="A list of 3 to 5 core takeaways or key points from the text."
    )
    keywords: list[str] = Field(
        description="A list of 3 to 5 relevant keywords or topic tags."
    )


# =====================================================================
# 2. PROMPT ENGINEERING (Techniques from the Prompting Guide)
# =====================================================================
# Prompt Engineering Techniques Used:
# 1. Role: Establishes persona ("Expert executive research assistant").
# 2. Clear Instructions: Explicit rules for short, detailed, or bullet styles.
# 3. Output Format: Strict schema adherence instructions.
# 4. One Few-Shot Example: Provides a concrete example of input -> output.
# 5. "Do Not Invent Facts": Explicit anti-hallucination guardrail.
# =====================================================================

def build_prompt(text: str, style: str) -> Tuple[str, str]:
    """
    Builds system prompt and user prompt applying prompt engineering best practices.
    """
    style_guidelines = {
        "short": (
            "- Write a concise summary of exactly 2-3 sentences.\n"
            "- Focus strictly on the overarching thesis."
        ),
        "detailed": (
            "- Write an in-depth, comprehensive summary (1-2 detailed paragraphs).\n"
            "- Cover the core thesis, supporting arguments, and final takeaway."
        ),
        "bullet_points": (
            "- Format the summary as clear bullet points (- point).\n"
            "- Highlight key chronological or logical developments."
        ),
    }

    selected_guideline = style_guidelines.get(style.lower(), style_guidelines["short"])

    # SYSTEM PROMPT: Role + Constraints + Few-Shot Example + Anti-Hallucination
    system_prompt = (
        "You are an expert executive research assistant and professional synthesizer of complex information.\n\n"
        "Strict Guidelines:\n"
        f"{selected_guideline}\n"
        "- Extract exactly 3 to 5 core key points (clear, actionable takeaways).\n"
        "- Extract exactly 3 to 5 single-word or short-phrase keywords/topic tags.\n"
        "- CRITICAL: Do NOT invent facts, assume unstated details, or extrapolate.\n"
        "- Maintain an objective, neutral, professional tone.\n\n"
        "--- FEW-SHOT DEMONSTRATION ---\n"
        "Example Input:\n"
        "'The Apollo 11 mission was the first spaceflight that landed humans on the Moon in July 1969. "
        "American astronauts Neil Armstrong and Buzz Aldrin spent over two hours exploring the lunar surface.'\n"
        "Example Output JSON:\n"
        "{\n"
        '  "summary": "Apollo 11 achieved the historic milestone of landing the first humans on the Moon in July 1969.",\n'
        '  "key_points": [\n'
        '    "Apollo 11 was the first spaceflight to land humans on the Moon in July 1969.",\n'
        '    "Astronauts Neil Armstrong and Buzz Aldrin explored the lunar surface for over two hours."\n'
        "  ],\n"
        '  "keywords": ["Apollo 11", "Moon landing", "Neil Armstrong", "Space exploration"]\n'
        "}\n"
        "--- END OF DEMONSTRATION ---"
    )

    # USER PROMPT: The actual payload
    user_prompt = (
        f"Please analyze and summarize the following text according to the '{style}' style:\n\n"
        "--- SOURCE TEXT ---\n"
        f"{text.strip()}\n"
        "--- END OF SOURCE TEXT ---"
    )

    return system_prompt, user_prompt


# =====================================================================
# 3. FUNCTION CALLING DEFINITION
# =====================================================================
# How does Function Calling differ from Structured Output?
# - Structured Output: Tells the LLM "Format your text answer directly
#   as valid JSON matching this schema."
# - Function Calling: Tells the LLM "You have access to a tool named
#   save_summary(). Do not output text; call this function with arguments."
# Function calling is the foundation of AI Agents that take real actions.
# =====================================================================

def save_summary(summary: str, key_points: list[str], keywords: list[str]) -> dict:
    """
    Saves the structured summary containing executive summary, key points, and keywords.
    
    Args:
        summary: The summary of the source text.
        key_points: 3 to 5 core takeaways.
        keywords: 3 to 5 relevant topic tags.
    """
    return {
        "summary": summary,
        "key_points": key_points,
        "keywords": keywords,
    }


# =====================================================================
# 4. GEMINI API CALL FUNCTIONS
# =====================================================================

def call_gemini_structured(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
) -> Tuple[str, SummaryResponse]:
    """
    Calls Google Gemini using native Structured Output (response_schema).
    """
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=SummaryResponse,
    )

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )

    raw_json_str = response.text
    if not raw_json_str:
        raise ValueError("Gemini returned an empty response.")

    parsed_response = SummaryResponse.model_validate_json(raw_json_str)
    return raw_json_str, parsed_response


def clean_markdown_json(raw: str) -> str:
    """Strips markdown code fences (```json ... ```) if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def get_function_tool() -> types.Tool:
    """
    Defines the save_summary function schema for Gemini Function Calling.
    Using FunctionDeclaration gives strict control without AFC auto-execution overhead.
    """
    func_decl = types.FunctionDeclaration(
        name="save_summary",
        description="Saves the structured summary containing executive summary, key points, and keywords.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "summary": types.Schema(
                    type=types.Type.STRING,
                    description="The executive summary of the source text."
                ),
                "key_points": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="3 to 5 core key points from the text."
                ),
                "keywords": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="3 to 5 topic keywords or tags."
                ),
            },
            required=["summary", "key_points", "keywords"],
        ),
    )
    return types.Tool(function_declarations=[func_decl])


def call_gemini_function(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
) -> Tuple[str, SummaryResponse]:
    """
    Calls Google Gemini using Function Calling (tools=[save_summary]).
    The model generates a tool call containing the structured arguments.
    """
    client = genai.Client(api_key=api_key)
    tool = get_function_tool()

    fc_system_prompt = (
        f"{system_prompt}\n\n"
        "You have access to the save_summary tool. "
        "Invoke save_summary with the extracted summary, key_points, and keywords."
    )

    config = types.GenerateContentConfig(
        system_instruction=fc_system_prompt,
        tools=[tool],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["save_summary"],
            )
        ),
    )

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )

    # 1. Extract arguments from function call
    if response.function_calls:
        first_call = response.function_calls[0]
        call_args = dict(first_call.args)
        raw_json_str = json.dumps(call_args, indent=2)
        parsed_response = SummaryResponse(**call_args)
        return raw_json_str, parsed_response

    # 2. Fallback: Parse text response (stripping markdown fences if present)
    if response.text:
        cleaned_json = clean_markdown_json(response.text)
        parsed_response = SummaryResponse.model_validate_json(cleaned_json)
        return cleaned_json, parsed_response

    raise ValueError("Gemini did not invoke the function or return text.")




# =====================================================================
# 5. COOKBOOK IDEA: MAP-REDUCE CHUNKING FOR LONG TEXTS
# =====================================================================
# How Chunking Works:
# When text is very long, it may exceed context budgets or lose focus.
# Map-Reduce splits the text into chunks:
# 1. Map Step: Summarize each chunk individually into concise notes.
# 2. Reduce Step: Combine all chunk summaries into a final structured summary.
# =====================================================================

def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    """
    Splits text into manageable chunks respecting paragraphs.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        if current_length + len(para_clean) > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para_clean]
            current_length = len(para_clean)
        else:
            current_chunk.append(para_clean)
            current_length += len(para_clean)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks if chunks else [text]


def summarize_with_chunks(
    text: str,
    style: str,
    model: str,
    api_key: str,
    mode: str = "structured",
) -> Tuple[str, SummaryResponse]:
    """
    Map-Reduce summarizer for long documents.
    """
    import time
    chunks = chunk_text(text)
    print(f"\n[Map-Reduce] Document split into {len(chunks)} chunks.", flush=True)

    client = genai.Client(api_key=api_key)
    intermediate_summaries = []

    # Map Step: Summarize each chunk
    for i, chunk in enumerate(chunks, 1):
        print(f"  -> [Map {i}/{len(chunks)}] Processing chunk ({len(chunk):,} chars)...", flush=True)
        prompt = f"Condense the following section into 2-3 essential sentences:\n\n{chunk}"
        for attempt in range(3):
            try:
                res = client.models.generate_content(model=model, contents=prompt)
                intermediate_summaries.append(res.text.strip())
                time.sleep(2)  # Courteous pause to respect Gemini RPM rate limit
                break
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 2:
                    print(f"     [Backoff] Rate limit or spike detected. Pausing 8s before retry (attempt {attempt+2}/3)...", flush=True)
                    time.sleep(8)
                    continue
                raise

    # Reduce Step: Synthesize all chunk summaries into final structured output
    print("  -> [Reduce Step] Synthesizing all chunk summaries into final structured output...", flush=True)
    combined_intermediate = "\n\n".join(
        [f"Section {i+1} Summary:\n{s}" for i, s in enumerate(intermediate_summaries)]
    )

    system_prompt, user_prompt = build_prompt(combined_intermediate, style)

    if mode == "function":
        return call_gemini_function(system_prompt, user_prompt, model, api_key)
    return call_gemini_structured(system_prompt, user_prompt, model, api_key)


# =====================================================================
# 6. PIPELINE DISPATCHER & ERROR HANDLING
# =====================================================================

def summarize_text(
    text: str,
    style: str = "short",
    mode: str = "structured",
    use_langchain: bool = False,
    use_chunk: bool = False,
    gemini_key: Optional[str] = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> Tuple[str, SummaryResponse]:
    """
    Validates inputs and routes the request to the appropriate method.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty! Please provide valid text to summarize.")

    if not gemini_key or gemini_key.startswith("your_") or "here" in gemini_key:
        raise ValueError(
            "Missing or invalid GEMINI_API_KEY in your .env file.\n"
            "Get your free key from: https://aistudio.google.com/app/apikey"
        )

    # Character count guardrail
    char_count = len(text)
    if char_count > 35000 and not use_chunk:
        raise ValueError(
            f"Input text is {char_count:,} characters (exceeds 35,000 char threshold).\n"
            "Please add the --chunk flag to summarize long text using Map-Reduce!"
        )

    import time
    for attempt in range(2):
        try:
            # Route 1: LangChain Mode
            if use_langchain:
                from summarizer_langchain import summarize_with_langchain
                return summarize_with_langchain(text, style, gemini_key, gemini_model)

            # Route 2: Chunking (Map-Reduce) Mode
            if use_chunk:
                return summarize_with_chunks(text, style, gemini_model, gemini_key, mode)

            # Route 3: Standard Structured vs Function Calling
            system_prompt, user_prompt = build_prompt(text, style)

            if mode == "function":
                return call_gemini_function(system_prompt, user_prompt, gemini_model, gemini_key)
            else:
                return call_gemini_structured(system_prompt, user_prompt, gemini_model, gemini_key)

        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt == 0:
                print("\n[Notice] Gemini service busy (503). Retrying automatically in 4s...", flush=True)
                time.sleep(4)
                continue
            if "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                raise ConnectionError("Authentication failed: Your Gemini API key is invalid.")
            elif "NOT_FOUND" in err_msg or "404" in err_msg:
                raise ConnectionError(
                    f"Model '{gemini_model}' was not found (404). "
                    "Update GEMINI_MODEL in your .env file."
                )
            elif "UNAVAILABLE" in err_msg or "503" in err_msg:
                raise ConnectionError(
                    "Gemini service is temporarily experiencing high demand (503). "
                    "Please retry in a few moments."
                )
            elif "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                raise ConnectionError("Rate limit exceeded (429). Please slow down requests.")
            elif "Connection" in type(e).__name__ or "timeout" in err_msg.lower():
                raise ConnectionError("Network connection error. Check your internet connection.")
            raise RuntimeError(f"Summarizer error: {err_msg}")


# =====================================================================
# 7. DISPLAY RESULTS
# =====================================================================

def display_result(raw_json_str: str, result: SummaryResponse, style: str, mode: str):
    """
    Prints the raw JSON response once, then the formatted output.
    """
    print("\n" + "=" * 65, flush=True)
    print(f" [STEP 1] RAW JSON RESPONSE FROM MODEL (Mode: {mode.upper()})", flush=True)
    print("=" * 65, flush=True)
    print("Notice the raw JSON data received from Gemini:\n", flush=True)

    if RICH_AVAILABLE:
        syntax = Syntax(raw_json_str, "json", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        print(raw_json_str)

    print("\n" + "=" * 65, flush=True)
    print(" [STEP 2] PARSED & BEAUTIFULLY FORMATTED OUTPUT", flush=True)
    print("=" * 65, flush=True)

    if RICH_AVAILABLE:
        title = f"[bold green]Executive Summary ({style.upper()} - Mode: {mode.upper()})[/bold green]"
        console.print(Panel(result.summary, title=title, border_style="cyan", padding=(1, 2)))

        table = Table(title="[bold yellow]Key Takeaways[/bold yellow]", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4, justify="center")
        table.add_column("Point", style="white")

        for idx, point in enumerate(result.key_points, 1):
            table.add_row(str(idx), f"[white]{point}[/white]")
        console.print(table)

        keyword_badges = " ".join([f"[black on cyan] #{kw.strip('# ')} [/black on cyan]" for kw in result.keywords])
        console.print("\n[bold yellow]Keywords & Tags:[/bold yellow]")
        console.print(keyword_badges)
        console.print()
    else:
        print(f"\n--- SUMMARY ({style.upper()}) ---")
        print(result.summary)
        print("\n--- KEY POINTS ---")
        for idx, pt in enumerate(result.key_points, 1):
            print(f"  {idx}. {pt}")
        print("\n--- KEYWORDS ---")
        print("  " + ", ".join([f"#{kw}" for kw in result.keywords]))
        print()


# =====================================================================
# 8. INPUT HELPERS
# =====================================================================

def read_input() -> str:
    """
    Menu for reading text interactively.
    """
    print("\nSelect input method:")
    print("  [1] Paste text directly")
    print("  [2] Read from a .txt file path")
    print("  [3] Use built-in 'sample.txt' (Quick Test)")

    while True:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        if choice == "1":
            print("\nPaste your text below. (Type 'END' on a new line when done):\n")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if text:
                return text
            print("Error: No text entered. Please try again.")

        elif choice == "2":
            file_path = input("\nEnter .txt file path: ").strip().strip('"\'')
            if not os.path.isfile(file_path):
                print(f"Error: File '{file_path}' not found.")
                continue
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()

        elif choice == "3":
            sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.txt")
            if os.path.isfile(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            print("Error: 'sample.txt' not found.")
        else:
            print("Invalid choice. Enter 1, 2, or 3.")


def choose_style() -> str:
    """
    Menu for selecting style.
    """
    print("\nChoose summary style:")
    print("  [1] Short (Concise, 2-3 sentences)")
    print("  [2] Detailed (Comprehensive paragraph)")
    print("  [3] Bullet Points (Bullet-pointed key takeaways)")

    while True:
        choice = input("\nEnter choice (1, 2, or 3) [default: 1]: ").strip()
        if choice in ("", "1"):
            return "short"
        elif choice == "2":
            return "detailed"
        elif choice == "3":
            return "bullet_points"
        print("Invalid choice. Enter 1, 2, or 3.")


# =====================================================================
# 9. MAIN CLI ENTRYPOINT
# =====================================================================

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="AI Text Summarizer - Google Gemini Edition (Structured Outputs, Function Calling, LangChain, Chunks)."
    )
    parser.add_argument("-f", "--file", help="Path to a .txt file to summarize.")
    parser.add_argument("-t", "--text", help="Raw text string to summarize directly.")
    parser.add_argument(
        "-s", "--style", choices=["short", "detailed", "bullet_points"],
        help="Summary style: 'short', 'detailed', or 'bullet_points'."
    )
    parser.add_argument(
        "-m", "--mode", choices=["structured", "function"], default="structured",
        help="Summarization mode: 'structured' (default) or 'function' (Gemini function calling)."
    )
    parser.add_argument(
        "--langchain", action="store_true",
        help="Run summarization using LangChain LCEL chain."
    )
    parser.add_argument(
        "--chunk", action="store_true",
        help="Use Map-Reduce chunking for long documents."
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Launch the browser web application interface."
    )

    args = parser.parse_args()

    # If --web flag is given, start the web interface
    if args.web:
        import uvicorn
        from app import app
        print("\nStarting AI Text Summarizer Web App at http://127.0.0.1:8001 ...")
        uvicorn.run(app, host="127.0.0.1", port=8001)
        return

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()

    # Determine mode label
    if args.langchain:
        active_mode = "langchain"
    elif args.mode == "function":
        active_mode = "function"
    else:
        active_mode = "structured"

    # Banner
    print("\n" + "=" * 65, flush=True)
    print("   AI TEXT SUMMARIZER - Google Gemini Edition", flush=True)
    print("=" * 65, flush=True)
    print(f" Provider       : Google Gemini", flush=True)
    print(f" Model          : {gemini_model}", flush=True)
    print(f" Mode           : {active_mode.upper()}", flush=True)
    print(f" Chunking       : {'Enabled' if args.chunk else 'Disabled'}", flush=True)
    print(f" API Key Loaded : {'Yes' if gemini_key and not gemini_key.startswith('your_') else 'No'}", flush=True)
    print("=" * 65, flush=True)

    # 1. Get Text
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: Specified file '{args.file}' does not exist.")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text.strip()
    else:
        text = read_input()

    # 2. Get Style
    style = args.style or choose_style()

    # 3. Summarize
    print(f"\nProcessing text with Gemini ({active_mode} mode, {style} style)... Please wait...\n", flush=True)
    try:
        raw_json_str, result = summarize_text(
            text=text,
            style=style,
            mode=args.mode,
            use_langchain=args.langchain,
            use_chunk=args.chunk,
            gemini_key=gemini_key,
            gemini_model=gemini_model,
        )
        # 4. Display
        display_result(raw_json_str, result, style, active_mode)

    except (ValueError, ConnectionError, RuntimeError) as err:
        print(f"\n[ERROR] {err}\n", flush=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
