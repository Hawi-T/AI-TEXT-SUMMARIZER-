"""
AI Text Summarizer (CLI)
========================
A beginner-friendly command-line AI tool that summarizes text using
OpenAI or Google Gemini with Structured Outputs (guaranteed JSON).

Author: Antigravity Pair Programmer
Language: Python 3
"""

import os
import sys
import json
import argparse
from typing import Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Rich terminal styling (with standard fallback)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# Default model names
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


# =====================================================================
# 1. STRUCTURED OUTPUT SCHEMA (Pydantic Model)
# =====================================================================
# What is Structured Output?
# Traditional LLMs output free-form text, which can sometimes deviate
# from what your code expects. Structured Outputs enforce a strict JSON
# schema so the model is mathematically constrained to return valid JSON
# matching the exact fields and data types we define here.
# =====================================================================

class SummaryResponse(BaseModel):
    """
    Pydantic schema defining the structure of the LLM response.
    Both OpenAI and Google Gemini use this schema to shape their output.
    """
    summary: str = Field(
        description="The summary of the input text matching the requested style."
    )
    key_points: list[str] = Field(
        description="A list of 3 to 5 key takeaways or core points from the text."
    )
    keywords: list[str] = Field(
        description="A list of 3 to 5 relevant keywords or topic tags."
    )


# =====================================================================
# 2. PROMPT ENGINEERING
# =====================================================================
# Prompt Engineering is how we instruct the model to produce the best result.
# We apply three core techniques:
# 1. Role: Defines who the model is (Executive AI Summarizer).
# 2. Instructions: Clear guidelines on tone, length, and hallucination prevention.
# 3. Output Format: Explicit schema rules (short/detailed/bullet points).
# =====================================================================

def build_prompt(text: str, style: str) -> Tuple[str, str]:
    """
    Constructs the system prompt and user prompt based on the selected style.
    
    Args:
        text: The source text to summarize.
        style: 'short', 'detailed', or 'bullet_points'.
        
    Returns:
        tuple of (system_prompt, user_prompt)
    """
    style_guidelines = {
        "short": (
            "- Write a concise summary of exactly 2-3 sentences.\n"
            "- Focus only on the overarching message without minor details."
        ),
        "detailed": (
            "- Write a comprehensive summary (1-2 detailed paragraphs).\n"
            "- Cover the main thesis, supporting context, and conclusion thoroughly."
        ),
        "bullet_points": (
            "- Format the summary field as concise bullet points (using '-' prefixes).\n"
            "- Highlight the chronological or logical flow of ideas."
        ),
    }

    selected_guideline = style_guidelines.get(style.lower(), style_guidelines["short"])

    # SYSTEM PROMPT: Sets role, constraints, and behavior rules
    system_prompt = (
        "You are an expert executive research assistant and professional synthesizer of complex information.\n\n"
        "Your mission:\n"
        "Analyze the provided text objectively and produce a high-quality, structured summary.\n\n"
        "Strict Guidelines:\n"
        f"{selected_guideline}\n"
        "- Extract exactly 3 to 5 core key points (actionable, clear takeaways).\n"
        "- Extract exactly 3 to 5 single-word or short-phrase keywords/topic tags.\n"
        "- Do NOT hallucinate or extrapolate facts not present in the source text.\n"
        "- Maintain an objective, neutral, professional tone.\n"
        "- Output strictly in the requested JSON structure."
    )

    # USER PROMPT: Provides the payload / raw text
    user_prompt = (
        f"Please analyze and summarize the following text according to the '{style}' style:\n\n"
        "--- START OF TEXT ---\n"
        f"{text.strip()}\n"
        "--- END OF TEXT ---"
    )

    return system_prompt, user_prompt


# =====================================================================
# 3. PROVIDER FUNCTIONS (OpenAI & Google Gemini)
# =====================================================================
# Each provider is kept in its own small, isolated function.
# This makes switching providers seamless and easy to maintain.
# =====================================================================

def call_openai(system_prompt: str, user_prompt: str, model: str, api_key: str) -> Tuple[str, SummaryResponse]:
    """
    Calls the OpenAI API using the official openai Python SDK.
    Uses OpenAI's native Structured Outputs feature (client.beta.chat.completions.parse)
    which guarantees 100% schema compliance.
    """
    try:
        from openai import OpenAI, OpenAIError, AuthenticationError, APIConnectionError
    except ImportError:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)

    # What is Temperature?
    # Temperature controls randomness:
    # 0.0 = deterministic, focused, predictable (great for summarization & facts)
    # 1.0 = creative, varied (great for creative writing)
    # We use 0.2 for accurate, grounded summarization.
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=SummaryResponse,
        temperature=0.2,
    )

    # Raw JSON string received directly from the model
    raw_json_str = completion.choices[0].message.content
    # Parsed Pydantic instance provided by OpenAI SDK
    parsed_response = completion.choices[0].message.parsed

    if not parsed_response:
        # Fallback manual parse if parsed object is None
        parsed_response = SummaryResponse.model_validate_json(raw_json_str)

    return raw_json_str, parsed_response


def call_gemini(system_prompt: str, user_prompt: str, model: str, api_key: str) -> Tuple[str, SummaryResponse]:
    """
    Calls the Google Gemini API using the official google-genai SDK.
    Uses Gemini's structured output configuration (response_schema=SummaryResponse).
    """
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError
    except ImportError:
        raise RuntimeError("Google GenAI SDK is not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)

    # Configure Gemini for Structured JSON Output
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

    # Parse and validate against the Pydantic schema
    parsed_response = SummaryResponse.model_validate_json(raw_json_str)
    return raw_json_str, parsed_response


# =====================================================================
# 4. SUMMARIZATION DISPATCHER & ERROR HANDLING
# =====================================================================

def summarize_text(
    text: str,
    style: str,
    provider: str,
    openai_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> Tuple[str, SummaryResponse]:
    """
    Validates inputs, builds prompts, dispatches to the selected provider,
    and handles potential network and API errors with clear user feedback.
    """
    # 1. Validation: Empty or whitespace text
    if not text or not text.strip():
        raise ValueError("Input text is empty! Please provide meaningful text to summarize.")

    # 2. Validation: Text length safeguard (approx 30,000 characters ~ 7,500 tokens)
    char_count = len(text)
    if char_count > 35000:
        raise ValueError(
            f"Input text is too long ({char_count:,} characters). "
            "Please limit input to under 35,000 characters to prevent context window overflow."
        )

    # 3. Build Prompt
    system_prompt, user_prompt = build_prompt(text, style)

    provider_clean = provider.strip().lower()

    # 4. Dispatch to selected provider
    if provider_clean == "openai":
        if not openai_key or openai_key.startswith("your_") or "here" in openai_key:
            raise ValueError(
                "Missing or invalid OPENAI_API_KEY in your .env file.\n"
                "Get an API key from https://platform.openai.com/api-keys\n"
                "or set PROVIDER=gemini in .env to use Google Gemini."
            )
        try:
            return call_openai(system_prompt, user_prompt, openai_model, openai_key)
        except Exception as e:
            err_msg = str(e)
            if "AuthenticationError" in type(e).__name__ or "invalid_api_key" in err_msg.lower():
                raise ConnectionError("Authentication failed: Your OpenAI API key is invalid or expired.")
            elif "APIConnectionError" in type(e).__name__ or "Connection" in type(e).__name__:
                raise ConnectionError("No internet connection or cannot reach OpenAI servers.")
            elif "RateLimitError" in type(e).__name__ or "quota" in err_msg.lower():
                raise ConnectionError("OpenAI quota exceeded or rate limit hit. Check your OpenAI billing or switch to PROVIDER=gemini.")
            raise RuntimeError(f"OpenAI error: {err_msg}")

    elif provider_clean == "gemini":
        if not gemini_key or gemini_key.startswith("your_") or "here" in gemini_key:
            raise ValueError(
                "Missing or invalid GEMINI_API_KEY in your .env file.\n"
                "Get your free API key from https://aistudio.google.com/app/apikey and paste it in .env"
            )
        try:
            return call_gemini(system_prompt, user_prompt, gemini_model, gemini_key)
        except Exception as e:
            err_msg = str(e)
            if "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                raise ConnectionError("Authentication failed: Your Gemini API key is invalid or lacks permissions.")
            elif "Connection" in type(e).__name__ or "timeout" in err_msg.lower():
                raise ConnectionError("No internet connection or cannot reach Google Gemini servers.")
            raise RuntimeError(f"Gemini error: {err_msg}")

    else:
        raise ValueError(f"Unknown PROVIDER '{provider}' in .env. Please set PROVIDER to 'openai' or 'gemini'.")


# =====================================================================
# 5. RESULT DISPLAY & FORMATTING
# =====================================================================

def display_result(raw_json_str: str, result: SummaryResponse, style: str, provider: str):
    """
    Displays the raw JSON payload first (for learning/inspection),
    followed by a beautifully formatted breakdown of the summary,
    key points, and keywords.
    """
    print("\n" + "=" * 65)
    print(" [STEP 1] RAW JSON RESPONSE FROM MODEL")
    print("=" * 65)
    print("Notice how the LLM returned exact, validated JSON without markdown fences:\n")

    if RICH_AVAILABLE:
        syntax = Syntax(raw_json_str, "json", theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        print(raw_json_str)

    print("\n" + "=" * 65)
    print(" [STEP 2] PARSED & BEAUTIFULLY FORMATTED OUTPUT")
    print("=" * 65)

    if RICH_AVAILABLE:
        # Display Summary Panel
        title = f"[bold green]Executive Summary ({style.upper()} - Provider: {provider.upper()})[/bold green]"
        console.print(Panel(result.summary, title=title, border_style="cyan", padding=(1, 2)))

        # Display Key Points Table
        table = Table(title="[bold yellow]Key Takeaways[/bold yellow]", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4, justify="center")
        table.add_column("Point", style="white")

        for idx, point in enumerate(result.key_points, 1):
            table.add_row(str(idx), f"[white]{point}[/white]")
        console.print(table)

        # Display Keywords
        keyword_badges = " ".join([f"[black on cyan] #{kw.strip('# ')} [/black on cyan]" for kw in result.keywords])
        console.print("\n[bold yellow]Keywords & Tags:[/bold yellow]")
        console.print(keyword_badges)
        console.print()
    else:
        print(f"\n--- SUMMARY ({style.upper()} | {provider.upper()}) ---")
        print(result.summary)
        print("\n--- KEY POINTS ---")
        for idx, pt in enumerate(result.key_points, 1):
            print(f"  {idx}. {pt}")
        print("\n--- KEYWORDS ---")
        print("  " + ", ".join([f"#{kw}" for kw in result.keywords]))
        print()


# =====================================================================
# 6. USER INPUT HELPER FUNCTIONS
# =====================================================================

def read_input() -> str:
    """
    Guides the user interactively to either:
    1. Paste text directly.
    2. Provide a .txt file path.
    3. Use the built-in sample.txt file.
    """
    print("\nSelect input method:")
    print("  [1] Paste text directly")
    print("  [2] Read from a .txt file path")
    print("  [3] Use built-in 'sample.txt' (Quick Test)")

    while True:
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        if choice == "1":
            print("\nPaste your text below.")
            print("(When finished, type 'END' on a new line and press Enter):\n")
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
            file_path = input("\nEnter the path to your .txt file: ").strip().strip('"\'')
            if not file_path:
                print("Error: File path cannot be empty.")
                continue
            if not file_path.lower().endswith(".txt"):
                print("Warning: File does not have a .txt extension, attempting to read anyway...")

            if not os.path.isfile(file_path):
                print(f"Error: File '{file_path}' not found. Check the path and try again.")
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    print(f"Error: File '{file_path}' is empty.")
                    continue
                print(f"Successfully loaded {len(content):,} characters from '{file_path}'.")
                return content
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        return f.read().strip()
                except Exception as e:
                    print(f"Error reading file encoding: {e}")
            except Exception as e:
                print(f"Error reading file: {e}")

        elif choice == "3":
            sample_path = os.path.join(os.path.dirname(__file__), "sample.txt")
            if os.path.isfile(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                print(f"Loaded 'sample.txt' ({len(content):,} characters).")
                return content
            else:
                print("Error: 'sample.txt' not found in project folder.")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def choose_style() -> str:
    """
    Prompts the user to select their desired summary style.
    """
    print("\nChoose summary style:")
    print("  [1] Short (Concise, 2-3 sentences)")
    print("  [2] Detailed (Comprehensive paragraph)")
    print("  [3] Bullet Points (Crisp bullet-pointed takeaways)")

    while True:
        choice = input("\nEnter choice (1, 2, or 3) [default: 1]: ").strip()
        if choice in ("", "1"):
            return "short"
        elif choice == "2":
            return "detailed"
        elif choice == "3":
            return "bullet_points"
        else:
            print("Invalid choice. Enter 1, 2, or 3.")


# =====================================================================
# 7. MAIN APPLICATION ENTRYPOINT
# =====================================================================

def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="AI Text Summarizer - CLI tool with OpenAI and Gemini Structured Outputs."
    )
    parser.add_argument("-f", "--file", help="Path to a .txt file to summarize.")
    parser.add_argument("-t", "--text", help="Raw text string to summarize directly.")
    parser.add_argument(
        "-s", "--style", choices=["short", "detailed", "bullet_points"],
        help="Summary style: 'short', 'detailed', or 'bullet_points'."
    )
    parser.add_argument(
        "-p", "--provider", choices=["openai", "gemini"],
        help="Override LLM provider ('openai' or 'gemini')."
    )

    args = parser.parse_args()

    # Determine provider and configuration
    env_provider = os.getenv("PROVIDER", "openai").strip().lower()
    provider = args.provider or env_provider

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    openai_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()

    # Welcome banner
    print("\n" + "=" * 65)
    print("   AI TEXT SUMMARIZER (CLI) - Structured Outputs Edition")
    print("=" * 65)
    print(f" Active Provider : {provider.upper()}")
    if provider == "openai":
        print(f" Active Model    : {openai_model}")
        print(f" API Key Loaded  : {'Yes' if openai_key and not openai_key.startswith('your_') else 'No (Check .env)'}")
    else:
        print(f" Active Model    : {gemini_model}")
        print(f" API Key Loaded  : {'Yes' if gemini_key and not gemini_key.startswith('your_') else 'No (Check .env)'}")
    print("=" * 65)

    # 1. Obtain input text (from CLI flags or interactive prompt)
    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: Specified file '{args.file}' does not exist.")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text.strip()
    else:
        text = read_input()

    # 2. Obtain style (from CLI flag or interactive prompt)
    style = args.style or choose_style()

    # 3. Perform Summarization
    print(f"\nProcessing text with {provider.upper()} ({style} style)... Please wait...")
    try:
        raw_json_str, result = summarize_text(
            text=text,
            style=style,
            provider=provider,
            openai_key=openai_key,
            gemini_key=gemini_key,
            openai_model=openai_model,
            gemini_model=gemini_model,
        )
        # 4. Display Results (raw JSON first, then formatted)
        display_result(raw_json_str, result, style, provider)

    except (ValueError, ConnectionError, RuntimeError) as err:
        print(f"\n[ERROR] {err}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
