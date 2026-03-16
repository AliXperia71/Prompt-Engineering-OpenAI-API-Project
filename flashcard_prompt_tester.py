"""
Flashcard Prompt Engineer - Brainscape Practice Project
Author: Ali Aamir

This script tests different prompt strategies for converting text into
flashcard Q&A pairs using the OpenAI API. It logs results so you can
compare which prompts produce the best output — exactly what a prompt
engineer does on the job.

Setup:
    pip install openai
    Then set your API key as an environment variable:
    export OPENAI_API_KEY="your-key-here"
"""

from openai import OpenAI
import json
import datetime

client = OpenAI()  # automatically reads OPENAI_API_KEY from environment

# ── Sample input text ────────────────────────────────────────────────────────
# Swap this out for any topic you want to test with!
SAMPLE_TEXT = """
Minecraft is a 3D sandbox video game developed and published by Mojang Studios, 
originally created by Swedish programmer Markus "Notch" Persson in May 2009. It 
was officially released in November 2011 and has since become the best-selling 
video game in history, with over 350 million copies sold. 
The game features a procedurally generated, infinite world made of blocks (voxels), 
allowing players to explore, mine resources, craft tools and items, build structures, 
fight hostile mobs, and cooperate or compete with others in multiplayer.
"""

# ── Prompt versions to test ───────────────────────────────────────────────────
# This is the core of prompt engineering: testing different approaches
# and systematically comparing results.

PROMPT_VERSIONS = {

    "v1_basic": """
Convert the following text into 5 flashcard question-and-answer pairs.

Text: {text}
""",

    "v2_structured": """
You are a flashcard creation assistant. Convert the following text into
exactly 5 flashcard Q&A pairs.

Rules:
- Questions should test a single, specific concept
- Answers should be concise (1-2 sentences max)
- Avoid yes/no questions
- Return as JSON: [{{"Q": "...", "A": "..."}}]

Text: {text}
""",

    "v3_expert_role": """
You are an expert educator specializing in active recall and spaced repetition.
Your task is to create high-quality flashcards that help students truly understand
and retain information — not just memorize facts.

From the text below, generate 5 flashcards. Each should:
- Target a distinct concept
- Have a clear, specific question
- Have an accurate, concise answer (1-2 sentences)
- Be written at a college freshman level

Return ONLY valid JSON in this format:
[{{"Q": "...", "A": "..."}}]

Text: {text}
""",

}


def call_openai(system_prompt: str, user_text: str, model="gpt-4o-mini") -> str:
    """Make a single API call and return the response text."""
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,      # 0 = deterministic, 1 = more creative
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
    )
    return response.choices[0].message.content


def parse_flashcards(raw_output: str) -> list | None:
    """Try to parse JSON flashcards from the model output. Returns None if it fails."""
    try:
        # Strip markdown code fences if present
        cleaned = raw_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None  # v1_basic won't return JSON — that's fine, we'll log raw text


def run_experiment(input_text: str):
    """
    Run all prompt versions on the same input text.
    Log everything so you can compare and document your findings.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = {
        "timestamp": timestamp,
        "input_text": input_text.strip(),
        "results": {}
    }

    print(f"\n{'='*60}")
    print(f"  FLASHCARD PROMPT EXPERIMENT — {timestamp}")
    print(f"{'='*60}\n")

    for version_name, prompt_template in PROMPT_VERSIONS.items():
        print(f"\n--- Testing: {version_name} ---")

        # Fill in the text into the prompt template
        filled_prompt = prompt_template.format(text=input_text)

        # Split into system vs user parts (use the whole thing as system for simplicity)
        raw_output = call_openai(
            system_prompt=filled_prompt,
            user_text="Generate the flashcards now."
        )

        flashcards = parse_flashcards(raw_output)

        log["results"][version_name] = {
            "prompt_used": filled_prompt,
            "raw_output": raw_output,
            "parsed_flashcards": flashcards,
            "parseable_json": flashcards is not None
        }

        # Print results to terminal
        if flashcards:
            for i, card in enumerate(flashcards, 1):
                print(f"  Card {i}:")
                print(f"    Q: {card.get('Q', 'N/A')}")
                print(f"    A: {card.get('A', 'N/A')}")
        else:
            print("  Raw output (no JSON):")
            print(f"  {raw_output[:500]}")  # Print first 500 chars

    # ── Save log to file ──────────────────────────────────────────────────────
    log_filename = f"experiment_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Experiment complete! Results saved to: {log_filename}")
    print(f"{'='*60}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("SUMMARY:")
    for version_name, result in log["results"].items():
        parseable = "✅ Valid JSON" if result["parseable_json"] else "❌ No JSON"
        card_count = len(result["parsed_flashcards"]) if result["parsed_flashcards"] else 0
        print(f"  {version_name}: {parseable} | {card_count} cards parsed")

    print("\nNEXT STEPS:")
    print("  - Open the log JSON and compare card quality across versions")
    print("  - Tweak the prompts in PROMPT_VERSIONS and re-run")
    print("  - Try different input texts (history, science, coding concepts...)")
    print("  - Change temperature (0.0–1.0) and observe how outputs shift\n")


if __name__ == "__main__":
    run_experiment(SAMPLE_TEXT)
