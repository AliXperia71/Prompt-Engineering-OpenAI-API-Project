# Flashcard Prompt Engineering — OpenAI API

A Python tool for systematically testing and comparing LLM prompt strategies for flashcard generation. Built as a hands-on prompt engineering project using the OpenAI API.

## What it does

- Sends the same input text through multiple prompt versions simultaneously
- Compares output quality across different prompt strategies (basic, structured, expert-role)
- Automatically logs every experiment to a timestamped JSON file for analysis and documentation
- Surfaces real prompt engineering issues like hallucination, inconsistent JSON formatting, and output reliability

## Key findings

- Vague prompts (v1) produced unreliable output and failed to return structured JSON
- Adding explicit formatting rules (v2) dramatically improved consistency and parseability
- Role-based prompting (v3) improved answer depth but introduced hallucinated content not present in the source text
- Lower temperature values (0.0–0.2) reduced hallucination while maintaining accuracy

## Project structure

```
flashcard_prompt_tester.py   # Main script — prompt versions, API calls, logging
experiment_log_[timestamp].json  # Auto-generated experiment logs
notes.md                     # Observations and iteration notes
```

## Setup

```bash
pip install openai
export OPENAI_API_KEY="your-key-here"
python flashcard_prompt_tester.py
```

## Tech

- Python 3.13
- OpenAI API (gpt-4o-mini)
- JSON logging for experiment tracking
