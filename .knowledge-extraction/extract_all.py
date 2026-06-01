#!/usr/bin/env python3
"""
Main extraction pipeline - Extract knowledge from course notebooks and website
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from extractors.notebook_extractor import extract_all_notebooks, extract_notebooks_from_week
from extractors.web_scraper import scrape_website
from processors.llm_summarizer import LLMSummarizer, summarize_notebook
from generators.content_generator import ContentGenerator, generate_placeholder_topics


def extract_notebooks_phase(weeks=range(1, 9)):
    """Phase 1: Extract all notebooks"""
    print("\n=== Phase 1: Extracting Notebooks ===")
    notebook_data = extract_all_notebooks(weeks=weeks)

    total_notebooks = sum(len(week_data) for week_data in notebook_data.values())
    print(f"\n✓ Extracted {total_notebooks} notebooks from {len(notebook_data)} weeks")

    return notebook_data


def extract_website_phase():
    """Phase 2: Extract website content"""
    print("\n=== Phase 2: Extracting Website Content ===")

    try:
        website_data = scrape_website()
        print("✓ Website extraction complete")
        return website_data
    except Exception as e:
        print(f"⚠ Website extraction failed: {e}")
        print("  Continuing without website data...")
        return {}


def summarize_phase(notebook_data, use_llm=False, provider='openai', model=None):
    """Phase 3: LLM-assisted summarization (optional)"""
    if not use_llm:
        print("\n=== Phase 3: Skipping LLM Summarization ===")
        print("  (Run with --use-llm to enable)")
        return notebook_data

    print("\n=== Phase 3: LLM Summarization ===")

    if provider == 'ollama':
        print("  Using Ollama (local model) - FREE!")
        print(f"  Model: {model or 'llama3.2'}")
        print("  Make sure Ollama is running: ollama serve")
        cost_info = "FREE (local model)"
    else:
        print(f"  Using {provider.upper()} API")
        print(f"  Model: {model or ('gpt-4.1-mini' if provider == 'openai' else 'claude-3-5-sonnet')}")
        print("  This may incur API costs.")

    print(f"  Processing {sum(len(w) for w in notebook_data.values())} notebooks...")

    summarizer = LLMSummarizer(provider=provider, model=model)
    summarized_data = {}

    total_tokens = 0

    for week, notebooks in notebook_data.items():
        summarized_data[week] = []
        for nb_data in notebooks:
            summary = summarize_notebook(nb_data, summarizer)
            summarized_data[week].append(summary)
            total_tokens += summary.get('token_count', 0)
            print(f"  [OK] Summarized {nb_data['metadata']['filename']}")

    print(f"\nSummarization complete")
    print(f"  Total tokens: {total_tokens:,}")

    if provider == 'ollama':
        print(f"  Cost: FREE (local model)")
    else:
        estimated_cost = (total_tokens / 1_000_000) * (0.15 if provider == 'openai' else 3.0)
        print(f"  Estimated cost: ${estimated_cost:.2f}")

    return summarized_data


def generate_content_phase(notebook_data, website_data):
    """Phase 4: Generate content"""
    print("\n=== Phase 4: Generating Content ===")

    generator = ContentGenerator()

    # Generate week summaries
    print("\nGenerating week summaries...")
    for week, notebooks in notebook_data.items():
        output = generator.generate_week_summary(week, notebooks)
        print(f"  ✓ Week {week}: {output}")

    # Generate placeholder topic pages (will be filled in manually or with LLM later)
    print("\nGenerating topic pages...")
    generate_placeholder_topics()

    print("\n✓ Content generation complete")


def main():
    parser = argparse.ArgumentParser(description='Extract knowledge from LLM Engineering course')
    parser.add_argument('--weeks', type=str, help='Comma-separated week numbers (default: all)')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for summarization')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'anthropic', 'ollama'],
                       help='LLM provider: openai (paid), anthropic (paid), or ollama (FREE, local)')
    parser.add_argument('--model', type=str, help='Model name (e.g., llama3.2, qwen2.5, mistral)')
    parser.add_argument('--skip-website', action='store_true', help='Skip website scraping')
    parser.add_argument('--generate-only', action='store_true', help='Only generate content from existing data')

    args = parser.parse_args()

    # Parse weeks
    if args.weeks:
        weeks = [int(w.strip()) for w in args.weeks.split(',')]
    else:
        weeks = range(1, 9)

    print("=" * 62)
    print("  LLM Engineering Course - Knowledge Extraction Pipeline")
    print("=" * 62)

    if args.generate_only:
        print("\nGenerating content from existing data...")
        generate_placeholder_topics()
        print("\nDone!")
        return

    # Run pipeline
    notebook_data = extract_notebooks_phase(weeks=weeks)

    if not args.skip_website:
        website_data = extract_website_phase()
    else:
        print("\n=== Phase 2: Skipping Website Extraction ===")
        website_data = {}

    notebook_data = summarize_phase(notebook_data, use_llm=args.use_llm,
                                   provider=args.provider, model=args.model)

    generate_content_phase(notebook_data, website_data)

    print("\n" + "=" * 62)
    print("            Extraction Complete!")
    print("=" * 62)
    print("\nNext steps:")
    print("  1. cd knowledge-base")
    print("  2. python -m mkdocs serve")
    print("  3. Open http://localhost:8000")
    print("\nTo enable LLM summarization:")
    print("  FREE (local):  python extract_all.py --use-llm --provider ollama")
    print("  OpenAI (~$5): python extract_all.py --use-llm --provider openai")
    print("  Anthropic:    python extract_all.py --use-llm --provider anthropic")


if __name__ == '__main__':
    main()
