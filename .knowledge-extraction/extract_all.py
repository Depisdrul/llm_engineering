#!/usr/bin/env python3
"""
Main extraction pipeline - Extract knowledge from course notebooks and website.

Run from anywhere; all paths resolve against the repo root.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from extractors.nb2md_loader import find_repo_root
from extractors.notebook_extractor import extract_all_notebooks
from extractors.web_scraper import scrape_website
from generators.content_generator import (
    ContentGenerator,
    generate_topic_pages,
    load_taxonomy,
)
from generators.reference_sync import generate_reference_index, sync_references
from processors.llm_summarizer import LLMSummarizer

REPO_ROOT = find_repo_root()
CACHE_PATH = REPO_ROOT / '.knowledge-extraction' / '.cache' / 'extracted.json'


def load_cache() -> dict:
    """Load previously extracted notebook data, keyed by int week number."""
    if not CACHE_PATH.is_file():
        return {}
    raw = json.loads(CACHE_PATH.read_text(encoding='utf-8'))
    return {int(week): notebooks for week, notebooks in raw.items()}


def save_cache(notebook_data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(notebook_data, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def extract_notebooks_phase(weeks=range(1, 9)):
    """Phase 1: Extract all notebooks"""
    print("\n=== Phase 1: Extracting Notebooks ===")
    notebook_data = extract_all_notebooks(weeks=weeks)

    total_notebooks = sum(len(week_data) for week_data in notebook_data.values())
    stubs = sum(
        1
        for week_data in notebook_data.values()
        for nb in week_data
        if nb['metadata']['kind'] == 'stub'
    )
    print(f"\n[OK] Extracted {total_notebooks} notebooks from {len(notebook_data)} weeks "
          f"({stubs} Colab stubs with no local content)")

    return notebook_data


def extract_website_phase(strict=False):
    """Phase 2: Extract website content"""
    print("\n=== Phase 2: Extracting Website Content ===")

    try:
        website_data = scrape_website()
        print("[OK] Website extraction complete")
        return website_data
    except Exception as exc:
        if strict:
            raise
        print(f"[WARN] Website extraction failed: {exc}")
        print("       Continuing without website data. Re-run with --strict-website to fail here.")
        return {}


def summarize_phase(notebook_data, use_llm=False, provider='openai', model=None):
    """Phase 3: LLM-assisted per-notebook summarisation (optional).

    Returns (notebook_data, summarizer). The summarizer is reused for topic-page
    synthesis so a single client and model choice covers the whole run.
    """
    if not use_llm:
        print("\n=== Phase 3: Skipping LLM Summarization ===")
        print("  (Run with --use-llm to enable)")
        return notebook_data, None

    print("\n=== Phase 3: LLM Summarization ===")

    if provider == 'ollama':
        print("  Using Ollama (local model) - FREE")
        print(f"  Model: {model or 'llama3.2'}")
        print("  Make sure Ollama is running: ollama serve")
    else:
        print(f"  Using {provider.upper()} API")
        print(f"  Model: {model or ('gpt-4.1-mini' if provider == 'openai' else 'claude-3-5-sonnet')}")
        print("  This may incur API costs.")

    summarizer = LLMSummarizer(provider=provider, model=model)
    total_tokens = 0

    for notebooks in notebook_data.values():
        for nb_data in notebooks:
            if nb_data['metadata']['kind'] == 'stub':
                continue
            try:
                result = summarizer.extract_key_concepts(nb_data)
            except Exception as exc:
                print(f"  [WARN] {nb_data['metadata']['filename']}: {exc}")
                continue
            nb_data['llm_summary'] = result['summary']
            total_tokens += result.get('token_count', 0)
            print(f"  [OK] Summarized {nb_data['metadata']['filename']}")

    print("\nSummarization complete")
    print(f"  Total tokens: {total_tokens:,}")

    if provider == 'ollama':
        print("  Cost: FREE (local model)")
    else:
        estimated_cost = (total_tokens / 1_000_000) * (0.15 if provider == 'openai' else 3.0)
        print(f"  Estimated cost: ${estimated_cost:.2f}")

    return notebook_data, summarizer


def sync_references_phase():
    """Phase 4: Publish the hand-written topic references into the docs tree."""
    print("\n=== Phase 4: Publishing Topic References ===")
    written = sync_references()
    generate_reference_index()
    for path in written:
        print(f"  [OK] {Path(path).name}")
    print(f"\n[OK] Published {len(written)} reference note(s)")
    return written


def generate_content_phase(notebook_data, summarizer=None):
    """Phase 5: Generate week summaries, topic pages and the index."""
    print("\n=== Phase 5: Generating Content ===")

    generator = ContentGenerator()
    taxonomy = load_taxonomy()

    print("\nGenerating week summaries...")
    for week, notebooks in sorted(notebook_data.items()):
        output = generator.generate_week_summary(week, notebooks)
        print(f"  [OK] Week {week}: {Path(output).name}")

    print("\nGenerating topic pages...")
    if summarizer:
        print("  (with LLM synthesis - one call per topic)")
    written = generate_topic_pages(notebook_data, summarizer=summarizer)
    for path in written:
        print(f"  [OK] {Path(path).parent.name}/{Path(path).name}")

    generator.generate_index_page(taxonomy)
    print("\n[OK] Content generation complete")


def main():
    parser = argparse.ArgumentParser(description='Extract knowledge from LLM Engineering course')
    parser.add_argument('--weeks', type=str, help='Comma-separated week numbers (default: all)')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for summarization and topic synthesis')
    parser.add_argument('--provider', type=str, default='openai',
                        choices=['openai', 'anthropic', 'ollama'],
                        help='LLM provider: openai (paid), anthropic (paid), or ollama (FREE, local)')
    parser.add_argument('--model', type=str, help='Model name (e.g., llama3.2, qwen2.5, mistral)')
    parser.add_argument('--skip-website', action='store_true', help='Skip website scraping')
    parser.add_argument('--strict-website', action='store_true',
                        help='Fail the run if website scraping fails, instead of warning')
    parser.add_argument('--generate-only', action='store_true',
                        help='Regenerate pages from the cached extraction, without re-reading notebooks')

    args = parser.parse_args()

    weeks = [int(w.strip()) for w in args.weeks.split(',')] if args.weeks else range(1, 9)

    print("=" * 62)
    print("  LLM Engineering Course - Knowledge Extraction Pipeline")
    print("=" * 62)
    print(f"  Repo root: {REPO_ROOT}")

    if args.generate_only:
        notebook_data = load_cache()
        if not notebook_data:
            print(f"\nNo cached extraction at {CACHE_PATH}. Run without --generate-only first.")
            return 1
        print(f"\nUsing cached extraction: {sum(len(v) for v in notebook_data.values())} notebooks")
        sync_references_phase()
        generate_content_phase(notebook_data)
    else:
        notebook_data = extract_notebooks_phase(weeks=weeks)

        if args.skip_website:
            print("\n=== Phase 2: Skipping Website Extraction ===")
        else:
            extract_website_phase(strict=args.strict_website)

        notebook_data, summarizer = summarize_phase(
            notebook_data, use_llm=args.use_llm, provider=args.provider, model=args.model
        )

        save_cache(notebook_data)
        print(f"\nCached extraction: {CACHE_PATH}")

        sync_references_phase()
        generate_content_phase(notebook_data, summarizer=summarizer)

    print("\n" + "=" * 62)
    print("            Extraction Complete!")
    print("=" * 62)
    print("\nNext steps:")
    print("  cd knowledge-base && python -m mkdocs serve")
    print("  Open http://localhost:8000")
    print("\nTo enable LLM synthesis of topic pages:")
    print("  FREE (local):  python extract_all.py --use-llm --provider ollama")
    print("  OpenAI:        python extract_all.py --use-llm --provider openai")
    print("  Anthropic:     python extract_all.py --use-llm --provider anthropic")
    return 0


if __name__ == '__main__':
    sys.exit(main())
