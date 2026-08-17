"""
Content Generator - Generate topic pages and quick references from extracted data
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from extractors.nb2md_loader import find_repo_root
from generators.reference_sync import REFERENCE_NOTES
from jinja2 import Environment, FileSystemLoader

REPO_ROOT = find_repo_root()
DEFAULT_TEMPLATES_DIR = REPO_ROOT / '.knowledge-extraction' / 'templates'
DEFAULT_TAXONOMY_PATH = REPO_ROOT / '.knowledge-extraction' / 'config' / 'taxonomy.yaml'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'knowledge-base' / 'docs'

# Terms too generic to discriminate between topics when scoring relevance.
KEYWORD_STOPWORDS = {
    'and', 'the', 'for', 'with', 'from', 'via', 'use', 'using', 'model',
    'models', 'data', 'code', 'api', 'apis', 'llm', 'llms', 'patterns',
    'pattern', 'formats', 'format', 'strategies', 'strategy', 'systems',
    'system', 'tools', 'tool', 'setup', 'workflows',
}

# How many notebooks and code samples a topic page shows before it stops being
# a reference and starts being a dump.
MAX_SOURCE_NOTEBOOKS = 6
MAX_CODE_SAMPLES = 4
MAX_CONCEPTS_PER_NOTEBOOK = 8
MAX_CODE_LINES = 30


def topic_keywords(topic: dict) -> list[str]:
    """Discriminating terms for a topic, taken from its name and description."""
    text = f"{topic.get('name', '')} {topic.get('description', '')}".lower()
    tokens = re.split(r'[^a-z0-9.+-]+', text)
    return [t for t in tokens if len(t) >= 3 and t not in KEYWORD_STOPWORDS]


def score_notebook(notebook: dict, keywords: list[str]) -> int:
    """How strongly a notebook matches a topic's keywords."""
    haystack = ' '.join([
        notebook['metadata'].get('title', ''),
        ' '.join(notebook.get('key_concepts', [])),
        ' '.join(example.get('code', '') for example in notebook.get('code_examples', [])),
    ]).lower()
    return sum(haystack.count(keyword) for keyword in keywords)


def truncate_code(code: str, max_lines: int = MAX_CODE_LINES) -> str:
    lines = code.rstrip().split('\n')
    if len(lines) <= max_lines:
        return '\n'.join(lines)
    return '\n'.join(lines[:max_lines] + ['# ... truncated'])


def notebook_label(notebook: dict) -> str:
    """Human-readable reference to a notebook, e.g. 'Week 5 Day 3 — RAG Day 3'."""
    meta = notebook['metadata']
    parts = []
    if meta.get('week'):
        parts.append(f"Week {meta['week']}")
    if meta.get('day'):
        parts.append(f"Day {meta['day']}")
    prefix = ' '.join(parts) if parts else meta['filename']
    title = meta.get('title') or meta['filename']
    return f"{prefix} — {title}" if title != prefix else prefix


def relative_topic_path(from_folder: str, to_folder: str, topic_id: str) -> str:
    if from_folder == to_folder:
        return f"{topic_id}.md"
    return f"../{to_folder}/{topic_id}.md"


def reference_title(slug: str) -> str | None:
    """Nav title of a published reference, or None when the slug is unknown."""
    for note in REFERENCE_NOTES:
        if note['slug'] == slug:
            return note['title']
    return None


def reference_callout(slug: str | None) -> str:
    """Admonition pointing a topic page at its authoritative hand-written reference."""
    if not slug:
        return (
            "!!! note \"No primary-source reference yet\"\n\n"
            "    This page is assembled from the course notebooks only. It has not been\n"
            "    cross-checked against primary sources, so treat dated claims with caution."
        )

    title = reference_title(slug) or slug
    # Topic pages live two levels below docs/, references one.
    return (
        f"!!! abstract \"Primary reference: [{title}](../../reference/{slug}.md)\"\n\n"
        "    That page is written from primary sources and supersedes anything below it.\n"
        "    This page maps the concept back to the notebooks that demonstrate it."
    )


class ContentGenerator:
    """Generate markdown content from templates"""

    def __init__(self, templates_dir: str | None = None,
                 output_dir: str | None = None):
        self.templates_dir = Path(templates_dir) if templates_dir else DEFAULT_TEMPLATES_DIR
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def generate_topic_page(self, topic_data: dict, output_path: str) -> str:
        """Generate a topic page from template"""
        template = self.env.get_template('topic_page.md.j2')

        template_data = {
            'title': topic_data.get('title', 'Untitled'),
            'topics': topic_data.get('topics', []),
            'difficulty': topic_data.get('difficulty', 'beginner'),
            'prerequisites': topic_data.get('prerequisites', []),
            'weeks': topic_data.get('weeks', []),
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
            'overview': topic_data.get('overview', ''),
            'core_concepts': topic_data.get('core_concepts', ''),
            'implementation_patterns': topic_data.get('implementation_patterns', ''),
            'common_challenges': topic_data.get('common_challenges', ''),
            'business_applications': topic_data.get('business_applications', ''),
            'troubleshooting': topic_data.get('troubleshooting', ''),
            'related_topics': topic_data.get('related_topics', []),
            'week_references': topic_data.get('week_references', []),
            'resources': topic_data.get('resources', [])
        }

        content = template.render(**template_data)

        full_path = self.output_dir / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')

        return str(full_path)

    def generate_quickref(self, quickref_data: dict, output_path: str) -> str:
        """Generate a quick reference page"""
        template = self.env.get_template('quickref.md.j2')

        template_data = {
            'title': quickref_data.get('title', 'Quick Reference'),
            'topics': quickref_data.get('topics', []),
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
            'five_min_setup': quickref_data.get('five_min_setup', ''),
            'common_tasks': quickref_data.get('common_tasks', []),
            'gotchas': quickref_data.get('gotchas', []),
            'costs': quickref_data.get('costs', ''),
            'main_topic_title': quickref_data.get('main_topic_title', ''),
            'main_topic_path': quickref_data.get('main_topic_path', '')
        }

        content = template.render(**template_data)

        full_path = self.output_dir / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')

        return str(full_path)

    def generate_week_summary(self, week: int, notebooks_data: list[dict]) -> str:
        """Generate a week summary page"""
        output_path = self.output_dir / f'week-summaries/week{week}.md'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stubs = [nb for nb in notebooks_data if nb['metadata']['kind'] == 'stub']
        with_content = [nb for nb in notebooks_data if nb['metadata']['kind'] != 'stub']
        all_concepts = {
            concept
            for nb in with_content
            for concept in nb.get('key_concepts', [])
        }

        lines = [
            f"# Week {week} Summary\n",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*\n",
            "## Overview\n",
            f"{len(notebooks_data)} notebook(s): {len(with_content)} with content in the repo, "
            f"{len(stubs)} Colab stub(s). {len(all_concepts)} distinct concepts extracted.\n",
        ]

        if stubs and not with_content:
            lines.append(
                "!!! warning \"This week is Colab-only\"\n\n"
                "    Every notebook for this week is a link to Google Colab, so there is no\n"
                "    content in the repo to extract. Open each link, save a copy, run it, then\n"
                "    download it back as `<name>_executed.ipynb` to make this week reviewable.\n"
            )

        for nb in notebooks_data:
            meta = nb['metadata']
            day = meta.get('day')
            heading = f"### Day {day}: {meta['title']}" if day else f"### {meta['title']}"
            lines.append(f"{heading}\n")
            lines.append(f"`{meta['path']}`\n")

            if meta['kind'] == 'stub':
                lines.append("*Colab stub — content not in the repo.*\n")
                for link in meta.get('colab_links', []):
                    lines.append(f"- Colab: <{link}>")
                lines.append("")
                continue

            concepts = nb.get('key_concepts', [])
            if concepts:
                lines.append("**Key Concepts:**\n")
                for concept in concepts[:MAX_CONCEPTS_PER_NOTEBOOK]:
                    lines.append(f"- {concept}")
                lines.append("")

            biz_context = nb.get('business_context', [])
            if biz_context:
                lines.append("**Business Applications:**\n")
                lines.append(f"{biz_context[0][:300]}\n")

            if meta['cells_with_output']:
                lines.append(
                    f"*{meta['cells_with_output']} cell(s) carry saved output — "
                    "this notebook is reviewable without re-running.*\n"
                )

        content = '\n'.join(lines)
        output_path.write_text(content, encoding='utf-8')

        return str(output_path)

    def build_topic_page_data(self, topic: dict, notebook_data: dict[int, list[dict]],
                              taxonomy: dict,
                              summarizer: Any | None = None) -> dict[str, Any]:
        """Assemble one topic page's content from the notebooks of its weeks."""
        weeks = topic.get('weeks', [])
        keywords = topic_keywords(topic)

        candidates = [
            nb
            for week in weeks
            for nb in notebook_data.get(week, [])
            if nb['metadata']['kind'] != 'stub'
        ]
        stub_count = sum(
            1
            for week in weeks
            for nb in notebook_data.get(week, [])
            if nb['metadata']['kind'] == 'stub'
        )

        ranked = sorted(candidates, key=lambda nb: score_notebook(nb, keywords), reverse=True)
        sources = [nb for nb in ranked if score_notebook(nb, keywords) > 0][:MAX_SOURCE_NOTEBOOKS]
        if not sources:
            sources = ranked[:MAX_SOURCE_NOTEBOOKS]

        overview = self._topic_overview(topic, sources, candidates, stub_count, weeks, summarizer)
        core_concepts = self._topic_concepts(sources)
        implementation = self._topic_code(sources, keywords)
        business = self._topic_business(sources)

        related = []
        for other in taxonomy.get('topics', []):
            if other['id'] == topic['id']:
                continue
            if set(other.get('weeks', [])) & set(weeks):
                related.append({
                    'name': other['name'],
                    'path': relative_topic_path(topic['folder'], other['folder'], other['id']),
                })

        week_references = [
            f"[Week {week}](../../week-summaries/week{week}.md)" for week in weeks
        ]

        return {
            'title': topic['name'],
            'topics': [topic['id']],
            'difficulty': topic.get('difficulty', 'intermediate'),
            'weeks': weeks,
            'overview': overview,
            'core_concepts': core_concepts,
            'implementation_patterns': implementation,
            'business_applications': business,
            'related_topics': related[:8],
            'week_references': week_references,
        }

    def _topic_overview(self, topic: dict, sources: list[dict], candidates: list[dict],
                        stub_count: int, weeks: list[int],
                        summarizer: Any | None) -> str:
        week_list = ', '.join(str(w) for w in weeks) or 'n/a'
        lines = [
            reference_callout(topic.get('reference')),
            "",
            topic.get('description', ''),
            "",
        ]

        if summarizer and sources:
            sections = []
            for nb in sources:
                concepts = '\n'.join(f"- {c}" for c in nb.get('key_concepts', []))
                sections.append(f"{notebook_label(nb)}\n{concepts}")
            try:
                result = summarizer.generate_topic_summary(sections, topic['name'])
                lines.append(result['summary'].strip())
                lines.append("")
            except Exception as exc:  # noqa: BLE001 - a failed page must not stop the run
                print(f"  ! LLM synthesis failed for {topic['id']}: {exc}")

        lines.append(
            f"Covered in week(s) {week_list}: {len(candidates)} notebook(s) with content "
            f"in the repo" + (f", {stub_count} Colab stub(s)." if stub_count else ".")
        )

        if not candidates:
            lines.append("")
            lines.append(
                "!!! warning \"No local source material\"\n\n"
                "    Every notebook for this topic's weeks is a Colab stub. Download and run\n"
                "    them before this page can say anything about the topic."
            )

        return '\n'.join(lines).strip()

    def _topic_concepts(self, sources: list[dict]) -> str:
        if not sources:
            return "*No concepts extracted — the source notebooks are Colab stubs.*"

        lines = []
        for nb in sources:
            concepts = nb.get('key_concepts', [])
            if not concepts:
                continue
            lines.append(f"### {notebook_label(nb)}\n")
            for concept in concepts[:MAX_CONCEPTS_PER_NOTEBOOK]:
                lines.append(f"- {concept}")
            lines.append("")

        return '\n'.join(lines).strip() or "*No headings in the source notebooks named a concept.*"

    def _topic_code(self, sources: list[dict], keywords: list[str]) -> str:
        scored = []
        for nb in sources:
            for example in nb.get('code_examples', []):
                code = example.get('code', '')
                hits = sum(code.lower().count(keyword) for keyword in keywords)
                if hits:
                    scored.append((hits, nb, example))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)

        lines = []
        for _, nb, example in scored[:MAX_CODE_SAMPLES]:
            lang = example.get('language', 'python')
            lines.append(f"**{example['type']}** — {notebook_label(nb)}\n")
            lines.append(f"```{lang}")
            lines.append(truncate_code(example['code']))
            lines.append("```")
            if example.get('output'):
                lines.append("\n> Output\n")
                lines.append("```text")
                lines.append(example['output'])
                lines.append("```")
            lines.append("")

        return '\n'.join(lines).strip()

    def _topic_business(self, sources: list[dict]) -> str:
        entries = []
        seen = set()
        for nb in sources:
            for text in nb.get('business_context', []):
                key = text[:120].lower()
                if key in seen:
                    continue
                seen.add(key)
                entries.append(f"- **{notebook_label(nb)}**: {text[:400]}")

        return '\n'.join(entries[:6])

    def generate_index_page(self, taxonomy: dict) -> str:
        """Generate the main index page"""
        output_path = self.output_dir / 'index.md'

        lines = [
            "# LLM Engineering Knowledge Base\n",
            "*Personal study notes from Ed Donner's LLM Engineering course*\n",
            "## Legal Disclaimer\n",
            "This knowledge base contains personal study notes from the LLM Engineering course by Ed Donner. ",
            "All content has been paraphrased and reorganized for personal reference.\n",
            "**Original course**: [https://edwarddonner.com/](https://edwarddonner.com/)  ",
            "**Udemy course**: [LLM Engineering Course](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/)\n",
            "No course materials are redistributed. All code examples are original implementations based on concepts learned.\n",
            "For the actual course content, please enroll in the course.\n",
            "---\n",
            "## Navigation\n",
            "### Topic References\n",
            "Written from primary sources rather than lecture transcripts. Start here — "
            "these supersede the generated pages below.\n",
        ]

        for note in REFERENCE_NOTES:
            lines.append(f"- [{note['title']}](reference/{note['slug']}.md)")

        lines.append("")
        lines.append("### By Topic\n")

        if 'categories' in taxonomy:
            for cat_id, cat_info in sorted(taxonomy['categories'].items(),
                                          key=lambda x: x[1].get('order', 99)):
                lines.append(f"#### {cat_info['name']}\n")

                for topic in taxonomy.get('topics', []):
                    if topic.get('category') == cat_id:
                        folder = topic.get('folder', '')
                        topic_id = topic.get('id', '')
                        topic_name = topic.get('name', '')
                        lines.append(f"- [{topic_name}](topics/{folder}/{topic_id}.md)")

                lines.append("")

        lines.extend([
            "### Quick References\n",
            "- [API Syntax](quick-ref/api-syntax.md)",
            "- [Common Errors](quick-ref/common-errors.md)",
            "- [Code Snippets](quick-ref/code-snippets.md)\n",
            "### By Week\n"
        ])

        for week in range(1, 9):
            lines.append(f"- [Week {week} Summary](week-summaries/week{week}.md)")

        lines.extend([
            "\n### Projects\n",
            "- [Website Summarizer](projects/website-summarizer.md) - Week 1",
            "- [Multi-Model Integration](projects/multi-model.md) - Week 2",
            "- [RAG System](projects/rag-system.md) - Week 5",
            "- [Price Is Right](projects/price-is-right.md) - Weeks 6-7",
            "- [Autonomous Agents](projects/autonomous-agents.md) - Week 8\n",
            "## Search\n",
            "Use the search bar above to find topics, concepts, or code examples.\n",
            "## About This Knowledge Base\n",
            "This knowledge base is automatically generated from course notebooks and supplemented with manual notes. ",
            "It's organized by topic rather than chronologically, making it easier to review concepts and find information.\n",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*\n"
        ])

        content = '\n'.join(lines)
        output_path.write_text(content, encoding='utf-8')

        return str(output_path)


def load_taxonomy(taxonomy_path: str | None = None) -> dict:
    path = Path(taxonomy_path) if taxonomy_path else DEFAULT_TAXONOMY_PATH
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_topic_pages(notebook_data: dict[int, list[dict]],
                         taxonomy_path: str | None = None,
                         output_dir: str | None = None,
                         summarizer: Any | None = None) -> list[str]:
    """Write every topic page in the taxonomy from extracted notebook content."""
    taxonomy = load_taxonomy(taxonomy_path)
    generator = ContentGenerator(output_dir=output_dir)

    written = []
    for topic in taxonomy.get('topics', []):
        page_data = generator.build_topic_page_data(topic, notebook_data, taxonomy, summarizer)
        output_path = f"topics/{topic['folder']}/{topic['id']}.md"
        written.append(generator.generate_topic_page(page_data, output_path))

    return written


def generate_placeholder_topics(taxonomy_path: str | None = None,
                                output_dir: str | None = None):
    """Write empty topic pages for every taxonomy entry, for scaffolding a new taxonomy."""
    taxonomy = load_taxonomy(taxonomy_path)
    generator = ContentGenerator(output_dir=output_dir)

    for topic in taxonomy.get('topics', []):
        placeholder_data = {
            'title': topic.get('name'),
            'topics': [topic.get('id')],
            'difficulty': 'intermediate',
            'weeks': topic.get('weeks', []),
            'overview': f"{topic.get('name')}\n\n{topic.get('description')}",
            'core_concepts': "*Not yet extracted — run the pipeline without --scaffold.*",
            'week_references': [f"Week {w}" for w in topic.get('weeks', [])],
            'related_topics': []
        }
        generator.generate_topic_page(placeholder_data, f"topics/{topic['folder']}/{topic['id']}.md")
