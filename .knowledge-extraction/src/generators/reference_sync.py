"""
Reference Sync - publish the hand-written topic references into the MkDocs site.

The references in `.knowledge-extraction/improvements-from-chrome-session/` are
written from primary sources and are the authoritative layer of the knowledge
base; the generated topic pages are the provenance layer that points back at the
notebooks. This module copies the references into `knowledge-base/docs/reference/`
so MkDocs serves and indexes them.

The copies are build output. Edit the source files.
"""
import re
from pathlib import Path
from typing import Any

from extractors.nb2md_loader import find_repo_root

REPO_ROOT = find_repo_root()
DEFAULT_SOURCE_DIR = REPO_ROOT / '.knowledge-extraction' / 'improvements-from-chrome-session'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'knowledge-base' / 'docs'
REFERENCE_SUBDIR = 'reference'

SYNC_BANNER = (
    "<!-- Generated copy. Source: .knowledge-extraction/improvements-from-chrome-session/{source}\n"
    "     Edit the source file; this copy is overwritten on every extraction run. -->\n\n"
)

# Source file -> published slug and nav title. Only files listed here are
# published, so drafts can sit in the source folder without going live.
#
# `aliases` are the filenames the notes use when they cross-link each other.
# The notes were written against a numbered naming convention that the files on
# disk do not use, so every alias is rewritten to the published slug on sync.
REFERENCE_NOTES: list[dict[str, Any]] = [
    {
        'source': '01llmfoundations.md',
        'slug': 'llm-foundations',
        'title': 'LLM Foundations (Weeks 1-3)',
        'aliases': ['01-llm-foundations', '01llmfoundations'],
    },
    {
        'source': '05ragandvectorsearch.md',
        'slug': 'rag-and-vector-search',
        'title': 'RAG & Vector Search (Week 5)',
        'aliases': ['05-rag-and-vector-search', '05ragandvectorsearch'],
    },
    {
        'source': '06trainingandfinetuning.md',
        'slug': 'training-and-finetuning',
        'title': 'Training & Fine-Tuning (Weeks 6-7)',
        'aliases': ['06-training-and-finetuning', '06trainingandfinetuning'],
    },
    {
        'source': '07agentsanddeployment.md',
        'slug': 'agents-and-deployment',
        'title': 'Agents & Deployment (Week 8)',
        'aliases': ['07-agents-and-deployment', '07agentsanddeployment'],
    },
    {
        'source': '08apikeysandrunnability.md',
        'slug': 'api-keys-and-runnability',
        'title': 'API Keys & Runnability',
        'aliases': ['08-api-keys-and-runnability', '08apikeysandrunnability'],
    },
    {
        'source': 'LECTUREDISTILL.md',
        'slug': 'lecture-distillation-workflow',
        'title': 'Lecture Distillation Workflow',
        'aliases': ['LECTURE-DISTILL', 'LECTUREDISTILL'],
    },
]

# Matches a relative markdown link target, e.g. "](./01-llm-foundations.md)".
RELATIVE_MD_LINK_RE = re.compile(r'\]\((\./)?([\w.-]+)\.md((?:#[\w-]+)?)\)')


def build_alias_map() -> dict[str, str]:
    """Alias filename (without extension) -> published slug."""
    alias_map = {}
    for note in REFERENCE_NOTES:
        alias_map[note['slug']] = note['slug']
        alias_map[Path(note['source']).stem] = note['slug']
        for alias in note.get('aliases', []):
            alias_map[alias] = note['slug']
    return alias_map


def rewrite_cross_links(body: str, alias_map: dict[str, str] | None = None) -> str:
    """Point the notes' cross-links at published slugs, leaving unknown links alone."""
    alias_map = alias_map if alias_map is not None else build_alias_map()

    def replace(match: re.Match) -> str:
        target, anchor = match.group(2), match.group(3)
        slug = alias_map.get(target)
        if slug is None:
            return match.group(0)
        return f"]({slug}.md{anchor})"

    return RELATIVE_MD_LINK_RE.sub(replace, body)


def reference_path(slug: str) -> str:
    """Site-relative path of a published reference, for linking from topic pages."""
    return f"{REFERENCE_SUBDIR}/{slug}.md"


def sync_references(source_dir: Path | None = None,
                    output_dir: Path | None = None) -> list[str]:
    """Copy every listed reference note into the docs tree. Returns written paths."""
    source_dir = Path(source_dir) if source_dir else DEFAULT_SOURCE_DIR
    output_root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    target_dir = output_root / REFERENCE_SUBDIR

    if not source_dir.is_dir():
        print(f"  ! reference source folder not found: {source_dir}")
        return []

    target_dir.mkdir(parents=True, exist_ok=True)
    alias_map = build_alias_map()
    written = []

    for note in REFERENCE_NOTES:
        source_path = source_dir / note['source']
        if not source_path.is_file():
            print(f"  ! missing reference note, skipped: {note['source']}")
            continue

        body = rewrite_cross_links(source_path.read_text(encoding='utf-8'), alias_map)
        target_path = target_dir / f"{note['slug']}.md"
        target_path.write_text(
            SYNC_BANNER.format(source=note['source']) + body,
            encoding='utf-8',
        )
        written.append(str(target_path))

    return written


def generate_reference_index(output_dir: Path | None = None) -> str:
    """Write the landing page that lists every published reference."""
    output_root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    target_dir = output_root / REFERENCE_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Topic References",
        "",
        "Hand-written references built from primary sources — arXiv, ACL Anthology, and",
        "official provider documentation — rather than from lecture transcripts. Where a",
        "source contradicts what the course teaches, the contradiction is flagged inline.",
        "",
        "These are the authoritative layer of this knowledge base. The generated topic",
        "pages under **Topics** are the provenance layer: they map concepts back to the",
        "notebooks that demonstrate them.",
        "",
    ]

    for note in REFERENCE_NOTES:
        lines.append(f"- [{note['title']}]({note['slug']}.md)")

    lines.append("")

    index_path = target_dir / 'index.md'
    index_path.write_text('\n'.join(lines), encoding='utf-8')
    return str(index_path)
