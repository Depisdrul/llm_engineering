"""
Notebook Extractor - Parse Jupyter notebooks and extract structured content.

Notebook parsing, classification and cell access are delegated to nb2md
(`study/nb2md.py`), which is the repo's single notebook parser. This module
adds the knowledge-base-specific layer on top: heading filtering,
code-pattern classification, business/exercise sections and output capture.
"""
import json
import re
from pathlib import Path
from typing import Any

from .nb2md_loader import find_repo_root, load_nb2md

nb2md = load_nb2md()
REPO_ROOT = find_repo_root()


def repo_relative(path: Path) -> str:
    """Repo-relative POSIX path, so generated docs never leak a local filesystem layout."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()

# Notebooks that are course material rather than helper modules or scratch work.
NOTEBOOK_GLOB = '*.ipynb'
SKIP_PATH_PARTS = ('community-contributions', 'community_contributions', '.ipynb_checkpoints')

# Headings that narrate the lecture rather than name a concept. The course
# notebooks are conversational, so an unfiltered heading scrape yields lines
# like "Donezo! On to Step 2" as if they were learning objectives.
NARRATION_PREFIXES = (
    'lets', "let's", 'let us', 'well', 'now', 'and now', 'finally', 'first',
    'but first', 'so', 'ok', 'okay', 'alright', 'donezo', 'time to', 'on to',
    'next', 'before we', 'hooray', 'wow', 'oh', 'and', 'but', 'here we go',
    'welcome', 'congratulations', 'phew', 'thats', "that's", 'this is it',
    'admit it', 'what could', 'did you', 'shall we', 'ready', 'how cool',
    'guess what', 'i hope', 'i promise', 'believe it', 'surely', 'remember',
)
NARRATION_SUBSTRINGS = (
    "let's", 'lets go', "we'll", "you'll", "here's", 'that was', 'as promised',
    'as always', 'exciting part', 'have a go', 'try it', 'over to you',
    'you thought', 'more complicated than', 'possibly come next',
)
# Emoji and pictographs mark a heading as an aside rather than a concept name.
EMOJI_RE = re.compile(
    '[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF⬀-⯿️]'
)
STEP_MARKER_RE = re.compile(r'^(step|part|stage)\s*\d', re.IGNORECASE)
MARKDOWN_NOISE_RE = re.compile(r'[*_`]+')
LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]*\)')
HTML_TAG_RE = re.compile(r'<[^>]+>')
TABLE_MARKER = '<table'

MAX_CONCEPT_LENGTH = 100
MAX_OUTPUT_CHARS = 800


def clean_heading(raw: str) -> str:
    """Strip markdown decoration and trailing punctuation from a heading."""
    text = LINK_RE.sub(r'\1', raw)
    text = MARKDOWN_NOISE_RE.sub('', text)
    text = HTML_TAG_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.rstrip(' :;,.-')


def callout_text(source: str) -> str:
    """Plain text of a markdown cell's HTML callout box, excluding prose around it.

    The course puts business notes and warnings in `<table>` boxes. Only the box
    is wanted; the surrounding lecture prose is a separate section.
    """
    start = source.find(TABLE_MARKER)
    body = source[start:] if start != -1 else source
    text = HTML_TAG_RE.sub(' ', body)
    return re.sub(r'\s+', ' ', text).strip()


def starts_with_narration(text: str) -> bool:
    """True when a heading opens with a conversational lead-in."""
    # Compare against a punctuation-free form so "Let's" and "Lets" both match.
    normalized = re.sub(r"[^a-z0-9\s']", ' ', text.lower()).strip()
    return any(
        normalized == prefix or normalized.startswith(prefix + ' ')
        for prefix in NARRATION_PREFIXES
    )


def is_section_marker(heading: str, text: str) -> bool:
    """True for short trailing-colon markers such as "TODAY:" or "Three steps:"."""
    return heading.strip().endswith(':') and len(text.split()) <= 3


def is_narration(heading: str) -> bool:
    """True when a heading reads as lecture narration rather than a concept."""
    text = clean_heading(heading)
    if not text or len(text) > MAX_CONCEPT_LENGTH:
        return True
    if not re.search(r'[a-zA-Z]', text) or EMOJI_RE.search(text):
        return True
    if starts_with_narration(text):
        return True
    if any(fragment in text.lower() for fragment in NARRATION_SUBSTRINGS):
        return True
    return bool(STEP_MARKER_RE.match(text)) or is_section_marker(heading, text)


class NotebookSection:
    """A single markdown region of a notebook, tagged with its role."""

    def __init__(self, section_type: str, content: str, metadata: dict | None = None):
        self.type = section_type
        self.content = content
        self.metadata = metadata or {}


class NotebookExtractor:
    """Extract structured content from a Jupyter notebook."""

    def __init__(self, notebook_path: str):
        self.path = Path(notebook_path)
        notebook = nb2md.load_notebook(self.path)
        if notebook is None:
            raise ValueError(f"{self.path} is not a readable notebook")
        self.notebook = notebook
        self.kind, self.stats = nb2md.classify(notebook)
        self.language = nb2md.cell_language(notebook)

    @property
    def cells(self) -> list[dict]:
        return self.notebook.get('cells', [])

    def cell_source(self, cell: dict) -> str:
        return nb2md.source_text(cell)

    @property
    def is_stub(self) -> bool:
        """True for Colab-link-only notebooks, whose content is not in the repo."""
        return self.kind == 'stub'

    def extract_metadata(self) -> dict[str, Any]:
        """Metadata from the notebook path, its first heading and nb2md's classification."""
        week_match = None
        day_match = None
        for part in self.path.parts:
            if part.startswith('week'):
                week_match = re.search(r'week(\d+)', part)
            if 'day' in part:
                day_match = re.search(r'day(\d+)', part)

        title = None
        for cell in self.cells:
            if cell.get('cell_type') != 'markdown':
                continue
            for line in self.cell_source(cell).split('\n'):
                if line.startswith('# ') and not line.startswith('## '):
                    title = clean_heading(line.lstrip('#'))
                    break
            if title:
                break

        return {
            'week': int(week_match.group(1)) if week_match else None,
            'day': int(day_match.group(1)) if day_match else None,
            'title': title or self.path.stem,
            'filename': self.path.name,
            'path': repo_relative(self.path),
            'kind': self.kind,
            'language': self.language,
            'cells': self.stats['cells'],
            'cells_with_output': self.stats['with_output'],
            'colab_links': list(dict.fromkeys(self.stats['colab_links'])),
        }

    def classify_markdown_cell(self, content: str) -> str:
        """Classify a markdown cell by the role it plays in the lecture."""
        content_lower = content.lower()

        if TABLE_MARKER in content and 'important' in content_lower:
            return 'important_note'
        if TABLE_MARKER in content and 'business' in content_lower:
            return 'business_application'
        if TABLE_MARKER in content and 'resources' in content_lower:
            return 'resources'
        if 'before you continue' in content_lower or 'exercise' in content_lower:
            return 'exercise'
        if 'troubleshooting' in content_lower or 'error' in content_lower:
            return 'troubleshooting'
        if content.strip().startswith('#'):
            return 'theory'
        return 'general_markdown'

    def extract_code_pattern(self, code: str) -> dict | None:
        """Classify a code cell, or return None when it carries no reusable pattern."""
        if not code.strip():
            return None
        if code.strip().startswith('#') and len(code.strip().split('\n')) == 1:
            return None

        lines = [line.strip() for line in code.split('\n') if line.strip()]
        if all(line.startswith(('import ', 'from ', '#')) for line in lines):
            if len(lines) > 3:
                return {'type': 'imports', 'code': code}
            return None

        lowered = code.lower()
        if 'openai' in lowered or 'anthropic' in lowered or '.create(' in code:
            return {'type': 'api_call', 'code': code}
        if 'def ' in code:
            return {'type': 'function', 'code': code}
        if 'api_key' in lowered or 'load_dotenv' in code:
            return {'type': 'setup', 'code': code}
        if len(code.strip()) > 20:
            return {'type': 'code_example', 'code': code}
        return None

    def extract_cell_output(self, cell: dict) -> str | None:
        """Compact text form of a code cell's saved output, or None if it has none."""
        fragments: list[str] = []

        for out in cell.get('outputs') or []:
            otype = out.get('output_type')
            if otype == 'stream':
                text = nb2md.strip_ansi(nb2md.join_data(out.get('text', '')))
                if text.strip():
                    fragments.append(text.rstrip())
            elif otype in ('execute_result', 'display_data'):
                data = out.get('data', {})
                if 'text/plain' in data:
                    text = nb2md.strip_ansi(nb2md.join_data(data['text/plain']))
                    if text.strip():
                        fragments.append(text.rstrip())
            elif otype == 'error':
                name = out.get('ename', 'Error')
                value = out.get('evalue', '')
                fragments.append(f"{name}: {value}".strip())

        if not fragments:
            return None

        combined = '\n'.join(fragments)
        if len(combined) > MAX_OUTPUT_CHARS:
            combined = combined[:MAX_OUTPUT_CHARS] + '\n... [truncated]'
        return combined

    def extract_all(self) -> dict[str, Any]:
        """Full structured extraction: metadata, markdown sections and code examples."""
        sections = []
        code_examples = []

        for i, cell in enumerate(self.cells):
            ctype = cell.get('cell_type')
            source = self.cell_source(cell)

            if ctype == 'markdown':
                sections.append(NotebookSection(
                    section_type=self.classify_markdown_cell(source),
                    content=source,
                    metadata={'cell_index': i},
                ))

            elif ctype == 'code':
                output = self.extract_cell_output(cell)
                pattern = self.extract_code_pattern(source)
                # A cell that produced output is evidence of what the code did,
                # so it is kept even when the code itself looks too small to matter.
                if pattern is None and output and source.strip():
                    pattern = {'type': 'code_example', 'code': source}
                if pattern:
                    pattern['cell_index'] = i
                    pattern['language'] = self.language
                    if output:
                        pattern['output'] = output
                    code_examples.append(pattern)

        return {
            'metadata': self.extract_metadata(),
            'sections': [
                {'type': s.type, 'content': s.content, 'metadata': s.metadata}
                for s in sections
            ],
            'code_examples': code_examples,
            'total_cells': self.stats['cells'],
        }

    def extract_key_concepts(self) -> list[str]:
        """Concept names from sub-headings, with lecture narration filtered out."""
        if self.is_stub:
            return []

        concepts: list[str] = []
        seen = set()

        for cell in self.cells:
            if cell.get('cell_type') != 'markdown':
                continue
            for line in self.cell_source(cell).split('\n'):
                if not line.startswith(('## ', '### ')):
                    continue
                raw = line.lstrip('#')
                if is_narration(raw):
                    continue
                concept = clean_heading(raw)
                key = concept.lower()
                if key not in seen:
                    seen.add(key)
                    concepts.append(concept)

        return concepts

    def extract_business_context(self) -> list[str]:
        """Plain text of the notebook's business-application callout boxes."""
        business_sections = []
        for cell in self.cells:
            if cell.get('cell_type') != 'markdown':
                continue
            source = self.cell_source(cell)
            if self.classify_markdown_cell(source) == 'business_application':
                text = callout_text(source)
                if text:
                    business_sections.append(text)
        return business_sections

    def extract_exercises(self) -> list[dict]:
        """Exercise prompts, as plain text with their cell index."""
        exercises = []
        for i, cell in enumerate(self.cells):
            if cell.get('cell_type') != 'markdown':
                continue
            source = self.cell_source(cell)
            if self.classify_markdown_cell(source) == 'exercise':
                text = callout_text(source)
                exercises.append({'cell_index': i, 'content': text})
        return exercises


def extract_notebook(notebook_path: Path) -> dict[str, Any]:
    """Extract one notebook into the dict shape the generators consume."""
    extractor = NotebookExtractor(str(notebook_path))
    data = extractor.extract_all()
    data['key_concepts'] = extractor.extract_key_concepts()
    data['business_context'] = extractor.extract_business_context()
    data['exercises'] = extractor.extract_exercises()
    return data


def _week_notebooks(week_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(week_dir.glob(NOTEBOOK_GLOB)):
        if any(part in str(path) for part in SKIP_PATH_PARTS):
            continue
        paths.append(path)
    return paths


def extract_notebooks_from_week(week: int, base_path: str | None = None) -> list[dict]:
    """Extract every course notebook in `week{N}/`, sorted by filename."""
    base = Path(base_path) if base_path else find_repo_root()
    week_dir = base / f'week{week}'
    if not week_dir.exists():
        return []

    results = []
    for notebook_path in _week_notebooks(week_dir):
        try:
            results.append(extract_notebook(notebook_path))
        except Exception as exc:  # noqa: BLE001 - one bad notebook must not stop the run
            print(f"Error extracting {notebook_path}: {exc}")

    return results


def extract_all_notebooks(base_path: str | None = None,
                          weeks: Any = range(1, 9)) -> dict[int, list[dict]]:
    """Extract every course notebook across the given weeks, keyed by week number."""
    all_data = {}

    for week in weeks:
        week_data = extract_notebooks_from_week(week, base_path)
        if week_data:
            all_data[week] = week_data
            stubs = sum(1 for nb in week_data if nb['metadata']['kind'] == 'stub')
            suffix = f" ({stubs} Colab stub(s))" if stubs else ""
            print(f"  Extracted {len(week_data)} notebooks from Week {week}{suffix}")

    return all_data


if __name__ == '__main__':
    print("Testing notebook extraction...")
    data = extract_notebooks_from_week(1)
    print(f"\nExtracted {len(data)} notebooks from Week 1")

    if data:
        print(f"\nSample metadata from {data[0]['metadata']['filename']}:")
        print(json.dumps(data[0]['metadata'], indent=2))
        print(f"\nFound {len(data[0]['key_concepts'])} key concepts")
        print(f"Found {len(data[0]['code_examples'])} code examples")
