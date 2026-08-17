"""
Lecture Sync - index the per-lecture notes and publish them into the MkDocs site.

Lecture notes live in `study/notes/lectures/weekN/NNN-slug.md` and follow the
template in `study/notes/LECTURE-DISTILL.md`. They are the provenance layer organised by where
you actually were in the course; the topic references in `study/notes/` are the
durable layer organised by concept.

`study/notes/lectures/INDEX.md` is regenerated from the notes themselves, so it is
build output - edit the notes, not the index.
"""
import re
from pathlib import Path
from typing import Any

from extractors.nb2md_loader import find_repo_root

REPO_ROOT = find_repo_root()
DEFAULT_LECTURE_DIR = REPO_ROOT / 'study' / 'notes' / 'lectures'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'study' / 'docs'
LECTURE_SUBDIR = 'lectures'
INDEX_FILENAME = 'INDEX.md'

SYNC_BANNER = (
    "<!-- Generated copy. Source: study/notes/lectures/{source}\n"
    "     Edit the source file; this copy is overwritten on every extraction run. -->\n\n"
)

# `108-simple-rag-dictionary-lookup.md` -> number 108, slug the rest.
LECTURE_FILENAME_RE = re.compile(r'^(\d{3})-(.+)\.md$')
# `# 108 - Day 1 - Building a Simple RAG System`, any dash flavour after the number.
LECTURE_TITLE_RE = re.compile(r'^#\s*\d+\s*[-–—]\s*(.+?)\s*$', re.MULTILINE)
# `**Week 5, Day 1** - 9 min - `week5/day1.ipynb` - slides ...`
WEEK_DAY_RE = re.compile(r'\*\*Week\s*(\d+),\s*Day\s*(\d+)\*\*')
NOTEBOOK_RE = re.compile(r'`([\w./ -]+\.ipynb)`')
NO_NOTEBOOK_MARKERS = ('no notebook', 'concept only', 'concept-only')


def open_question_count(body: str) -> int:
    """Number of top-level bullets under the note's `## Open` heading."""
    match = re.search(r'^##\s+Open\s*$(.*?)(?=^##\s|\Z)', body,
                      re.MULTILINE | re.DOTALL)
    if not match:
        return 0
    return len(re.findall(r'^[-*]\s+\S', match.group(1), re.MULTILINE))


def parse_lecture_note(path: Path) -> dict[str, Any] | None:
    """Read one lecture note into index fields, or None if the filename is off-template."""
    name_match = LECTURE_FILENAME_RE.match(path.name)
    if not name_match:
        return None

    body = path.read_text(encoding='utf-8')
    title_match = LECTURE_TITLE_RE.search(body)
    week_match = WEEK_DAY_RE.search(body)
    notebook_match = NOTEBOOK_RE.search(body)

    header = body[:body.find('\n## ')] if '\n## ' in body else body
    declared_none = any(marker in header.lower() for marker in NO_NOTEBOOK_MARKERS)

    return {
        'number': int(name_match.group(1)),
        'slug': name_match.group(2),
        'filename': path.name,
        'week': int(week_match.group(1)) if week_match else 0,
        'day': int(week_match.group(2)) if week_match else 0,
        'title': title_match.group(1) if title_match else name_match.group(2),
        'notebook': None if declared_none else (
            notebook_match.group(1) if notebook_match else None),
        'open_count': open_question_count(body),
        'path': path,
    }


def collect_lecture_notes(lecture_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every parseable lecture note under `weekN/` subfolders, in lecture order."""
    lecture_dir = Path(lecture_dir) if lecture_dir else DEFAULT_LECTURE_DIR
    if not lecture_dir.is_dir():
        return []

    notes = []
    for week_dir in sorted(lecture_dir.glob('week*')):
        if not week_dir.is_dir():
            continue
        for path in sorted(week_dir.glob('*.md')):
            note = parse_lecture_note(path)
            if note:
                note['week_dir'] = week_dir.name
                notes.append(note)

    return sorted(notes, key=lambda n: n['number'])


def build_index(notes: list[dict[str, Any]], link_prefix: str = '') -> str:
    """Render INDEX.md: lectures grouped by week, with open-question counts."""
    lines = [
        "# Lecture Notes Index",
        "",
        "Condensed per-lecture notes, numbered by Udemy lecture. Open this at the start",
        "of a study block to remember where you were. Regenerated from the notes - see",
        "`study/notes/LECTURE-DISTILL.md` section 5.",
        "",
    ]

    if not notes:
        lines += ["No lecture notes yet. The workflow to produce them is in",
                  "`study/notes/LECTURE-DISTILL.md`.", ""]
        return '\n'.join(lines)

    total_open = sum(n['open_count'] for n in notes)
    lines += [f"**{len(notes)} lecture(s) distilled - {total_open} open question(s).**", ""]

    weeks = sorted({n['week'] for n in notes})
    for week in weeks:
        in_week = [n for n in notes if n['week'] == week]
        week_open = sum(n['open_count'] for n in in_week)
        lines += [f"## Week {week}",
                  "",
                  f"{len(in_week)} note(s), {week_open} open question(s).",
                  "",
                  "| # | Lecture | Notebook | Open |",
                  "| --- | --- | --- | --- |"]
        for note in in_week:
            target = f"{link_prefix}{note['week_dir']}/{note['filename']}"
            notebook = f"`{note['notebook']}`" if note['notebook'] else "-"
            lines.append(
                f"| {note['number']:03d} | [{note['title']}]({target}) "
                f"| {notebook} | {note['open_count']} |"
            )
        lines.append("")

    return '\n'.join(lines)


def generate_lecture_index(lecture_dir: Path | None = None) -> str:
    """Write `study/notes/lectures/INDEX.md` from the notes on disk. Returns its path."""
    lecture_dir = Path(lecture_dir) if lecture_dir else DEFAULT_LECTURE_DIR
    lecture_dir.mkdir(parents=True, exist_ok=True)
    index_path = lecture_dir / INDEX_FILENAME
    index_path.write_text(build_index(collect_lecture_notes(lecture_dir)), encoding='utf-8')
    return str(index_path)


def sync_lectures(lecture_dir: Path | None = None,
                  output_dir: Path | None = None) -> list[str]:
    """Publish the lecture notes and their index into the docs tree.

    Individual notes are deliberately left out of the MkDocs nav - there will
    eventually be hundreds. The index page is the entry point.
    """
    lecture_dir = Path(lecture_dir) if lecture_dir else DEFAULT_LECTURE_DIR
    output_root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    target_dir = output_root / LECTURE_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)

    notes = collect_lecture_notes(lecture_dir)
    written = []

    for note in notes:
        week_target = target_dir / note['week_dir']
        week_target.mkdir(parents=True, exist_ok=True)
        target_path = week_target / note['filename']
        source_rel = f"{note['week_dir']}/{note['filename']}"
        target_path.write_text(
            SYNC_BANNER.format(source=source_rel) + note['path'].read_text(encoding='utf-8'),
            encoding='utf-8',
        )
        written.append(str(target_path))

    index_path = target_dir / 'index.md'
    index_path.write_text(build_index(notes), encoding='utf-8')
    written.append(str(index_path))

    return written
