#!/usr/bin/env python3
"""
Slides Extractor - render the course slide decks to Markdown and pull out their images.

The decks arrive as one zip per week from Google Drive, each holding one `.pptx`
per day. A `.pptx` is itself a zip of XML, so both layers are read in place and
nothing is unpacked to disk except the images.

Everything is written beside the decks in `study/notes/lectures/_slides/`, which
is gitignored - the decks are the instructor's material, and so is a
transcription of them. No pipeline phase may read this output and nothing under
`study/docs/` may depend on it: the corpus is not in git, so a fresh clone does
not have it. That is why this is a standalone script rather than a phase in
`extract_all.py`.

Images are deduplicated by content hash, because the same logos and banners
recur across dozens of decks. Each unique image is recorded in `manifest.json`
with every occurrence and that slide's text, which is the context needed to
judge what the image is worth.

Usage
-----
  # everything, into _slides/text/ and _slides/media/
  python study/pipeline/src/extractors/slides_extractor.py

  # report what is there, render nothing
  python study/pipeline/src/extractors/slides_extractor.py --audit

  # text only, no image dump
  python study/pipeline/src/extractors/slides_extractor.py --no-images

Options
-------
  -i, --input DIR    folder holding the week zips or loose .pptx (default: _slides/)
  -o, --out DIR      output root (default: the input folder)
  --audit            report only
  --no-images        skip the image dump; the manifest still counts them
  --min-pixels N     ignore images smaller than N pixels of area (default 0)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractors.nb2md_loader import find_repo_root  # noqa: E402

REPO_ROOT = find_repo_root()
DEFAULT_SLIDES_DIR = REPO_ROOT / 'study' / 'notes' / 'lectures' / '_slides'
TEXT_SUBDIR = 'text'
MEDIA_SUBDIR = 'media'
MANIFEST_FILENAME = 'manifest.json'

NS_P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

TAG_SP = f'{{{NS_P}}}sp'
TAG_PH = f'{{{NS_P}}}ph'
TAG_SLD_ID = f'{{{NS_P}}}sldId'
TAG_SP_TREE = f'{{{NS_P}}}cSld/{{{NS_P}}}spTree'
TAG_NV_PH = f'{{{NS_P}}}nvSpPr/{{{NS_P}}}nvPr/{{{NS_P}}}ph'
TAG_A_P = f'{{{NS_A}}}p'
TAG_A_T = f'{{{NS_A}}}t'
TAG_A_BR = f'{{{NS_A}}}br'
TAG_BLIP = f'{{{NS_A}}}blip'
ATTR_R_ID = f'{{{NS_R}}}id'
ATTR_R_EMBED = f'{{{NS_R}}}embed'

# Slide-number, date and footer placeholders repeat on every slide and say nothing.
PLACEHOLDER_SKIP = {'sldNum', 'dt', 'ftr'}
TITLE_PLACEHOLDERS = {'title', 'ctrTitle'}

# `Copy of LLM - Week 5 Day 2.pptx`, `Copy of LLMs Week 6 Day 3.pptx`, and friends.
DECK_NAME_RE = re.compile(r'week\s*[-–]?\s*(\d+)\s*[-–]?\s*day\s*[-–]?\s*(\d+)', re.IGNORECASE)
ZIP_WEEK_RE = re.compile(r'week\s*[-–]?\s*(\d+)', re.IGNORECASE)

# A first line longer than this is a sentence someone wrapped, not a slide title.
TITLE_PROMOTION_MAX_CHARS = 80


def ascii_safe(text: str) -> str:
    """Console output only - a cp1252 Windows terminal raises on non-ASCII."""
    return text.encode('ascii', 'replace').decode('ascii')


# --------------------------------------------------------------------------- #
# image probing
# --------------------------------------------------------------------------- #

def _jpeg_size(data: bytes) -> tuple[int, int]:
    offset, end = 2, len(data)
    while offset + 9 < end:
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        segment_length = struct.unpack('>H', data[offset + 2:offset + 4])[0]
        # SOF0-SOF15 carry the frame header; DHT/JPG/DAC share the range and do not.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack('>HH', data[offset + 5:offset + 9])
            return width, height
        offset += 2 + segment_length
    return 0, 0


def image_size(data: bytes) -> tuple[int, int]:
    """Pixel dimensions of a PNG, JPEG or GIF. `(0, 0)` for anything else."""
    if data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) >= 24:
        return struct.unpack('>II', data[16:24])
    if data[:2] == b'\xff\xd8':
        return _jpeg_size(data)
    if data[:4] in (b'GIF8',) and len(data) >= 10:
        return struct.unpack('<HH', data[6:10])
    return 0, 0


# --------------------------------------------------------------------------- #
# pptx parts
# --------------------------------------------------------------------------- #

def _resolve(base_part: str, target: str) -> str:
    """Resolve a relationship target against the part that declared it."""
    if target.startswith('/'):
        return target.lstrip('/')
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_part), target))


def _relationships(pptx: zipfile.ZipFile, part: str) -> dict[str, str]:
    """Map relationship id -> resolved part name for one part, or `{}` if it has none."""
    rels_part = posixpath.join(posixpath.dirname(part), '_rels', posixpath.basename(part) + '.rels')
    try:
        root = ET.fromstring(pptx.read(rels_part))
    except (KeyError, ET.ParseError):
        return {}
    return {
        rel.get('Id'): _resolve(part, rel.get('Target', ''))
        for rel in root
        if rel.get('Id') and rel.get('Target')
    }


def _numeric_slide_order(pptx: zipfile.ZipFile) -> list[str]:
    slides = [n for n in pptx.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
    return sorted(slides, key=lambda n: int(re.search(r'(\d+)', posixpath.basename(n)).group(1)))


def slide_order(pptx: zipfile.ZipFile) -> list[str]:
    """Slide part names in presentation order.

    The order comes from `ppt/presentation.xml`, not from the `slideN.xml`
    filenames: reordering a deck rewrites that list and leaves the filenames
    alone, so numeric order can silently disagree with what the audience saw.
    Falls back to numeric order when the part is missing or unparseable.
    """
    try:
        presentation = ET.fromstring(pptx.read('ppt/presentation.xml'))
    except (KeyError, ET.ParseError):
        return _numeric_slide_order(pptx)

    targets = _relationships(pptx, 'ppt/presentation.xml')
    names = set(pptx.namelist())
    ordered = []
    for slide_id in presentation.iter(TAG_SLD_ID):
        part = targets.get(slide_id.get(ATTR_R_ID, ''))
        if part and part in names and part not in ordered:
            ordered.append(part)
    return ordered or _numeric_slide_order(pptx)


def paragraph_text(paragraph: ET.Element) -> str:
    """One `<a:p>` flattened to a single line, runs joined and whitespace collapsed."""
    parts = []
    for node in paragraph.iter():
        if node.tag == TAG_A_T:
            parts.append(node.text or '')
        elif node.tag == TAG_A_BR:
            parts.append(' ')
    return ' '.join(''.join(parts).split())


def shape_lines(shape: ET.Element) -> list[str]:
    """Every non-empty paragraph under a shape, in document order.

    Descends into groups and tables, so a table's cells arrive as one line each.
    """
    return [text for text in (paragraph_text(p) for p in shape.iter(TAG_A_P)) if text]


def placeholder_type(shape: ET.Element) -> str | None:
    """The shape's placeholder type, `'body'` for an untyped placeholder, else None."""
    placeholder = shape.find(TAG_NV_PH)
    if placeholder is None:
        return None
    return placeholder.get('type', 'body')


def parse_slide(xml_bytes: bytes) -> dict[str, Any]:
    """One slide's title and body lines, from the raw part bytes."""
    try:
        return parse_slide_element(ET.fromstring(xml_bytes))
    except ET.ParseError:
        return {'title': '', 'lines': []}


def parse_slide_element(root: ET.Element) -> dict[str, Any]:
    """One slide's title and body lines."""
    tree = root.find(TAG_SP_TREE)
    if tree is None:
        return {'title': '', 'lines': []}

    title = ''
    lines: list[str] = []
    for shape in tree:
        if shape.tag == TAG_SP:
            kind = placeholder_type(shape)
            if kind in PLACEHOLDER_SKIP:
                continue
            found = shape_lines(shape)
            if not found:
                continue
            if kind in TITLE_PLACEHOLDERS and not title:
                title, found = found[0], found[1:]
            lines.extend(found)
        else:
            lines.extend(shape_lines(shape))

    # Decks that style their titles as plain text boxes have no title placeholder.
    # Promote a short opening line rather than leave the heading blank; a long one
    # is prose and stays in the body.
    if not title and lines and len(lines[0]) <= TITLE_PROMOTION_MAX_CHARS:
        title = lines.pop(0)

    return {'title': title, 'lines': lines}


def slide_image_parts(pptx: zipfile.ZipFile, slide_part: str, root: ET.Element) -> list[str]:
    """Media part names referenced by one slide, in document order, without repeats."""
    relationships = _relationships(pptx, slide_part)
    parts: list[str] = []
    for blip in root.iter(TAG_BLIP):
        part = relationships.get(blip.get(ATTR_R_EMBED, ''))
        if part and part not in parts:
            parts.append(part)
    return parts


# --------------------------------------------------------------------------- #
# deck discovery
# --------------------------------------------------------------------------- #

def deck_identity(deck_name: str, source_name: str = '') -> tuple[int, int]:
    """`(week, day)` parsed from the deck filename, falling back to the zip's week."""
    match = DECK_NAME_RE.search(deck_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    week_match = ZIP_WEEK_RE.search(deck_name) or ZIP_WEEK_RE.search(source_name)
    return (int(week_match.group(1)) if week_match else 0), 0


def iter_decks(input_dir: Path):
    """Yield `(source_name, deck_name, deck_bytes)` for every deck found.

    Reads `.pptx` members straight out of the week zips; loose `.pptx` files in
    the same folder are picked up too, so an unpacked folder works unchanged.
    """
    for archive_path in sorted(input_dir.glob('*.zip')):
        with zipfile.ZipFile(archive_path) as archive:
            for name in sorted(n for n in archive.namelist() if n.lower().endswith('.pptx')):
                yield archive_path.name, name, archive.read(name)

    for deck_path in sorted(input_dir.rglob('*.pptx')):
        yield '', str(deck_path.relative_to(input_dir)).replace('\\', '/'), deck_path.read_bytes()


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def render_deck(deck: dict[str, Any]) -> str:
    """A deck's Markdown page: one section per slide, images named at the end of each."""
    source = f"`{deck['deck']}`"
    if deck['archive']:
        source += f" in `{deck['archive']}`"

    lines = [
        f"# Week {deck['week']}, Day {deck['day']}" if deck['day']
        else f"# Week {deck['week']} - {deck['deck']}",
        "",
        f"<!-- Generated from {source} by study/pipeline/src/extractors/slides_extractor.py.",
        "     The deck is the instructor's material and so is this rendering:",
        "     both stay inside the gitignored _slides/ folder. -->",
        "",
        f"**{len(deck['slides'])} slides, {deck['image_count']} image reference(s).**",
        "",
    ]

    for number, slide in enumerate(deck['slides'], start=1):
        heading = f"## Slide {number}"
        if slide['title']:
            heading += f" - {slide['title']}"
        lines += [heading, ""]
        lines += [f"- {text}" for text in slide['lines']]
        if slide['lines']:
            lines.append("")
        if slide['images']:
            lines += ["_Images: " + ', '.join(f"`{name}`" for name in slide['images']) + "_", ""]

    return '\n'.join(lines).rstrip() + '\n'


def extract_slides(input_dir: Path | None = None,
                   output_dir: Path | None = None,
                   write_images: bool = True,
                   min_pixels: int = 0) -> dict[str, Any]:
    """Parse every deck, writing one Markdown page each plus deduplicated images.

    Returns the manifest. Images are keyed by SHA-1 of their bytes, so a logo
    repeated across decks is stored once and carries one occurrence per slide it
    appears on.
    """
    input_dir = Path(input_dir) if input_dir else DEFAULT_SLIDES_DIR
    output_root = Path(output_dir) if output_dir else input_dir
    text_dir = output_root / TEXT_SUBDIR
    media_dir = output_root / MEDIA_SUBDIR

    decks: list[dict[str, Any]] = []
    images: dict[str, dict[str, Any]] = {}
    used_stems: set[str] = set()

    for archive_name, deck_name, deck_bytes in iter_decks(input_dir):
        week, day = deck_identity(deck_name, archive_name)
        try:
            pptx = zipfile.ZipFile(io.BytesIO(deck_bytes))
        except zipfile.BadZipFile:
            print(ascii_safe(f"  ! not a readable .pptx: {deck_name}"), file=sys.stderr)
            continue

        with pptx:
            slides = []
            image_count = 0
            for slide_part in slide_order(pptx):
                try:
                    root = ET.fromstring(pptx.read(slide_part))
                except (KeyError, ET.ParseError):
                    continue
                slide = parse_slide_element(root)
                slide_images = []

                for media_part in slide_image_parts(pptx, slide_part, root):
                    try:
                        data = pptx.read(media_part)
                    except KeyError:
                        continue
                    width, height = image_size(data)
                    if min_pixels and width * height < min_pixels:
                        continue
                    image_count += 1
                    digest = hashlib.sha1(data).hexdigest()
                    extension = posixpath.splitext(media_part)[1].lstrip('.').lower() or 'bin'
                    filename = f"{digest[:12]}.{extension}"
                    entry = images.setdefault(digest, {
                        'file': f"{MEDIA_SUBDIR}/{filename}",
                        'sha1': digest,
                        'width': width,
                        'height': height,
                        'bytes': len(data),
                        'occurrences': [],
                    })
                    entry['occurrences'].append({
                        'week': week,
                        'day': day,
                        'deck': deck_name,
                        'slide': len(slides) + 1,
                        'slide_title': slide['title'],
                        'slide_text': ' '.join(slide['lines']),
                    })
                    slide_images.append(filename)

                    if write_images:
                        media_dir.mkdir(parents=True, exist_ok=True)
                        target = media_dir / filename
                        if not target.exists():
                            target.write_bytes(data)

                slide['images'] = slide_images
                slides.append(slide)

        stem = f"week{week}-day{day}" if day else f"week{week}-{Path(deck_name).stem.lower()}"
        stem = re.sub(r'[^a-z0-9-]+', '-', stem).strip('-')
        candidate, suffix = stem, 2
        while candidate in used_stems:
            candidate, suffix = f"{stem}-{suffix}", suffix + 1
        used_stems.add(candidate)

        deck = {
            'week': week,
            'day': day,
            'deck': deck_name,
            'archive': archive_name,
            'slides': slides,
            'image_count': image_count,
            'text_chars': sum(len(line) for slide in slides for line in slide['lines']),
            'markdown': f"{TEXT_SUBDIR}/{candidate}.md",
        }
        decks.append(deck)

        text_dir.mkdir(parents=True, exist_ok=True)
        (text_dir / f"{candidate}.md").write_text(render_deck(deck), encoding='utf-8')

    decks.sort(key=lambda d: (d['week'], d['day'], d['deck']))
    manifest = {
        'decks': [{k: v for k, v in deck.items() if k != 'slides'} | {'slides': len(deck['slides'])}
                  for deck in decks],
        'images': sorted(images.values(), key=lambda i: -len(i['occurrences'])),
    }

    if decks:
        (output_root / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    return manifest


def audit(input_dir: Path | None = None, min_pixels: int = 0) -> dict[str, Any]:
    """Count decks, slides, text and images without writing anything."""
    input_dir = Path(input_dir) if input_dir else DEFAULT_SLIDES_DIR
    decks = 0
    slides = 0
    chars = 0
    instances = 0
    unique: set[str] = set()

    for _archive_name, _deck_name, deck_bytes in iter_decks(input_dir):
        try:
            pptx = zipfile.ZipFile(io.BytesIO(deck_bytes))
        except zipfile.BadZipFile:
            continue
        decks += 1
        with pptx:
            for slide_part in slide_order(pptx):
                try:
                    root = ET.fromstring(pptx.read(slide_part))
                except (KeyError, ET.ParseError):
                    continue
                slides += 1
                parsed = parse_slide_element(root)
                chars += len(parsed['title']) + sum(len(line) for line in parsed['lines'])
                for media_part in slide_image_parts(pptx, slide_part, root):
                    try:
                        data = pptx.read(media_part)
                    except KeyError:
                        continue
                    width, height = image_size(data)
                    if min_pixels and width * height < min_pixels:
                        continue
                    instances += 1
                    unique.add(hashlib.sha1(data).hexdigest())

    return {'decks': decks, 'slides': slides, 'text_chars': chars,
            'image_instances': instances, 'unique_images': len(unique)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', type=Path, default=DEFAULT_SLIDES_DIR,
                        help='folder holding the week zips or loose .pptx files')
    parser.add_argument('-o', '--out', type=Path, default=None,
                        help='output root (default: the input folder)')
    parser.add_argument('--audit', action='store_true', help='report only, render nothing')
    parser.add_argument('--no-images', action='store_true', help='skip the image dump')
    parser.add_argument('--min-pixels', type=int, default=0,
                        help='ignore images below this pixel area')
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"No slides folder at {args.input}", file=sys.stderr)
        return 1

    if args.audit:
        report = audit(args.input, args.min_pixels)
        print(f"decks            {report['decks']}")
        print(f"slides           {report['slides']}")
        print(f"text             {report['text_chars']} chars (~{report['text_chars'] // 4} tokens)")
        print(f"image references {report['image_instances']}")
        print(f"unique images    {report['unique_images']}")
        return 0

    manifest = extract_slides(args.input, args.out,
                              write_images=not args.no_images,
                              min_pixels=args.min_pixels)
    if not manifest['decks']:
        print(f"No decks found in {args.input}", file=sys.stderr)
        return 1

    for deck in manifest['decks']:
        print(ascii_safe(f"  [OK] {deck['markdown']:<28} {deck['slides']:>3} slides, "
                         f"{deck['image_count']:>3} images"))

    total_slides = sum(deck['slides'] for deck in manifest['decks'])
    total_chars = sum(deck['text_chars'] for deck in manifest['decks'])
    print(f"\n[OK] {len(manifest['decks'])} decks, {total_slides} slides, "
          f"{total_chars} chars (~{total_chars // 4} tokens)")
    print(f"[OK] {len(manifest['images'])} unique images "
          f"from {sum(len(i['occurrences']) for i in manifest['images'])} references")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
