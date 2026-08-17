#!/usr/bin/env python3
"""
Tests for the study pipeline.

Run from the repo root:  python -m unittest discover -s study/pipeline
"""
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from extractors import nb2md_loader, slides_extractor
from extractors.notebook_extractor import (
    NotebookExtractor,
    callout_text,
    clean_heading,
    extract_notebooks_from_week,
    is_narration,
    repo_relative,
)
from generators import lecture_sync, reference_sync
from generators.content_generator import (
    ContentGenerator,
    load_taxonomy,
    reference_callout,
    relative_topic_path,
    score_notebook,
    topic_keywords,
)

REPO_ROOT = nb2md_loader.find_repo_root()


def make_notebook(cells, path: Path) -> Path:
    """Write a minimal but valid .ipynb containing `cells`."""
    path.write_text(json.dumps({
        'cells': cells,
        'metadata': {'language_info': {'name': 'python'}},
        'nbformat': 4,
        'nbformat_minor': 5,
    }), encoding='utf-8')
    return path


def md(source: str) -> dict:
    return {'cell_type': 'markdown', 'source': source, 'metadata': {}}


def code(source: str, outputs=None) -> dict:
    return {
        'cell_type': 'code',
        'source': source,
        'metadata': {},
        'execution_count': 1,
        'outputs': outputs or [],
    }


class Nb2mdLoaderTests(unittest.TestCase):
    def test_find_repo_root_locates_nb2md(self):
        self.assertTrue((REPO_ROOT / 'study' / 'nb2md.py').is_file())

    def test_loaded_module_exposes_parsing_primitives(self):
        nb2md = nb2md_loader.load_nb2md()
        for name in ('load_notebook', 'classify', 'source_text', 'cell_language',
                     'strip_ansi', 'join_data'):
            self.assertTrue(hasattr(nb2md, name), f"nb2md is missing {name}")

    def test_module_is_cached(self):
        first = nb2md_loader.load_nb2md()
        second = nb2md_loader.load_nb2md()
        self.assertIs(first, second)

    def test_gather_skips_dotted_directories(self):
        nb2md = nb2md_loader.load_nb2md()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ('week1/day1.ipynb',
                        '.venv/Lib/site-packages/vendor/demo.ipynb',
                        'week1/.ipynb_checkpoints/day1-checkpoint.ipynb'):
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('{"cells": []}', encoding='utf-8')
            found = nb2md.gather(root, recursive=True, excludes=[])
        self.assertEqual([p.name for p in found], ['day1.ipynb'])

    def test_index_links_are_posix(self):
        nb2md = nb2md_loader.load_nb2md()
        notebook = {'cells': [{'cell_type': 'code', 'source': 'print(1)',
                               'metadata': {}, 'execution_count': 1, 'outputs': []}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'src' / 'week1'
            src.mkdir(parents=True)
            (src / 'day1.ipynb').write_text(json.dumps(notebook), encoding='utf-8')
            out = root / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                rc = nb2md.main([str(root / 'src'), '-o', str(out), '--recursive', '--index'])
            self.assertEqual(rc, 0)
            index = (out / 'INDEX.md').read_text(encoding='utf-8')
        self.assertIn('(week1/day1.md)', index)
        self.assertNotIn('\\', index)

    def test_gather_does_not_read_dots_in_the_target_path(self):
        nb2md = nb2md_loader.load_nb2md()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / '.hidden-workspace'
            (root / 'week1').mkdir(parents=True)
            (root / 'week1' / 'day1.ipynb').write_text('{"cells": []}', encoding='utf-8')
            found = nb2md.gather(root, recursive=True, excludes=[])
        self.assertEqual([p.name for p in found], ['day1.ipynb'])


class HeadingFilterTests(unittest.TestCase):
    def test_clean_heading_strips_decoration(self):
        self.assertEqual(clean_heading(' **Tokenizers**: '), 'Tokenizers')
        self.assertEqual(clean_heading('[Chroma](https://x.dev) basics'), 'Chroma basics')

    def test_narration_is_rejected(self):
        for heading in [
            "Let's read in all employee data into a dictionary",
            'Donezo! On to Step 2 - make the chunks',
            'Well that was easy! If a bit slow.',
            'Finally, Step 3 - save the embeddings',
            'TODAY:',
            'Three steps:',
            'What could possibly come next? 😂',
            'Admit it - you thought RAG would be more complicated than that!!',
            'Step 1',
            '   ',
            '12345',
        ]:
            self.assertTrue(is_narration(heading), f"should be narration: {heading!r}")

    def test_concepts_are_kept(self):
        for heading in [
            'Expert Knowledge Worker',
            'Connect to Chroma; use Hugging Face all-MiniLM-L6-v2',
            'Tokenizers',
            'RAG Day 3',
            'Evaluation!',
            'Expert Question Answerer for InsureLLM',
        ]:
            self.assertFalse(is_narration(heading), f"should be a concept: {heading!r}")

    def test_overlong_heading_is_rejected(self):
        self.assertTrue(is_narration('x' * 200))


class CalloutTextTests(unittest.TestCase):
    def test_only_the_table_is_extracted(self):
        source = (
            '# Welcome to RAG week!!\n\n## Expert Knowledge Worker\n\n'
            '<table><tr><td>Business applications of this week</td></tr></table>'
        )
        text = callout_text(source)
        self.assertIn('Business applications of this week', text)
        self.assertNotIn('Welcome to RAG week', text)

    def test_falls_back_to_whole_cell_without_a_table(self):
        self.assertEqual(callout_text('plain   prose'), 'plain prose')


class RepoRelativeTests(unittest.TestCase):
    def test_path_inside_repo_is_relative_and_posix(self):
        self.assertEqual(repo_relative(REPO_ROOT / 'week5' / 'day1.ipynb'), 'week5/day1.ipynb')

    def test_path_outside_repo_is_left_alone(self):
        outside = Path(tempfile.gettempdir()).resolve() / 'elsewhere.ipynb'
        self.assertNotIn('..', repo_relative(outside))


class NotebookExtractorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_colab_stub_is_detected_and_yields_no_concepts(self):
        path = make_notebook(
            [md('# Day 1\n\nhttps://colab.research.google.com/drive/abc123?usp=sharing')],
            self.tmp / 'day1.ipynb',
        )
        extractor = NotebookExtractor(str(path))
        self.assertTrue(extractor.is_stub)
        self.assertEqual(extractor.extract_key_concepts(), [])
        self.assertTrue(extractor.extract_metadata()['colab_links'])

    def test_executed_notebook_captures_output(self):
        path = make_notebook([
            md('# Day 4\n\n## Evaluation!'),
            code('print("hello")', outputs=[
                {'output_type': 'stream', 'name': 'stdout', 'text': ['hello\n']},
            ]),
        ], self.tmp / 'day4.ipynb')

        extractor = NotebookExtractor(str(path))
        self.assertEqual(extractor.kind, 'executed')

        data = extractor.extract_all()
        self.assertEqual(data['metadata']['cells_with_output'], 1)
        self.assertEqual(data['code_examples'][0]['output'], 'hello')

    def test_error_output_is_captured(self):
        path = make_notebook([
            code('boom()', outputs=[
                {'output_type': 'error', 'ename': 'NameError',
                 'evalue': "name 'boom' is not defined", 'traceback': []},
            ]),
        ], self.tmp / 'day9.ipynb')

        example = NotebookExtractor(str(path)).extract_all()['code_examples'][0]
        self.assertIn('NameError', example['output'])

    def test_concepts_are_deduplicated_and_filtered(self):
        path = make_notebook([
            md('# Day 2\n\n## Expert Knowledge Worker\n\n## expert knowledge worker'),
            md("## Let's go PRO!\n\n## Chunking strategy"),
        ], self.tmp / 'day2.ipynb')

        concepts = NotebookExtractor(str(path)).extract_key_concepts()
        self.assertEqual(concepts, ['Expert Knowledge Worker', 'Chunking strategy'])

    def test_unreadable_file_raises(self):
        bad = self.tmp / 'broken.ipynb'
        bad.write_text('not json', encoding='utf-8')
        bad_path = str(bad)
        with self.assertRaises(ValueError):
            NotebookExtractor(bad_path)


class RealRepoExtractionTests(unittest.TestCase):
    """Guards against the pipeline silently extracting nothing from the real repo."""

    def test_week5_has_content_and_relative_paths(self):
        notebooks = extract_notebooks_from_week(5)
        self.assertTrue(notebooks, "week5 should contain notebooks")
        for nb in notebooks:
            path = nb['metadata']['path']
            self.assertFalse(Path(path).is_absolute(), f"absolute path leaked: {path}")
            self.assertTrue(path.startswith('week5/'), path)

    def test_week3_is_all_colab_stubs(self):
        kinds = {nb['metadata']['kind'] for nb in extract_notebooks_from_week(3)}
        self.assertEqual(kinds, {'stub'})


class ReferenceSyncTests(unittest.TestCase):
    def test_every_listed_source_exists(self):
        for note in reference_sync.REFERENCE_NOTES:
            source = reference_sync.DEFAULT_SOURCE_DIR / note['source']
            self.assertTrue(source.is_file(), f"missing reference source: {note['source']}")

    def test_sources_live_in_notes(self):
        self.assertEqual(reference_sync.DEFAULT_SOURCE_DIR, REPO_ROOT / 'study' / 'notes')

    def test_cross_links_are_rewritten_to_published_slugs(self):
        body = 'see [foundations](./01-llm-foundations.md) and [rag](05ragandvectorsearch.md#4-chunking)'
        rewritten = reference_sync.rewrite_cross_links(body)
        self.assertIn('(llm-foundations.md)', rewritten)
        self.assertIn('(rag-and-vector-search.md#4-chunking)', rewritten)

    def test_unknown_and_external_links_are_untouched(self):
        body = '[x](./nope.md) and [y](https://example.com/README.md)'
        self.assertEqual(reference_sync.rewrite_cross_links(body), body)

    def test_index_omits_notes_with_no_source_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / 'src').mkdir()
            (tmp_path / 'src' / '01-llm-foundations.md').write_text('# x', encoding='utf-8')
            index = Path(reference_sync.generate_reference_index(
                output_dir=tmp_path / 'docs', source_dir=tmp_path / 'src'))
            body = index.read_text(encoding='utf-8')
            self.assertIn('(llm-foundations.md)', body)
            self.assertNotIn('(rag-and-vector-search.md)', body)

    def test_taxonomy_references_all_resolve(self):
        slugs = {note['slug'] for note in reference_sync.REFERENCE_NOTES}
        for topic in load_taxonomy().get('topics', []):
            slug = topic.get('reference')
            if slug:
                self.assertIn(slug, slugs, f"{topic['id']} points at unknown reference {slug}")


LECTURE_NOTE = """# 108 — Day 1 - Building a Simple RAG System

**Week 5, Day 1** · 9 min · `week5/day1.ipynb` · deck *LLM - Week 5 Day 1*

## Claim

A naive dictionary lookup is enough to prove context injection works.

## Open

- What is the token size of the corpus?
- Does the week reach hybrid retrieval?

## Links

- Repo: `week5/day1.ipynb`
"""

CONCEPT_ONLY_NOTE = """# 087 — AI Model Benchmarks

**Week 4, Day 2** · 12 min · no notebook

## Claim

Benchmarks are contaminated.

## Open

- Which leaderboards are still trustworthy?
"""


class LectureSyncTests(unittest.TestCase):
    def write_lectures(self, tmp: Path) -> Path:
        lectures = tmp / 'lectures'
        (lectures / 'week5').mkdir(parents=True)
        (lectures / 'week4').mkdir(parents=True)
        (lectures / 'week5' / '108-simple-rag.md').write_text(LECTURE_NOTE, encoding='utf-8')
        (lectures / 'week4' / '087-benchmarks.md').write_text(CONCEPT_ONLY_NOTE, encoding='utf-8')
        (lectures / 'week5' / 'scratch.md').write_text('# not a lecture', encoding='utf-8')
        return lectures

    def test_notes_are_parsed_in_lecture_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = lecture_sync.collect_lecture_notes(self.write_lectures(Path(tmp)))
            self.assertEqual([n['number'] for n in notes], [87, 108])

    def test_off_template_filenames_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = lecture_sync.collect_lecture_notes(self.write_lectures(Path(tmp)))
            self.assertNotIn('scratch.md', {n['filename'] for n in notes})

    def test_fields_come_from_the_note_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = lecture_sync.collect_lecture_notes(self.write_lectures(Path(tmp)))
            by_number = {n['number']: n for n in notes}
            self.assertEqual(by_number[108]['week'], 5)
            self.assertEqual(by_number[108]['notebook'], 'week5/day1.ipynb')
            self.assertEqual(by_number[108]['open_count'], 2)
            self.assertIn('Building a Simple RAG System', by_number[108]['title'])

    def test_concept_only_lectures_report_no_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = lecture_sync.collect_lecture_notes(self.write_lectures(Path(tmp)))
            note = next(n for n in notes if n['number'] == 87)
            self.assertIsNone(note['notebook'])
            self.assertEqual(note['open_count'], 1)

    def test_index_groups_by_week_and_totals_open_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes = lecture_sync.collect_lecture_notes(self.write_lectures(Path(tmp)))
            index = lecture_sync.build_index(notes)
            self.assertIn('## Week 4', index)
            self.assertIn('## Week 5', index)
            self.assertIn('3 open question(s)', index)
            self.assertIn('week5/108-simple-rag.md', index)

    def test_empty_index_is_still_valid(self):
        self.assertIn('No lecture notes yet', lecture_sync.build_index([]))

    def test_sync_publishes_notes_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lectures = self.write_lectures(tmp_path)
            docs = tmp_path / 'docs'
            written = lecture_sync.sync_lectures(lectures, docs)
            self.assertEqual(len(written), 3)
            self.assertTrue((docs / 'lectures' / 'index.md').is_file())
            published = docs / 'lectures' / 'week5' / '108-simple-rag.md'
            self.assertTrue(published.is_file())
            self.assertIn('Generated copy', published.read_text(encoding='utf-8'))


class TopicPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = load_taxonomy()
        cls.notebook_data = {5: extract_notebooks_from_week(5)}
        cls.generator = ContentGenerator()

    def test_topic_keywords_drop_generic_terms(self):
        keywords = topic_keywords({'name': 'RAG Systems', 'description': 'Vector databases and chunking'})
        self.assertIn('chunking', keywords)
        self.assertNotIn('and', keywords)
        self.assertNotIn('systems', keywords)

    def test_score_notebook_prefers_matching_content(self):
        notebooks = self.notebook_data[5]
        keywords = ['chroma', 'chunk']
        self.assertGreater(max(score_notebook(nb, keywords) for nb in notebooks), 0)

    def test_relative_topic_path(self):
        self.assertEqual(relative_topic_path('04-rag', '04-rag', 'embeddings'), 'embeddings.md')
        self.assertEqual(
            relative_topic_path('04-rag', '05-agents', 'tool-use'),
            '../05-agents/tool-use.md',
        )

    def test_reference_callout_links_known_slug(self):
        callout = reference_callout('rag-and-vector-search')
        self.assertIn('../../reference/rag-and-vector-search.md', callout)

    def test_reference_callout_warns_when_absent(self):
        self.assertIn('No primary-source reference yet', reference_callout(None))

    def test_rag_topic_page_has_real_content(self):
        topic = next(t for t in self.taxonomy['topics'] if t['id'] == 'rag-systems')
        page = self.generator.build_topic_page_data(topic, self.notebook_data, self.taxonomy)

        self.assertIn('reference/rag-and-vector-search.md', page['overview'])
        self.assertIn('Expert Knowledge Worker', page['core_concepts'])
        self.assertIn('Chroma', page['implementation_patterns'])

    def test_topic_page_never_ships_placeholder_text(self):
        for topic in self.taxonomy['topics']:
            page = self.generator.build_topic_page_data(topic, self.notebook_data, self.taxonomy)
            rendered = ' '.join(str(v) for v in page.values())
            for banned in ('Coming soon', 'To be extracted', 'Content is being generated',
                           'To be filled during extraction'):
                self.assertNotIn(banned, rendered, f"{topic['id']} still ships a placeholder")


class GeneratedDocsTests(unittest.TestCase):
    """Checks the docs currently on disk, i.e. the last pipeline run's output."""

    DOCS = REPO_ROOT / 'study' / 'docs'

    def test_no_placeholders_remain_in_topic_pages(self):
        pages = list((self.DOCS / 'topics').rglob('*.md'))
        self.assertTrue(pages, "no topic pages have been generated")
        for page in pages:
            text = page.read_text(encoding='utf-8')
            self.assertNotIn('*Coming soon*', text, f"{page.name} still ships a placeholder")

    def test_references_were_published(self):
        for note in reference_sync.REFERENCE_NOTES:
            published = self.DOCS / 'reference' / f"{note['slug']}.md"
            self.assertTrue(published.is_file(), f"reference not published: {note['slug']}")

    def test_no_absolute_paths_leak_into_week_summaries(self):
        for page in (self.DOCS / 'week-summaries').glob('*.md'):
            text = page.read_text(encoding='utf-8')
            self.assertNotIn(str(REPO_ROOT), text, f"{page.name} leaks an absolute path")


SLIDE_XML = (
    '<p:sld xmlns:p="{p}" xmlns:a="{a}" xmlns:r="{r}">'
    '<p:cSld><p:spTree><p:nvGrpSpPr/>{shapes}</p:spTree></p:cSld></p:sld>'
)
RELS_XML = (
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '{items}</Relationships>'
)


def shape(paragraphs, placeholder: str | None = None, untyped: bool = False) -> str:
    """A `<p:sp>` text shape. `placeholder` sets its `<p:ph type=...>`."""
    if untyped:
        ph = '<p:ph/>'
    elif placeholder:
        ph = f'<p:ph type="{placeholder}"/>'
    else:
        ph = ''
    body = ''.join(f'<a:p><a:r><a:t>{text}</a:t></a:r></a:p>' for text in paragraphs)
    return (f'<p:sp><p:nvSpPr><p:nvPr>{ph}</p:nvPr></p:nvSpPr>'
            f'<p:txBody>{body}</p:txBody></p:sp>')


def picture(rel_id: str) -> str:
    return f'<p:pic><p:blipFill><a:blip r:embed="{rel_id}"/></p:blipFill></p:pic>'


def table(rows) -> str:
    cells = ''.join(
        '<a:tr>' + ''.join(
            f'<a:tc><a:txBody><a:p><a:r><a:t>{cell}</a:t></a:r></a:p></a:txBody></a:tc>'
            for cell in row
        ) + '</a:tr>'
        for row in rows
    )
    return ('<p:graphicFrame><a:graphic><a:graphicData>'
            f'<a:tbl>{cells}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>')


def png_bytes(width: int, height: int, salt: bytes = b'') -> bytes:
    """A PNG header real enough for `image_size`. `salt` changes the content hash."""
    return (b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR'
            + struct.pack('>II', width, height) + b'\x08\x06\x00\x00\x00' + salt)


def make_pptx(slides, media=None, order=None) -> bytes:
    """Build a minimal .pptx in memory.

    `slides` is a list of dicts with `shapes` (XML string) and optional `images`
    ({rel_id: media filename}). `order` is the 1-based slide sequence written into
    `presentation.xml`, which is how a real deck records its display order.
    """
    order = order or list(range(1, len(slides) + 1))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        items = ''.join(
            f'<Relationship Id="rId{position}" Target="slides/slide{number}.xml" Type="slide"/>'
            for position, number in enumerate(order, start=1)
        )
        archive.writestr('ppt/_rels/presentation.xml.rels', RELS_XML.format(items=items))
        slide_ids = ''.join(
            f'<p:sldId id="{255 + position}" r:id="rId{position}"/>'
            for position in range(1, len(order) + 1)
        )
        archive.writestr(
            'ppt/presentation.xml',
            f'<p:presentation xmlns:p="{slides_extractor.NS_P}" xmlns:r="{slides_extractor.NS_R}">'
            f'<p:sldIdLst>{slide_ids}</p:sldIdLst></p:presentation>',
        )
        for number, slide in enumerate(slides, start=1):
            archive.writestr(f'ppt/slides/slide{number}.xml', SLIDE_XML.format(
                p=slides_extractor.NS_P, a=slides_extractor.NS_A, r=slides_extractor.NS_R,
                shapes=slide['shapes'],
            ))
            images = slide.get('images') or {}
            if images:
                rels = ''.join(
                    f'<Relationship Id="{rel_id}" Target="../media/{name}" Type="image"/>'
                    for rel_id, name in images.items()
                )
                archive.writestr(f'ppt/slides/_rels/slide{number}.xml.rels',
                                 RELS_XML.format(items=rels))
        for name, data in (media or {}).items():
            archive.writestr(f'ppt/media/{name}', data)
    return buffer.getvalue()


def write_week_zip(directory: Path, week: int, decks: dict) -> Path:
    """Write a `Week N-....zip` holding `{deck filename: pptx bytes}`."""
    path = directory / f'Week {week}-20260817T125248Z-1-001.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        for name, data in decks.items():
            archive.writestr(f'Week {week}/{name}', data)
    return path


class SlidesExtractorTests(unittest.TestCase):
    def test_output_stays_inside_the_gitignored_slides_folder(self):
        expected = REPO_ROOT / 'study' / 'notes' / 'lectures' / '_slides'
        self.assertEqual(slides_extractor.DEFAULT_SLIDES_DIR, expected)

    def test_deck_identity_handles_real_naming_variants(self):
        for name, expected in [
            ('Copy of LLM Week 1 Day 3.pptx', (1, 3)),
            ('Copy of LLMs Week 2 Day 3.pptx', (2, 3)),
            ('Copy of LLM - Week 5 Day 2.pptx', (5, 2)),
            ('Copy of LLMs Week 8 Day 4.pptx', (8, 4)),
        ]:
            self.assertEqual(slides_extractor.deck_identity(name), expected, name)

    def test_deck_identity_falls_back_to_the_archive_week(self):
        self.assertEqual(
            slides_extractor.deck_identity('Intro.pptx', 'Week 6-20260817T125258Z-1-001.zip'),
            (6, 0),
        )

    def test_slide_order_follows_presentation_not_filenames(self):
        deck = make_pptx(
            [{'shapes': shape(['first file'])}, {'shapes': shape(['second file'])}],
            order=[2, 1],
        )
        with zipfile.ZipFile(io.BytesIO(deck)) as pptx:
            parts = slides_extractor.slide_order(pptx)
            texts = [slides_extractor.parse_slide(pptx.read(p))['title'] for p in parts]
        self.assertEqual(texts, ['second file', 'first file'])

    def test_title_placeholder_becomes_the_heading(self):
        parsed = slides_extractor.parse_slide(SLIDE_XML.format(
            p=slides_extractor.NS_P, a=slides_extractor.NS_A, r=slides_extractor.NS_R,
            shapes=shape(['Five Leaderboards'], placeholder='title')
                   + shape(['Hugging Face', 'Vellum'], placeholder='body'),
        ).encode('utf-8'))
        self.assertEqual(parsed['title'], 'Five Leaderboards')
        self.assertEqual(parsed['lines'], ['Hugging Face', 'Vellum'])

    def test_short_opening_line_is_promoted_when_no_title_placeholder(self):
        parsed = slides_extractor.parse_slide(SLIDE_XML.format(
            p=slides_extractor.NS_P, a=slides_extractor.NS_A, r=slides_extractor.NS_R,
            shapes=shape(['The Arena', 'LM Arena is a resource']),
        ).encode('utf-8'))
        self.assertEqual(parsed['title'], 'The Arena')
        self.assertEqual(parsed['lines'], ['LM Arena is a resource'])

    def test_long_opening_line_stays_in_the_body(self):
        sentence = ('Build a product that converts Python code to C++ for performance, '
                    'solving it once with a frontier model and once with an open-source one')
        parsed = slides_extractor.parse_slide(SLIDE_XML.format(
            p=slides_extractor.NS_P, a=slides_extractor.NS_A, r=slides_extractor.NS_R,
            shapes=shape([sentence, 'and then measure it']),
        ).encode('utf-8'))
        self.assertEqual(parsed['title'], '')
        self.assertEqual(parsed['lines'], [sentence, 'and then measure it'])

    def test_footer_and_slide_number_placeholders_are_dropped(self):
        parsed = slides_extractor.parse_slide(SLIDE_XML.format(
            p=slides_extractor.NS_P, a=slides_extractor.NS_A, r=slides_extractor.NS_R,
            shapes=shape(['Real content'], placeholder='title')
                   + shape(['7'], placeholder='sldNum')
                   + shape(['edwarddonner.com'], placeholder='ftr'),
        ).encode('utf-8'))
        self.assertEqual(parsed['title'], 'Real content')
        self.assertEqual(parsed['lines'], [])

    def test_table_cells_become_lines(self):
        parsed = slides_extractor.parse_slide(SLIDE_XML.format(
            p=slides_extractor.NS_P, a=slides_extractor.NS_A, r=slides_extractor.NS_R,
            shapes=shape(['Benchmarks'], placeholder='title') + table([['Model', 'Score']]),
        ).encode('utf-8'))
        self.assertEqual(parsed['lines'], ['Model', 'Score'])

    def test_image_size_reads_png_dimensions(self):
        self.assertEqual(slides_extractor.image_size(png_bytes(1280, 720)), (1280, 720))
        self.assertEqual(slides_extractor.image_size(b'not an image'), (0, 0))

    def test_identical_images_dedupe_across_decks(self):
        logo = png_bytes(1280, 720, salt=b'logo')
        deck = make_pptx(
            [{'shapes': shape(['Deck title'], placeholder='title') + picture('rId9'),
              'images': {'rId9': 'image1.png'}}],
            media={'image1.png': logo},
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_week_zip(tmp_path, 4, {'Copy of LLM Week 4 Day 2.pptx': deck})
            write_week_zip(tmp_path, 5, {'Copy of LLM - Week 5 Day 3.pptx': deck})
            manifest = slides_extractor.extract_slides(tmp_path)

            self.assertEqual(len(manifest['images']), 1)
            image = manifest['images'][0]
            self.assertEqual(len(image['occurrences']), 2)
            self.assertEqual([(o['week'], o['day']) for o in image['occurrences']],
                             [(4, 2), (5, 3)])
            self.assertEqual(image['occurrences'][0]['slide_title'], 'Deck title')
            self.assertEqual(len(list((tmp_path / 'media').glob('*.png'))), 1)
            self.assertTrue((tmp_path / 'text' / 'week4-day2.md').is_file())
            self.assertTrue((tmp_path / 'manifest.json').is_file())

    def test_min_pixels_filters_icons(self):
        deck = make_pptx(
            [{'shapes': shape(['Title'], placeholder='title')
              + picture('rId1') + picture('rId2'),
              'images': {'rId1': 'icon.png', 'rId2': 'chart.png'}}],
            media={'icon.png': png_bytes(32, 32, salt=b'i'),
                   'chart.png': png_bytes(1280, 720, salt=b'c')},
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_week_zip(tmp_path, 4, {'Copy of LLM Week 4 Day 1.pptx': deck})
            manifest = slides_extractor.extract_slides(tmp_path, min_pixels=400 * 300)
            self.assertEqual(len(manifest['images']), 1)
            self.assertEqual(manifest['images'][0]['width'], 1280)

    def test_no_images_still_counts_them_without_writing(self):
        deck = make_pptx(
            [{'shapes': shape(['Title'], placeholder='title') + picture('rId1'),
              'images': {'rId1': 'chart.png'}}],
            media={'chart.png': png_bytes(800, 600, salt=b'c')},
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_week_zip(tmp_path, 7, {'Copy of LLMs Week 7 Day 1.pptx': deck})
            manifest = slides_extractor.extract_slides(tmp_path, write_images=False)
            self.assertEqual(len(manifest['images']), 1)
            self.assertFalse((tmp_path / 'media').exists())

    def test_rendered_page_carries_slides_and_image_names(self):
        deck = make_pptx(
            [{'shapes': shape(['Five Leaderboards'], placeholder='title')
              + shape(['Hugging Face', 'Vellum'], placeholder='body') + picture('rId1'),
              'images': {'rId1': 'chart.png'}}],
            media={'chart.png': png_bytes(1280, 720, salt=b'x')},
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_week_zip(tmp_path, 4, {'Copy of LLM Week 4 Day 2.pptx': deck})
            slides_extractor.extract_slides(tmp_path)
            page = (tmp_path / 'text' / 'week4-day2.md').read_text(encoding='utf-8')

        self.assertIn('# Week 4, Day 2', page)
        self.assertIn('## Slide 1 - Five Leaderboards', page)
        self.assertIn('- Hugging Face', page)
        self.assertIn('_Images: `', page)
        self.assertIn("instructor's material", page)

    def test_audit_reports_without_writing(self):
        deck = make_pptx([{'shapes': shape(['Title'], placeholder='title')
                           + shape(['a line'], placeholder='body')}])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            write_week_zip(tmp_path, 3, {'Copy of LLM Week 3 Day 1.pptx': deck})
            report = slides_extractor.audit(tmp_path)
            self.assertEqual(report['decks'], 1)
            self.assertEqual(report['slides'], 1)
            self.assertEqual(report['unique_images'], 0)
            self.assertFalse((tmp_path / 'text').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
