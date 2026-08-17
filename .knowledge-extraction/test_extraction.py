#!/usr/bin/env python3
"""
Tests for the knowledge-extraction pipeline.

Run from the repo root:  python -m unittest discover -s .knowledge-extraction
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from extractors import nb2md_loader
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
        self.assertTrue((REPO_ROOT / 'knowledge-base' / 'nb2md.py').is_file())

    def test_loaded_module_exposes_parsing_primitives(self):
        nb2md = nb2md_loader.load_nb2md()
        for name in ('load_notebook', 'classify', 'source_text', 'cell_language',
                     'strip_ansi', 'join_data'):
            self.assertTrue(hasattr(nb2md, name), f"nb2md is missing {name}")

    def test_module_is_cached(self):
        first = nb2md_loader.load_nb2md()
        second = nb2md_loader.load_nb2md()
        self.assertIs(first, second)


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
        self.assertEqual(reference_sync.DEFAULT_SOURCE_DIR.name, 'notes')

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

    DOCS = REPO_ROOT / 'knowledge-base' / 'docs'

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
