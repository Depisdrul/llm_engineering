# study/

Everything added to this fork of Ed Donner's LLM Engineering course. Nothing
outside this folder is modified, so the diff against upstream stays separable and
`git log -p study/` is the whole history of the work.

**Start with [`notes/HANDOFF.md`](notes/HANDOFF.md)** — it is the re-entry point:
what the layers are, what has been verified, what is still open, and why.

| | |
| --- | --- |
| [`notes/`](notes/) | Hand-written. **The only files you edit by hand.** Topic references, per-lecture notes, open questions. |
| [`pipeline/`](pipeline/) | The extractor — code, config, templates, tests. |
| `docs/` | MkDocs content. Build output, except `index.md`. |
| `nb2md.py` | The repo's single notebook parser. Standalone and stdlib-only. |
| `_site/` | Rendered HTML. Gitignored. |

```bash
# regenerate the site from notes/ and the cached notebook extraction
python study/pipeline/extract_all.py --generate-only

# tests
python -m unittest discover -s study/pipeline -t study/pipeline

# read it
cd study && python -m mkdocs serve
```

`mkdocs build --strict` must stay clean — it is what catches a cross-link broken
while moving content between layers.
