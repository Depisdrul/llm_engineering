# study/

Everything added to this fork of Ed Donner's LLM Engineering course. Nothing
outside this folder is modified, so the diff against upstream stays separable and
`git log -p study/` is the whole history of the work.

**Start with [`notes/DECISIONS.md`](notes/DECISIONS.md)** — the standing record of
why things are shaped the way they are, and where everything else lives.

| | |
| --- | --- |
| [`notes/`](notes/) | Hand-written. **The only files you edit by hand.** Topic references, per-lecture notes, open questions. |
| [`pipeline/`](pipeline/) | The extractor — code, config, templates, tests. |
| `docs/` | MkDocs content. Build output, except `projects/` and `quick-ref/`. |
| `nb2md.py` | The repo's single notebook parser. Standalone and stdlib-only. |
| `notebooks/` | Every course notebook rendered to Markdown. Gitignored; rebuild with the command below. |
| `_site/` | Rendered HTML. Gitignored. |

```bash
# regenerate the site from notes/ and the cached notebook extraction
python study/pipeline/extract_all.py --generate-only

# render every notebook to study/notebooks/ for reading and grepping
python study/nb2md.py . -o study/notebooks --recursive --index \
    --exclude community-contributions --exclude community_contributions

# tests
python -m unittest discover -s study/pipeline -t study/pipeline

# read it
cd study && python -m mkdocs serve
```

`mkdocs build --strict` must stay clean — it is what catches a cross-link broken
while moving content between layers.
