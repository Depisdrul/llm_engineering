#!/usr/bin/env python3
"""
nb2md.py - render Jupyter notebooks to reviewable Markdown, outputs included.

Built for reviewing course notebooks in long study blocks without re-running
anything. Turns each .ipynb into a single Markdown file containing the author's
prose, the code, and the actual captured outputs (text, tables, errors), with
images extracted to a sibling assets folder.

Zero dependencies - standard library only. Works on stripped notebooks too;
they just render without output blocks.

Usage
-----
  # one notebook
  python nb2md.py week3/day4_executed.ipynb

  # a whole week, into a notes folder
  python nb2md.py week3/ -o notes/week3

  # the entire repo, skipping the 3000+ community contributions
  python nb2md.py . -o notes/ --recursive --exclude community-contributions

  # see which notebooks are Colab stubs vs real content
  python nb2md.py . --recursive --exclude community-contributions --audit

Options
-------
  -o, --out DIR        output directory (default: alongside each notebook)
  -r, --recursive      descend into subdirectories
  --exclude PAT        skip paths containing PAT (repeatable)
  --audit              report only: classify notebooks, render nothing
  --max-output-lines N truncate long text outputs (default 40, 0 = unlimited)
  --no-images          skip image extraction, leave a placeholder note
  --index              also write INDEX.md linking every rendered file
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

# Colab stub detection: a notebook whose only real content is a link to Colab.
COLAB_RE = re.compile(r"colab\.research\.google\.com/(?:drive|github)/\S+")

IMAGE_MIMES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/svg+xml": "svg",
}

FENCE = "```"


# --------------------------------------------------------------------------- #
# notebook loading
# --------------------------------------------------------------------------- #

def load_notebook(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            nb = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        print(f"  ! cannot read {path.name}: {exc}", file=sys.stderr)
        return None
    if not isinstance(nb, dict) or "cells" not in nb:
        print(f"  ! {path.name}: not a notebook (no 'cells')", file=sys.stderr)
        return None
    return nb


def source_text(cell: dict) -> str:
    """Notebook 'source' is a list of lines or a single string."""
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src or ""


def cell_language(nb: dict) -> str:
    meta = nb.get("metadata", {})
    info = meta.get("language_info", {})
    lang = info.get("name") or meta.get("kernelspec", {}).get("language") or "python"
    return str(lang).lower()


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #

def classify(nb: dict) -> tuple[str, dict]:
    """Return (kind, stats). kind is 'stub', 'stripped', or 'executed'."""
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]

    nonempty_code = [c for c in code_cells if source_text(c).strip()]
    with_output = [c for c in code_cells if c.get("outputs")]

    colab_links: list[str] = []
    for c in cells:
        colab_links.extend(COLAB_RE.findall(source_text(c)))

    stats = {
        "cells": len(cells),
        "code": len(code_cells),
        "code_nonempty": len(nonempty_code),
        "markdown": len(md_cells),
        "with_output": len(with_output),
        "colab_links": colab_links,
    }

    # A stub is a notebook that points at Colab and has essentially no code of
    # its own. This is exactly the week3/week7 pattern in the course repo.
    if colab_links and len(nonempty_code) <= 1:
        return "stub", stats
    if with_output:
        return "executed", stats
    return "stripped", stats


# --------------------------------------------------------------------------- #
# output rendering
# --------------------------------------------------------------------------- #

def truncate(text: str, max_lines: int) -> str:
    if max_lines <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = max_lines * 2 // 3
    tail = max_lines - head
    omitted = len(lines) - head - tail
    return "\n".join(
        lines[:head]
        + [f"... [{omitted} lines omitted by nb2md; rerun with --max-output-lines 0] ..."]
        + lines[-tail:]
    )


def join_data(value) -> str:
    if isinstance(value, list):
        return "".join(value)
    return str(value)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", text)


def render_outputs(
    cell: dict,
    out_lines: list[str],
    asset_dir: Path,
    asset_rel: str,
    stem: str,
    counter: list[int],
    max_lines: int,
    save_images: bool,
) -> None:
    outputs = cell.get("outputs") or []
    if not outputs:
        return

    rendered: list[str] = []

    for out in outputs:
        otype = out.get("output_type")

        if otype == "stream":
            text = strip_ansi(join_data(out.get("text", "")))
            if text.strip():
                label = "stderr" if out.get("name") == "stderr" else "stdout"
                rendered.append(
                    f"<sub>{label}</sub>\n\n{FENCE}text\n{truncate(text.rstrip(), max_lines)}\n{FENCE}"
                )

        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})

            # Prefer an image if present - it is usually the point of the cell.
            img_mime = next((m for m in IMAGE_MIMES if m in data), None)
            if img_mime:
                ext = IMAGE_MIMES[img_mime]
                counter[0] += 1
                name = f"{stem}_out{counter[0]:03d}.{ext}"
                if save_images:
                    try:
                        asset_dir.mkdir(parents=True, exist_ok=True)
                        payload = data[img_mime]
                        if img_mime == "image/svg+xml":
                            (asset_dir / name).write_text(
                                join_data(payload), encoding="utf-8"
                            )
                        else:
                            raw = payload if isinstance(payload, str) else "".join(payload)
                            (asset_dir / name).write_bytes(
                                base64.b64decode(raw)
                            )
                        rendered.append(f"![output]({asset_rel}/{name})")
                    except Exception as exc:  # noqa: BLE001 - best effort
                        rendered.append(f"<sub>[image output, could not save: {exc}]</sub>")
                else:
                    rendered.append("<sub>[image output omitted (--no-images)]</sub>")
                continue

            if "text/plain" in data:
                text = strip_ansi(join_data(data["text/plain"]))
                if text.strip():
                    rendered.append(
                        f"{FENCE}text\n{truncate(text.rstrip(), max_lines)}\n{FENCE}"
                    )
            elif "text/markdown" in data:
                rendered.append(join_data(data["text/markdown"]).rstrip())
            elif "text/html" in data:
                rendered.append("<sub>[HTML output - open the notebook to view]</sub>")

        elif otype == "error":
            name = out.get("ename", "Error")
            value = out.get("evalue", "")
            tb = strip_ansi("\n".join(out.get("traceback", [])))
            body = tb if tb.strip() else f"{name}: {value}"
            rendered.append(
                f"<sub>error</sub>\n\n{FENCE}text\n{truncate(body.rstrip(), max_lines)}\n{FENCE}"
            )

    if rendered:
        out_lines.append("> **Output**")
        out_lines.append("")
        out_lines.extend(rendered)
        out_lines.append("")


# --------------------------------------------------------------------------- #
# main rendering
# --------------------------------------------------------------------------- #

def render_notebook(
    path: Path,
    nb: dict,
    kind: str,
    stats: dict,
    out_path: Path,
    max_lines: int,
    save_images: bool,
) -> None:
    lang = cell_language(nb)
    stem = out_path.stem
    asset_rel = f"{stem}_assets"
    asset_dir = out_path.parent / asset_rel
    counter = [0]

    lines: list[str] = [f"# {path.stem}", ""]

    # Provenance header - matters when reviewing months later.
    lines.append(f"<sub>source: `{path}` &middot; {stats['cells']} cells "
                 f"({stats['code']} code, {stats['markdown']} markdown) &middot; "
                 f"{stats['with_output']} cells with saved output</sub>")
    lines.append("")

    if kind == "stub":
        lines.append("> [!WARNING]")
        lines.append("> **Colab stub — no local content.** This notebook in the repo is a")
        lines.append("> pointer only. To get the real material: open the link, "
                     "`File > Save a copy in Drive`,")
        lines.append("> run your copy, then `File > Download > Download .ipynb` and save it")
        lines.append(f"> next to the stub as `{path.stem}_executed.ipynb`. Re-run nb2md after that.")
        lines.append("")
        for link in dict.fromkeys(stats["colab_links"]):
            lines.append(f"> Colab: <{link}>")
        lines.append("")
    elif kind == "stripped":
        lines.append("> [!NOTE]")
        lines.append("> Outputs are stripped in this copy — code and prose only. "
                     "Run it and re-render to capture results.")
        lines.append("")

    lines.append("---")
    lines.append("")

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        src = source_text(cell)

        if ctype == "markdown":
            if src.strip():
                lines.append(src.rstrip())
                lines.append("")

        elif ctype == "code":
            if src.strip():
                lines.append(f"{FENCE}{lang}")
                lines.append(src.rstrip())
                lines.append(FENCE)
                lines.append("")
            render_outputs(
                cell, lines, asset_dir, asset_rel, stem,
                counter, max_lines, save_images,
            )

        elif ctype == "raw" and src.strip():
            lines.append(f"{FENCE}text")
            lines.append(src.rstrip())
            lines.append(FENCE)
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def gather(target: Path, recursive: bool, excludes: list[str]) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix == ".ipynb" else []
    pattern = "**/*.ipynb" if recursive else "*.ipynb"
    found = sorted(target.glob(pattern))
    out = []
    for p in found:
        s = str(p)
        if ".ipynb_checkpoints" in s:
            continue
        if any(x in s for x in excludes):
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render Jupyter notebooks to reviewable Markdown, outputs included.",
    )
    ap.add_argument("target", type=Path, help="notebook file or directory")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output directory (default: alongside each notebook)")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="descend into subdirectories")
    ap.add_argument("--exclude", action="append", default=[], metavar="PAT",
                    help="skip paths containing PAT (repeatable)")
    ap.add_argument("--audit", action="store_true",
                    help="classify notebooks and report; render nothing")
    ap.add_argument("--max-output-lines", type=int, default=40, metavar="N",
                    help="truncate text outputs to N lines (0 = unlimited)")
    ap.add_argument("--no-images", action="store_true",
                    help="do not extract image outputs")
    ap.add_argument("--index", action="store_true",
                    help="write INDEX.md linking every rendered file")
    args = ap.parse_args()

    if not args.target.exists():
        print(f"error: {args.target} does not exist", file=sys.stderr)
        return 2

    notebooks = gather(args.target, args.recursive, args.exclude)
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    buckets: dict[str, list[Path]] = {"stub": [], "stripped": [], "executed": []}
    written: list[tuple[Path, str, dict]] = []
    base = args.target if args.target.is_dir() else args.target.parent

    for path in notebooks:
        nb = load_notebook(path)
        if nb is None:
            continue
        kind, stats = classify(nb)
        buckets[kind].append(path)

        if args.audit:
            continue

        if args.out:
            try:
                rel = path.relative_to(base).parent
            except ValueError:
                rel = Path()
            out_path = args.out / rel / (path.stem + ".md")
        else:
            out_path = path.with_suffix(".md")

        render_notebook(path, nb, kind, stats, out_path,
                        args.max_output_lines, not args.no_images)
        written.append((out_path, kind, stats))

    # ---- report ----------------------------------------------------------- #
    print(f"\nScanned {len(notebooks)} notebook(s) under {args.target}\n")
    labels = {
        "executed": "executed (has saved outputs) - fully reviewable",
        "stripped": "stripped (code + prose, no outputs)",
        "stub": "COLAB STUB - link only, no local content",
    }
    for kind in ("executed", "stripped", "stub"):
        items = buckets[kind]
        print(f"  {len(items):>4}  {labels[kind]}")
    print()

    if buckets["stub"]:
        print("Colab stubs needing manual download:")
        for p in buckets["stub"]:
            print(f"  - {p}")
        print()

    if args.audit:
        return 0

    if args.index and written:
        idx_dir = args.out if args.out else base
        idx_lines = ["# Notebook index", "",
                     f"<sub>{len(written)} notebook(s) rendered by nb2md</sub>", ""]
        badge = {"executed": "reviewable", "stripped": "no outputs", "stub": "**COLAB STUB**"}
        for out_path, kind, stats in written:
            try:
                link = out_path.relative_to(idx_dir)
            except ValueError:
                link = out_path
            idx_lines.append(
                f"- [{out_path.stem}]({link}) — {badge[kind]}, "
                f"{stats['code_nonempty']} code cell(s)"
            )
        idx_path = idx_dir / "INDEX.md"
        idx_path.write_text("\n".join(idx_lines) + "\n", encoding="utf-8")
        print(f"Wrote index: {idx_path}")

    print(f"Rendered {len(written)} Markdown file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
