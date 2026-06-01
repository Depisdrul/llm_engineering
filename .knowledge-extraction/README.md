# LLM Engineering Course - Knowledge Extraction System

Automated knowledge extraction pipeline for Ed Donner's LLM Engineering course. Extracts key concepts, code patterns, and business applications from 35+ Jupyter notebooks and organizes them into a searchable, topic-based knowledge base.

## Features

- **Automated Extraction**: Parse all course notebooks and extract structured content
- **Topic Organization**: Content organized by topic (not chronologically) for easier review
- **LLM Summarization**: Optional AI-powered summarization of key concepts
- **Web Scraping**: Extract supplementary content from Ed Donner's website
- **MkDocs Site**: Beautiful, searchable documentation site with Material theme
- **Quick References**: Cheatsheets for syntax, common errors, and code snippets
- **Week Summaries**: High-level overviews of what each week covers

## Quick Start

### 1. Installation

Dependencies are already installed. If you need to reinstall:

```powershell
cd .knowledge-extraction
pip install -r requirements.txt
```

### 2. Generate Placeholder Content

Create the initial knowledge base structure:

```powershell
python .knowledge-extraction\extract_all.py --generate-only
```

### 3. Extract Course Content

Extract from all notebooks (no LLM summarization):

```powershell
python .knowledge-extraction\extract_all.py
```

Extract specific weeks:

```powershell
python .knowledge-extraction\extract_all.py --weeks 1,2,3
```

### 4. View the Knowledge Base

Build and serve the MkDocs site:

```powershell
cd knowledge-base
python -m mkdocs serve
```

Then open http://localhost:8000 in your browser.

## Usage Options

### Basic Extraction (Fast, No AI)

```powershell
python .knowledge-extraction\extract_all.py
```

This will:
- Extract all notebooks (structure, code, concepts)
- Generate week summaries
- Create topic placeholder pages
- No LLM calls = instant and free

### With Local AI (FREE, Best Option!)

```powershell
# First time: Install Ollama and pull a model
# Visit https://ollama.com to download
ollama pull llama3.2

# Then run extraction with local AI
python .knowledge-extraction\extract_all.py --use-llm --provider ollama

# Or specify a different model
python .knowledge-extraction\extract_all.py --use-llm --provider ollama --model qwen2.5
```

**Recommended local models:**
- `llama3.2` - Fast, good quality (3B parameters)
- `qwen2.5` - Better reasoning (7B parameters)  
- `mistral` - Excellent for code (7B parameters)
- `llama3.1` - Most capable (8B parameters)

Pull any model: `ollama pull <model-name>`

### With LLM Summarization - FREE with Ollama! 🎉

**Option A: Local Model (FREE, requires Ollama)**

```powershell
# 1. Install Ollama from https://ollama.com
# 2. Pull a model (recommended: llama3.2 or qwen2.5)
ollama pull llama3.2

# 3. Run extraction with local model
python .knowledge-extraction\extract_all.py --use-llm --provider ollama
```

**Option B: Paid APIs (Better quality, faster)**

```powershell
# OpenAI (~$5-10 for full course)
python .knowledge-extraction\extract_all.py --use-llm --provider openai

# Anthropic Claude (~$10-15 for full course)
python .knowledge-extraction\extract_all.py --use-llm --provider anthropic
```

Both add:
- AI-powered summarization of concepts
- Automated topic classification
- Quick-reference generation
- Extract key troubleshooting information

### Skip Website Scraping

```powershell
python .knowledge-extraction\extract_all.py --skip-website
```

### Extract Specific Weeks

```powershell
python .knowledge-extraction\extract_all.py --weeks 5,6,7
```

## Directory Structure

```
.knowledge-extraction/
├── src/
│   ├── extractors/          # Notebook and web content extraction
│   ├── processors/          # LLM summarization
│   └── generators/          # Content generation
├── templates/               # Jinja2 templates for markdown pages
├── config/
│   └── taxonomy.yaml       # Topic organization structure
└── extract_all.py          # Main pipeline script

knowledge-base/
├── docs/                    # Generated markdown content
│   ├── index.md            # Main page with navigation
│   ├── topics/             # Comprehensive topic pages
│   ├── quick-ref/          # Cheatsheets and quick references
│   ├── week-summaries/     # Week-by-week summaries
│   └── projects/           # Project guides
├── mkdocs.yml              # MkDocs configuration
└── site/                   # Built static site
```

## Workflow

### Initial Setup (One-time, ~1 hour)

1. Generate placeholder structure
2. Run extraction pipeline on all weeks
3. Review and manually enhance key topic pages
4. Build MkDocs site

### Ongoing Maintenance (After each session, ~15 min)

1. Extract newly attended week:
   ```powershell
   python .knowledge-extraction\extract_all.py --weeks 6
   ```

2. Add personal notes to relevant topic pages

3. Rebuild site:
   ```powershell
   cd knowledge-base
   python -m mkdocs build
   ```

## Customization

### Add New Topics

Edit `.knowledge-extraction/config/taxonomy.yaml`:

```yaml
topics:
  - id: new-topic-id
    name: New Topic Name
    category: category-name
    folder: 01-foundations
    description: Topic description
    weeks: [1, 2]
```

Then regenerate:

```powershell
python .knowledge-extraction\extract_all.py --generate-only
```

### Modify Templates

Edit Jinja2 templates in `.knowledge-extraction/templates/`:
- `topic_page.md.j2` - Comprehensive topic pages
- `quickref.md.j2` - Quick reference cheatsheets

### Customize MkDocs Theme

Edit `knowledge-base/mkdocs.yml` to change:
- Colors and theme
- Navigation structure  
- Plugins and extensions
- Search configuration

## Testing

Test extraction on a single week:

```powershell
cd .knowledge-extraction
python test_extraction.py
```

## Cost Estimation

**Basic extraction**: $0 (no AI)

**With Ollama (local AI)**: $0 (FREE - runs on your computer)
- Requires: ~4-8GB RAM for the model
- Speed: Depends on your hardware (slower than cloud APIs but free)
- Quality: Very good for educational content

**With OpenAI API**:
- Estimated tokens: ~500K-1M (with prompt caching)
- Model: GPT-4.1-mini ($0.15 per 1M tokens)
- **Total cost: $5-10 one-time**
- Ongoing: ~$1-2 per week for new content

**With Anthropic API**:
- Model: Claude 3.5 Sonnet
- **Total cost: ~$10-15 one-time**

## Troubleshooting

### No notebooks found

Check that you're running from the correct directory (should be `llm_engineering/`):

```powershell
cd C:\Users\Pazzucconibt\REPO\llm_engineering
python .knowledge-extraction\extract_all.py
```

### MkDocs build errors

Missing week summary files are normal before extraction. Warnings are OK; errors need fixing.

### API key errors

Ensure `.env` file contains:
```
OPENAI_API_KEY=sk-proj-...
```

## Next Steps

1. **Run full extraction**: Extract all 8 weeks to populate content
2. **Manual enhancement**: Review and improve key topic pages
3. **Add diagrams**: Create architecture diagrams for complex topics (RAG, Agents)
4. **Personal notes**: Add insights from attended sessions
5. **Deploy (optional)**: Publish to GitHub Pages:
   ```powershell
   cd knowledge-base
   python -m mkdocs gh-deploy
   ```

## Maintenance Schedule

**Weekly** (15 min):
- Extract newly attended sessions
- Add personal notes
- Quick validation

**Monthly** (1 hour):
- Review automated summaries
- Enhance one topic area with examples/diagrams
- Update troubleshooting based on issues encountered

## Legal

This is a personal study aid. All content is paraphrased and reorganized for educational purposes. Course materials are not redistributed.

Original course: https://edwarddonner.com/
