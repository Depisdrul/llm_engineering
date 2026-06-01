"""
Notebook Extractor - Parse Jupyter notebooks and extract structured content
"""
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import nbformat


class NotebookSection:
    """Represents a section of content extracted from a notebook"""
    def __init__(self, section_type: str, content: str, metadata: Optional[Dict] = None):
        self.type = section_type
        self.content = content
        self.metadata = metadata or {}


class NotebookExtractor:
    """Extract structured content from Jupyter notebooks"""

    def __init__(self, notebook_path: str):
        self.path = Path(notebook_path)
        self.notebook = self._load_notebook()
        self.sections = []

    def _load_notebook(self) -> nbformat.NotebookNode:
        """Load notebook from file"""
        with open(self.path, 'r', encoding='utf-8') as f:
            return nbformat.read(f, as_version=4)

    def extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from notebook path and content"""
        # Parse week and day from path (e.g., week1/day1.ipynb)
        parts = self.path.parts
        week_match = None
        day_match = None

        for part in parts:
            if part.startswith('week'):
                week_match = re.search(r'week(\d+)', part)
            if 'day' in part:
                day_match = re.search(r'day(\d+)', part)

        week = int(week_match.group(1)) if week_match else None
        day = int(day_match.group(1)) if day_match else None

        # Extract title from first markdown cell
        title = None
        for cell in self.notebook.cells:
            if cell.cell_type == 'markdown':
                # Look for main heading
                lines = cell.source.split('\n')
                for line in lines:
                    if line.startswith('# ') and not line.startswith('## '):
                        title = line.strip('# ').strip()
                        break
                if title:
                    break

        return {
            'week': week,
            'day': day,
            'title': title or self.path.stem,
            'filename': self.path.name,
            'path': str(self.path)
        }

    def classify_markdown_cell(self, content: str) -> str:
        """Classify markdown cell type based on content patterns"""
        content_lower = content.lower()

        # Check for special boxes/tables
        if '<table' in content and 'important' in content_lower:
            return 'important_note'
        if '<table' in content and 'business' in content_lower:
            return 'business_application'
        if '<table' in content and 'resources' in content_lower:
            return 'resources'

        # Check for exercises
        if 'before you continue' in content_lower or 'exercise' in content_lower:
            return 'exercise'

        # Check for troubleshooting
        if 'troubleshooting' in content_lower or 'error' in content_lower:
            return 'troubleshooting'

        # Check for headings (theory sections)
        if content.strip().startswith('#'):
            return 'theory'

        return 'general_markdown'

    def extract_code_pattern(self, code: str) -> Optional[Dict]:
        """Extract meaningful code patterns from code cell"""
        # Skip trivial cells
        if not code.strip() or code.strip().startswith('#') and len(code.strip().split('\n')) == 1:
            return None

        # Skip pure imports (unless important)
        lines = [l.strip() for l in code.split('\n') if l.strip()]
        if all(l.startswith('import ') or l.startswith('from ') or l.startswith('#') for l in lines):
            # Keep if it's the main imports cell
            if len(lines) > 3:
                return {
                    'type': 'imports',
                    'code': code
                }
            return None

        # Check for API calls
        if 'openai' in code.lower() or 'anthropic' in code.lower() or '.create(' in code:
            return {
                'type': 'api_call',
                'code': code
            }

        # Check for function definitions
        if 'def ' in code:
            return {
                'type': 'function',
                'code': code
            }

        # Check for configuration/setup
        if 'api_key' in code.lower() or 'load_dotenv' in code:
            return {
                'type': 'setup',
                'code': code
            }

        # General code pattern
        if len(code.strip()) > 20:  # Non-trivial code
            return {
                'type': 'code_example',
                'code': code
            }

        return None

    def extract_all(self) -> Dict[str, Any]:
        """Extract all content from notebook"""
        metadata = self.extract_metadata()
        sections = []
        code_examples = []

        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'markdown':
                cell_type = self.classify_markdown_cell(cell.source)
                section = NotebookSection(
                    section_type=cell_type,
                    content=cell.source,
                    metadata={'cell_index': i}
                )
                sections.append(section)

            elif cell.cell_type == 'code':
                code_pattern = self.extract_code_pattern(cell.source)
                if code_pattern:
                    code_pattern['cell_index'] = i
                    code_examples.append(code_pattern)

        return {
            'metadata': metadata,
            'sections': [{'type': s.type, 'content': s.content, 'metadata': s.metadata}
                        for s in sections],
            'code_examples': code_examples,
            'total_cells': len(self.notebook.cells)
        }

    def extract_key_concepts(self) -> List[str]:
        """Extract key concept names from headings and important sections"""
        concepts = []

        for cell in self.notebook.cells:
            if cell.cell_type == 'markdown':
                # Extract from headings
                lines = cell.source.split('\n')
                for line in lines:
                    if line.startswith('## ') or line.startswith('### '):
                        concept = line.lstrip('#').strip()
                        if concept and len(concept) < 100:  # Reasonable length
                            concepts.append(concept)

        return concepts

    def extract_business_context(self) -> List[str]:
        """Extract business application sections"""
        business_sections = []

        for cell in self.notebook.cells:
            if cell.cell_type == 'markdown':
                if self.classify_markdown_cell(cell.source) == 'business_application':
                    # Extract text content from HTML table
                    text = re.sub(r'<[^>]+>', '', cell.source)  # Remove HTML tags
                    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
                    business_sections.append(text)

        return business_sections

    def extract_exercises(self) -> List[Dict]:
        """Extract exercise prompts"""
        exercises = []

        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'markdown':
                if self.classify_markdown_cell(cell.source) == 'exercise':
                    # Extract text content
                    text = re.sub(r'<[^>]+>', '', cell.source)
                    text = re.sub(r'\s+', ' ', text).strip()
                    exercises.append({
                        'cell_index': i,
                        'content': text
                    })

        return exercises


def extract_notebooks_from_week(week: int, base_path: str = '.') -> List[Dict]:
    """Extract all notebooks from a specific week"""
    base = Path(base_path)
    week_dir = base / f'week{week}'

    if not week_dir.exists():
        return []

    results = []
    for notebook_path in sorted(week_dir.glob('day*.ipynb')):
        try:
            extractor = NotebookExtractor(str(notebook_path))
            data = extractor.extract_all()
            data['key_concepts'] = extractor.extract_key_concepts()
            data['business_context'] = extractor.extract_business_context()
            data['exercises'] = extractor.extract_exercises()
            results.append(data)
        except Exception as e:
            print(f"Error extracting {notebook_path}: {e}")

    return results


def extract_all_notebooks(base_path: str = '.', weeks: range = range(1, 9)) -> Dict[int, List[Dict]]:
    """Extract all notebooks from all weeks"""
    all_data = {}

    for week in weeks:
        week_data = extract_notebooks_from_week(week, base_path)
        if week_data:
            all_data[week] = week_data
            print(f"✓ Extracted {len(week_data)} notebooks from Week {week}")

    return all_data


if __name__ == '__main__':
    # Test extraction on Week 1
    print("Testing notebook extraction...")
    data = extract_notebooks_from_week(1)
    print(f"\nExtracted {len(data)} notebooks from Week 1")

    if data:
        print(f"\nSample metadata from {data[0]['metadata']['filename']}:")
        print(json.dumps(data[0]['metadata'], indent=2))
        print(f"\nFound {len(data[0]['key_concepts'])} key concepts")
        print(f"Found {len(data[0]['code_examples'])} code examples")
