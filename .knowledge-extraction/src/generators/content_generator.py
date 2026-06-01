"""
Content Generator - Generate topic pages and quick references from extracted data
"""
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from jinja2 import Environment, FileSystemLoader


class ContentGenerator:
    """Generate markdown content from templates"""

    def __init__(self, templates_dir: str = '.knowledge-extraction/templates',
                 output_dir: str = 'knowledge-base/docs'):
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)

        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def generate_topic_page(self, topic_data: Dict, output_path: str) -> str:
        """Generate a topic page from template"""
        template = self.env.get_template('topic_page.md.j2')

        # Prepare template data
        template_data = {
            'title': topic_data.get('title', 'Untitled'),
            'topics': topic_data.get('topics', []),
            'difficulty': topic_data.get('difficulty', 'beginner'),
            'prerequisites': topic_data.get('prerequisites', []),
            'weeks': topic_data.get('weeks', []),
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
            'overview': topic_data.get('overview', ''),
            'core_concepts': topic_data.get('core_concepts', ''),
            'implementation_patterns': topic_data.get('implementation_patterns', ''),
            'common_challenges': topic_data.get('common_challenges', ''),
            'business_applications': topic_data.get('business_applications', ''),
            'troubleshooting': topic_data.get('troubleshooting', ''),
            'related_topics': topic_data.get('related_topics', []),
            'week_references': topic_data.get('week_references', []),
            'resources': topic_data.get('resources', [])
        }

        # Render template
        content = template.render(**template_data)

        # Write to file
        full_path = self.output_dir / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')

        return str(full_path)

    def generate_quickref(self, quickref_data: Dict, output_path: str) -> str:
        """Generate a quick reference page"""
        template = self.env.get_template('quickref.md.j2')

        template_data = {
            'title': quickref_data.get('title', 'Quick Reference'),
            'topics': quickref_data.get('topics', []),
            'last_updated': datetime.now().strftime('%Y-%m-%d'),
            'five_min_setup': quickref_data.get('five_min_setup', ''),
            'common_tasks': quickref_data.get('common_tasks', []),
            'gotchas': quickref_data.get('gotchas', []),
            'costs': quickref_data.get('costs', ''),
            'main_topic_title': quickref_data.get('main_topic_title', ''),
            'main_topic_path': quickref_data.get('main_topic_path', '')
        }

        content = template.render(**template_data)

        full_path = self.output_dir / output_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')

        return str(full_path)

    def generate_week_summary(self, week: int, notebooks_data: List[Dict]) -> str:
        """Generate a week summary page"""
        output_path = self.output_dir / f'week-summaries/week{week}.md'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build content
        lines = [
            f"# Week {week} Summary\n",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*\n",
            f"## Overview\n"
        ]

        # Extract topics covered
        all_concepts = []
        for nb in notebooks_data:
            day = nb.get('metadata', {}).get('day', '?')
            title = nb.get('metadata', {}).get('title', 'Untitled')
            concepts = nb.get('key_concepts', [])

            lines.append(f"### Day {day}: {title}\n")

            if concepts:
                lines.append("**Key Concepts:**")
                for concept in concepts[:5]:  # Top 5
                    lines.append(f"- {concept}")
                    all_concepts.append(concept)
                lines.append("")

            # Business context
            biz_context = nb.get('business_context', [])
            if biz_context:
                lines.append("**Business Applications:**")
                lines.append(f"{biz_context[0][:200]}...\n")

        # Summary
        lines.insert(3, f"This week covered {len(set(all_concepts))} unique concepts across {len(notebooks_data)} days.\n")

        content = '\n'.join(lines)
        output_path.write_text(content, encoding='utf-8')

        return str(output_path)

    def generate_index_page(self, taxonomy: Dict) -> str:
        """Generate the main index page"""
        output_path = self.output_dir / 'index.md'

        lines = [
            "# LLM Engineering Knowledge Base\n",
            "*Personal study notes from Ed Donner's LLM Engineering course*\n",
            "## Legal Disclaimer\n",
            "This knowledge base contains personal study notes from the LLM Engineering course by Ed Donner. ",
            "All content has been paraphrased and reorganized for personal reference.\n",
            "**Original course**: [https://edwarddonner.com/](https://edwarddonner.com/)  ",
            "**Udemy course**: [LLM Engineering Course](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/)\n",
            "No course materials are redistributed. All code examples are original implementations based on concepts learned.\n",
            "For the actual course content, please enroll in the course.\n",
            "---\n",
            "## Navigation\n",
            "### By Topic\n"
        ]

        # Organize topics by category
        if 'categories' in taxonomy:
            for cat_id, cat_info in sorted(taxonomy['categories'].items(),
                                          key=lambda x: x[1].get('order', 99)):
                lines.append(f"#### {cat_info['name']}\n")

                # Find topics in this category
                for topic in taxonomy.get('topics', []):
                    if topic.get('category') == cat_id:
                        folder = topic.get('folder', '')
                        topic_id = topic.get('id', '')
                        topic_name = topic.get('name', '')
                        lines.append(f"- [{topic_name}](topics/{folder}/{topic_id}.md)")

                lines.append("")

        lines.extend([
            "### Quick References\n",
            "- [API Syntax](quick-ref/api-syntax.md)",
            "- [Common Errors](quick-ref/common-errors.md)",
            "- [Code Snippets](quick-ref/code-snippets.md)\n",
            "### By Week\n"
        ])

        for week in range(1, 9):
            lines.append(f"- [Week {week} Summary](week-summaries/week{week}.md)")

        lines.extend([
            "\n### Projects\n",
            "- [Website Summarizer](projects/website-summarizer.md) - Week 1",
            "- [Multi-Model Integration](projects/multi-model.md) - Week 2",
            "- [RAG System](projects/rag-system.md) - Week 5",
            "- [Price Is Right](projects/price-is-right.md) - Weeks 6-7",
            "- [Autonomous Agents](projects/autonomous-agents.md) - Week 8\n",
            "## Search\n",
            "Use the search bar above to find topics, concepts, or code examples.\n",
            "## About This Knowledge Base\n",
            "This knowledge base is automatically generated from course notebooks and supplemented with manual notes. ",
            "It's organized by topic rather than chronologically, making it easier to review concepts and find information.\n",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*\n"
        ])

        content = '\n'.join(lines)
        output_path.write_text(content, encoding='utf-8')

        return str(output_path)


def generate_placeholder_topics(taxonomy_path: str = '.knowledge-extraction/config/taxonomy.yaml',
                                output_dir: str = 'knowledge-base/docs'):
    """Generate placeholder topic pages for all topics in taxonomy"""
    with open(taxonomy_path, 'r') as f:
        taxonomy = yaml.safe_load(f)

    generator = ContentGenerator(output_dir=output_dir)

    for topic in taxonomy.get('topics', []):
        topic_id = topic.get('id')
        folder = topic.get('folder')
        name = topic.get('name')
        description = topic.get('description')
        weeks = topic.get('weeks', [])

        # Create placeholder data
        placeholder_data = {
            'title': name,
            'topics': [topic_id],
            'difficulty': 'intermediate',
            'weeks': weeks,
            'overview': f"{name}\n\n{description}\n\n*Content is being generated...*",
            'core_concepts': "*Coming soon*",
            'week_references': [f"Week {w}" for w in weeks],
            'related_topics': []
        }

        output_path = f"topics/{folder}/{topic_id}.md"
        generator.generate_topic_page(placeholder_data, output_path)
        print(f"[OK] Generated placeholder: {output_path}")

    # Generate index
    generator.generate_index_page(taxonomy)
    print("[OK] Generated index page")


if __name__ == '__main__':
    print("Generating placeholder topic pages...")
    generate_placeholder_topics()
