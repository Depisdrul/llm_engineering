"""
LLM Summarizer - Use LLMs to extract key concepts and generate summaries
"""
import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from anthropic import Anthropic
import tiktoken


class LLMSummarizer:
    """Use LLMs to summarize notebook content and generate knowledge base entries"""

    def __init__(self, provider: str = 'openai', model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize summarizer
        Args:
            provider: 'openai', 'anthropic', or 'ollama'
            model: Model to use (defaults to gpt-4.1-mini, claude-3-5-sonnet-20241022, or llama3.2)
            base_url: Base URL for API (used with ollama, defaults to http://localhost:11434/v1)
        """
        self.provider = provider

        if provider == 'openai':
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            self.model = model or 'gpt-4.1-mini'
            self.encoding = tiktoken.encoding_for_model('gpt-4')
        elif provider == 'anthropic':
            self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.model = model or 'claude-3-5-sonnet-20241022'
            self.encoding = tiktoken.get_encoding('cl100k_base')  # Approximation
        elif provider == 'ollama':
            # Ollama uses OpenAI-compatible API
            self.client = OpenAI(
                base_url=base_url or 'http://localhost:11434/v1',
                api_key='ollama'  # Ollama doesn't need a real API key
            )
            self.model = model or 'llama3.2'
            self.encoding = tiktoken.get_encoding('cl100k_base')  # Approximation for token counting
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(self.encoding.encode(text))

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with prompts"""
        if self.provider in ['openai', 'ollama']:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

        else:  # anthropic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return response.content[0].text

    def extract_key_concepts(self, notebook_data: Dict) -> Dict:
        """Extract key concepts from notebook data"""
        system_prompt = """You are an expert at analyzing educational content and extracting key learning objectives.
Your task is to identify the most important concepts, techniques, and skills covered in this material."""

        # Build context from notebook
        context_parts = []

        if notebook_data.get('metadata'):
            meta = notebook_data['metadata']
            context_parts.append(f"Week {meta.get('week')}, Day {meta.get('day')}: {meta.get('title')}")

        # Add theory sections
        for section in notebook_data.get('sections', []):
            if section['type'] in ['theory', 'important_note']:
                context_parts.append(section['content'][:500])  # First 500 chars

        # Add business applications
        for biz in notebook_data.get('business_context', []):
            context_parts.append(f"Business Application: {biz[:300]}")

        context = "\n\n".join(context_parts[:10])  # Limit context

        user_prompt = f"""Analyze this Jupyter notebook section and extract:

1. Main learning objective (1 sentence)
2. Key concepts introduced (bullet list, 3-5 items)
3. Important techniques or patterns (bullet list)
4. Business applications mentioned

Format as structured YAML for parsing.

Notebook content:
{context}
"""

        response = self._call_llm(system_prompt, user_prompt)

        return {
            'summary': response,
            'token_count': self.count_tokens(context + response)
        }

    def generate_topic_summary(self, sections: List[str], topic_name: str) -> Dict:
        """Generate a comprehensive topic summary from multiple sections"""
        system_prompt = f"""You are creating educational documentation for the topic: {topic_name}.
Your goal is to synthesize information from multiple sources into a clear, comprehensive explanation."""

        combined_content = "\n\n---\n\n".join(sections[:5])  # Limit to 5 sections

        user_prompt = f"""Create a comprehensive topic page for: {topic_name}

Based on this content, write:

1. **Overview** (2-3 paragraphs): What is this topic and why does it matter?
2. **Core Concepts** (detailed): Main ideas and principles
3. **Common Challenges**: Issues learners face and how to overcome them
4. **Business Applications**: Real-world use cases

Content to synthesize:
{combined_content}

Write in clear, educational style. Be comprehensive but concise.
"""

        response = self._call_llm(system_prompt, user_prompt)

        return {
            'summary': response,
            'token_count': self.count_tokens(combined_content + response),
            'sources_used': len(sections)
        }

    def generate_quickref_entry(self, detailed_content: str, topic_name: str) -> Dict:
        """Generate a quick reference from detailed content"""
        system_prompt = """You are creating quick reference cheatsheets for developers.
Focus on practical, immediately useful information that can be scanned in 30 seconds."""

        user_prompt = f"""Create a quick reference cheatsheet for: {topic_name}

Based on this detailed content, extract:

1. One-line summary
2. Common usage pattern (code snippet if applicable)
3. Top 3 gotchas or warnings
4. When to use vs alternatives

Format for quick scanning. Be concise and practical.

Detailed content:
{detailed_content[:1500]}
"""

        response = self._call_llm(system_prompt, user_prompt)

        return {
            'quickref': response,
            'token_count': self.count_tokens(detailed_content[:1500] + response)
        }

    def extract_troubleshooting_info(self, sections: List[Dict]) -> Dict:
        """Extract troubleshooting information from notebook sections"""
        system_prompt = """You are extracting troubleshooting information from educational materials.
Focus on identifying error messages, root causes, solutions, and prevention strategies."""

        # Filter for troubleshooting sections
        troubleshooting_content = []
        for section in sections:
            if section.get('type') == 'troubleshooting':
                troubleshooting_content.append(section.get('content', ''))

        if not troubleshooting_content:
            return {'troubleshooting': [], 'token_count': 0}

        combined = "\n\n".join(troubleshooting_content[:3])

        user_prompt = f"""Extract troubleshooting information:

For each issue mentioned, identify:
- Error message (if applicable)
- Root cause
- Solution provided
- Prevention strategy

Format as a table: Error | Cause | Solution | Prevention

Content:
{combined}
"""

        response = self._call_llm(system_prompt, user_prompt)

        return {
            'troubleshooting': response,
            'token_count': self.count_tokens(combined + response)
        }

    def classify_topic(self, content: str, available_topics: List[str]) -> List[str]:
        """Classify content into one or more topics"""
        system_prompt = """You are classifying educational content into topic categories.
Choose all relevant topics that this content covers."""

        user_prompt = f"""Available topics:
{', '.join(available_topics)}

Content to classify:
{content[:800]}

List all relevant topics that this content covers (comma-separated):
"""

        response = self._call_llm(system_prompt, user_prompt)

        # Parse response
        topics = [t.strip() for t in response.split(',')]
        # Match to available topics (fuzzy)
        matched = []
        for topic in topics:
            for available in available_topics:
                if topic.lower() in available.lower() or available.lower() in topic.lower():
                    matched.append(available)

        return list(set(matched))


def summarize_notebook(notebook_data: Dict, summarizer: Optional[LLMSummarizer] = None) -> Dict:
    """Summarize a single notebook"""
    if summarizer is None:
        summarizer = LLMSummarizer()

    result = summarizer.extract_key_concepts(notebook_data)

    return {
        'metadata': notebook_data.get('metadata'),
        'summary': result['summary'],
        'key_concepts': notebook_data.get('key_concepts', []),
        'code_examples': notebook_data.get('code_examples', []),
        'business_context': notebook_data.get('business_context', []),
        'token_count': result['token_count']
    }


if __name__ == '__main__':
    # Test summarizer
    print("Testing LLM summarizer...")

    # Sample notebook data
    sample_data = {
        'metadata': {
            'week': 1,
            'day': 1,
            'title': 'Your First Frontier LLM Project'
        },
        'sections': [
            {
                'type': 'theory',
                'content': '# LLM APIs\n\nLearn how to call OpenAI API'
            }
        ],
        'key_concepts': ['API calls', 'Prompts', 'Streaming'],
        'business_context': ['Summarization use cases in business']
    }

    summarizer = LLMSummarizer(provider='openai')
    result = summarizer.extract_key_concepts(sample_data)
    print(f"✓ Generated summary ({result['token_count']} tokens)")
    print(f"\nSummary preview:\n{result['summary'][:200]}...")
