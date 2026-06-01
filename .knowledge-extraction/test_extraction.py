#!/usr/bin/env python3
"""Quick test of notebook extraction"""
import sys
sys.path.insert(0, 'src')

from extractors.notebook_extractor import extract_notebooks_from_week
from pathlib import Path

print("Testing notebook extraction on Week 1...")
# Go up one directory to find week folders
base_path = Path('..').resolve()
print(f"Base path: {base_path}")
data = extract_notebooks_from_week(1, base_path=str(base_path))

print(f"\nExtracted {len(data)} notebooks from Week 1")

if data:
    print(f"\nFirst notebook: {data[0]['metadata']['title']}")
    print(f"  Week: {data[0]['metadata']['week']}")
    print(f"  Day: {data[0]['metadata']['day']}")
    print(f"  Key concepts: {len(data[0]['key_concepts'])}")
    print(f"  Code examples: {len(data[0]['code_examples'])}")
    print(f"  Business context: {len(data[0]['business_context'])}")
    print(f"  Exercises: {len(data[0]['exercises'])}")

    print("\nSample key concepts:")
    for concept in data[0]['key_concepts'][:5]:
        print(f"  - {concept}")

print("\nExtraction test complete!")
