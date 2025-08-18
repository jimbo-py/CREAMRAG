"""
Add Wikipedia data to existing corpus for enhanced CREAM-RAG training
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm
import random

def load_existing_corpus(corpus_path: str = "data/corpus.jsonl"):
    """Load existing corpus"""
    existing_docs = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            existing_docs.append(doc)
    return existing_docs

def get_wikipedia_data(num_articles: int = 10000):
    """Load Wikipedia articles from HuggingFace datasets"""
    print(f"Loading {num_articles} Wikipedia articles...")
    
    # Load Wikipedia dataset
    dataset = load_dataset("wikipedia", "20220301.en", split="train")
    
    # Sample articles
    sampled_dataset = dataset.shuffle(seed=42).select(range(min(num_articles, len(dataset))))
    
    wikipedia_docs = []
    for i, article in enumerate(tqdm(sampled_dataset, desc="Processing Wikipedia articles")):
        # Clean and format the article
        title = article['title']
        text = article['text']
        
        # Skip very short articles
        if len(text) < 100:
            continue
            
        # Create document in same format as existing corpus
        doc = {
            "id": f"wiki_{i}",
            "text": f"Title: {title}\n\n{text}"
        }
        wikipedia_docs.append(doc)
        
        if len(wikipedia_docs) >= num_articles:
            break
    
    return wikipedia_docs

def create_enhanced_corpus(existing_docs, wikipedia_docs, output_path: str = "data/enhanced_corpus.jsonl"):
    """Combine existing corpus with Wikipedia data"""
    print("Creating enhanced corpus...")
    
    # Combine datasets
    enhanced_docs = existing_docs + wikipedia_docs
    
    # Shuffle to mix the data
    random.shuffle(enhanced_docs)
    
    # Save enhanced corpus
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in enhanced_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"Enhanced corpus saved to {output_path}")
    print(f"Total documents: {len(enhanced_docs)}")
    print(f"Original documents: {len(existing_docs)}")
    print(f"Wikipedia articles: {len(wikipedia_docs)}")
    
    return output_path

def create_wikipedia_only_corpus(num_articles: int = 50000, output_path: str = "data/wikipedia_corpus.jsonl"):
    """Create a Wikipedia-only corpus for comparison"""
    print(f"Creating Wikipedia-only corpus with {num_articles} articles...")
    
    wikipedia_docs = get_wikipedia_data(num_articles)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in wikipedia_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"Wikipedia-only corpus saved to {output_path}")
    print(f"Total Wikipedia articles: {len(wikipedia_docs)}")
    
    return output_path

def main():
    """Main function to create enhanced datasets"""
    print("=== CREAM-RAG Data Enhancement ===")
    
    # Option 1: Create enhanced corpus (existing + Wikipedia)
    print("\n1. Creating enhanced corpus...")
    existing_docs = load_existing_corpus()
    wikipedia_docs = get_wikipedia_data(10000)  # Add 10K Wikipedia articles
    enhanced_path = create_enhanced_corpus(existing_docs, wikipedia_docs)
    
    # Option 2: Create Wikipedia-only corpus
    print("\n2. Creating Wikipedia-only corpus...")
    wiki_only_path = create_wikipedia_only_corpus(50000)
    
    print("\n=== Dataset Options ===")
    print(f"1. Enhanced Corpus (Mixed): {enhanced_path}")
    print(f"   - {len(existing_docs)} original documents")
    print(f"   - {len(wikipedia_docs)} Wikipedia articles")
    print(f"   - Total: {len(existing_docs) + len(wikipedia_docs)} documents")
    
    print(f"\n2. Wikipedia Only: {wiki_only_path}")
    print(f"   - 50,000 Wikipedia articles")
    
    print(f"\n3. Original Corpus: data/corpus.jsonl")
    print(f"   - {len(existing_docs)} original documents")
    
    print("\n=== Recommendation ===")
    print("For your CREAM-RAG research, I recommend:")
    print("1. Use the Enhanced Corpus for main training (mixed data)")
    print("2. Use Wikipedia-only for comparison experiments")
    print("3. Keep original corpus as baseline")

if __name__ == "__main__":
    main()

