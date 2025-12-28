#!/usr/bin/env python3
"""
Combine all JSONL files from qa_data directory into a single combined file.
Uses the same normalization and deduplication logic as download_datasets.py
"""

import argparse
import hashlib
import json
import os
import random
from collections import OrderedDict
from glob import glob

# Copy normalization functions from download_datasets.py
def normalize_answers_from_example(ex):
    """Try common patterns to extract answers as list of strings."""
    if ex is None:
        return []
    if "answers" in ex:
        a = ex["answers"]
        if isinstance(a, dict):
            if "text" in a and isinstance(a["text"], (list, tuple)):
                return [t for t in a["text"] if t is not None]
            if "answer" in a and isinstance(a["answer"], str):
                return [a["answer"]]
        elif isinstance(a, (list, tuple)):
            out = []
            for el in a:
                if isinstance(el, str):
                    out.append(el)
                elif isinstance(el, dict):
                    if "text" in el:
                        out.append(el["text"])
                    elif "answer" in el:
                        out.append(el["answer"])
            return [t for t in out if t is not None]
    for f in ("answer_text", "answer", "answers_text", "answer_strings"):
        if f in ex:
            v = ex[f]
            if isinstance(v, list):
                return [x for x in v if isinstance(x, str)]
            elif isinstance(v, str):
                return [v]
    if "annotations" in ex and isinstance(ex["annotations"], (list, tuple)) and len(ex["annotations"])>0:
        ann = ex["annotations"][0]
        if "short_answers" in ann and isinstance(ann["short_answers"], (list, tuple)):
            out = []
            for sa in ann["short_answers"]:
                if isinstance(sa, dict) and "text" in sa:
                    out.append(sa["text"])
            if out:
                return out
        if "long_answer" in ann and isinstance(ann["long_answer"], dict) and "text" in ann["long_answer"]:
            return [ann["long_answer"]["text"]]
    for k,v in ex.items():
        if isinstance(v, str) and len(v) > 0 and k.lower().startswith("answer"):
            return [v]
    return []

def get_field_safe(ex, *fields):
    for f in fields:
        if f in ex and ex[f] is not None:
            return ex[f]
    return None

def normalize_example(dataset_name, ex):
    """Return dict: {question, context, answers(list), source}"""
    q = get_field_safe(ex, "question", "Question", "query", "question_text", "question_str")
    context = get_field_safe(ex, "context", "contexts", "document", "documents", "passage", "text", "article", "wiki_context")
    if context is None and "long_answer" in ex and isinstance(ex["long_answer"], dict) and "text" in ex["long_answer"]:
        context = ex["long_answer"]["text"]
    answers = normalize_answers_from_example(ex)
    if q is None:
        return None
    if context is None:
        context = ""
    if not isinstance(answers, (list, tuple)):
        answers = []
    answers_flat = []
    for a in answers:
        if isinstance(a, str):
            s = a.strip()
            if s:
                answers_flat.append(s)
        elif isinstance(a, (list, tuple)):
            for x in a:
                if isinstance(x, str) and x.strip():
                    answers_flat.append(x.strip())
    answers_flat = list(OrderedDict.fromkeys(answers_flat))
    return {
        "question": q.strip() if isinstance(q, str) else str(q),
        "context": context.strip() if isinstance(context, str) else str(context),
        "answers": answers_flat,
        "source": dataset_name
    }

def dedupe_examples(items):
    seen = set()
    out = []
    for it in items:
        key = it.get("question","") + "||" + it.get("context","") + "||" + "|".join(it.get("answers",[]))
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(it)
    return out

def combine_qa_files(data_dir, max_combined=100000, seed=42):
    """Combine all JSONL files in data_dir into a single combined file."""
    random.seed(seed)
    
    # Get all jsonl files except combined ones
    jsonl_files = [f for f in glob(os.path.join(data_dir, "*.jsonl")) 
                   if not os.path.basename(f).startswith("combined")]
    
    print(f"Found {len(jsonl_files)} JSONL files to combine:")
    for f in jsonl_files:
        print(f"  - {os.path.basename(f)}")
    
    combined = []
    per_dataset_counts = {}
    
    for jsonl_file in jsonl_files:
        dataset_name = os.path.basename(jsonl_file).replace(".jsonl", "").replace("_train", "")
        print(f"\n=== Processing {dataset_name} ===")
        
        dataset_examples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    ex = json.loads(line.strip())
                    # Check if already normalized (has question, context, answers, source)
                    if all(k in ex for k in ["question", "context", "answers", "source"]):
                        # Already normalized, use as-is
                        norm = ex
                    else:
                        # Need normalization
                        norm = normalize_example(dataset_name, ex)
                    if norm is not None:
                        dataset_examples.append(norm)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        # Dedupe within dataset
        dataset_examples = dedupe_examples(dataset_examples)
        per_dataset_counts[dataset_name] = len(dataset_examples)
        print(f"  -> {dataset_name}: {len(dataset_examples)} normalized examples")
        
        combined.extend(dataset_examples)
    
    print("\n=== Combined before dedupe ===")
    print(f"Total collected examples (raw combined): {len(combined)}")
    
    # Dedupe across datasets
    combined = dedupe_examples(combined)
    print(f"After cross-dataset deduplication: {len(combined)}")
    
    # Shuffle
    random.shuffle(combined)
    
    # Sample down if needed
    if len(combined) > max_combined:
        print(f"Sampling down to {max_combined} examples (seed={seed})")
        combined = combined[:max_combined]
    else:
        print(f"Total combined {len(combined)} <= max_combined {max_combined}; using all")
    
    # Write combined file
    combined_path = os.path.join(data_dir, f"combined_{len(combined)}.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n✅ Wrote combined file: {combined_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Per-dataset counts:")
    for name, count in per_dataset_counts.items():
        print(f"  {name}: {count}")
    print(f"Final combined count: {len(combined)}")
    
    return combined_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./qa_data")
    parser.add_argument("--max_combined", type=int, default=100000, 
                       help="Target number of examples in combined file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    combine_qa_files(args.data_dir, args.max_combined, args.seed)


