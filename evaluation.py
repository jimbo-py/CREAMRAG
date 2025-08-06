from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall, answer_correctness, answer_similarity
from ragas import evaluate
from datasets import Dataset
import json
import pandas as pd
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import nltk
try:
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK not available, some metrics will be skipped")

def load_eval_outputs(file_path):
    """Load and transform eval_outputs.jsonl to RAGAS format"""
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                entry = json.loads(line)
                
                # Transform to RAGAS expected format
                ragas_entry = {
                    "question": entry["question"],
                    "answer": entry["generated_response"],
                    "contexts": entry.get("context_parts", []) or [],  # List of context strings
                    "ground_truth": entry["question"]  # Using question as placeholder - you might want actual ground truth
                }
                
                # Only include entries that have contexts for meaningful evaluation
                if ragas_entry["contexts"]:
                    data.append(ragas_entry)
    
    return data

def create_ragas_dataset(eval_data):
    """Create a Dataset object compatible with RAGAS"""
    df = pd.DataFrame(eval_data)
    return Dataset.from_pandas(df)

def comprehensive_evaluation_metrics(eval_data):
    """Comprehensive research-grade evaluation metrics for RAG systems"""
    print("Computing comprehensive evaluation metrics...")
    
    results = {}
    
    # Load sentence transformer for semantic similarity
    try:
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    except:
        semantic_model = None
        print("Warning: Sentence transformer not available for semantic metrics")
    
    # Basic statistics
    total_examples = len(eval_data)
    results['total_examples'] = total_examples
    
    # === CONTENT QUALITY METRICS ===
    
    # Answer length analysis
    answer_lengths = [len(entry['answer'].split()) for entry in eval_data]
    results['answer_length_stats'] = {
        'mean': np.mean(answer_lengths),
        'std': np.std(answer_lengths),
        'min': np.min(answer_lengths),
        'max': np.max(answer_lengths),
        'median': np.median(answer_lengths)
    }
    
    # Context utilization
    context_counts = [len(entry['contexts']) for entry in eval_data]
    context_lengths = [sum(len(ctx.split()) for ctx in entry['contexts']) for entry in eval_data]
    
    results['context_stats'] = {
        'avg_contexts_used': np.mean(context_counts),
        'avg_total_context_length': np.mean(context_lengths),
        'context_utilization_ratio': np.mean([len(entry['answer'].split()) / max(1, sum(len(ctx.split()) for ctx in entry['contexts'])) for entry in eval_data])
    }
    
    # === RELEVANCE METRICS ===
    
    # Keyword overlap (lexical similarity)
    lexical_similarities = []
    for entry in eval_data:
        question_words = set(entry['question'].lower().split())
        answer_words = set(entry['answer'].lower().split())
        
        if len(question_words) > 0:
            overlap = len(question_words.intersection(answer_words))
            similarity = overlap / len(question_words.union(answer_words))  # Jaccard similarity
            lexical_similarities.append(similarity)
    
    results['lexical_similarity'] = {
        'mean': np.mean(lexical_similarities) if lexical_similarities else 0,
        'std': np.std(lexical_similarities) if lexical_similarities else 0
    }
    
    # Semantic similarity using embeddings
    if semantic_model:
        semantic_similarities = []
        for entry in eval_data:
            q_emb = semantic_model.encode([entry['question']])
            a_emb = semantic_model.encode([entry['answer']])
            sim = cosine_similarity(q_emb, a_emb)[0][0]
            semantic_similarities.append(sim)
        
        results['semantic_similarity'] = {
            'mean': np.mean(semantic_similarities),
            'std': np.std(semantic_similarities),
            'distribution': {
                'high_similarity': sum(1 for s in semantic_similarities if s > 0.7) / len(semantic_similarities),
                'medium_similarity': sum(1 for s in semantic_similarities if 0.4 <= s <= 0.7) / len(semantic_similarities),
                'low_similarity': sum(1 for s in semantic_similarities if s < 0.4) / len(semantic_similarities)
            }
        }
    
    # === FAITHFULNESS METRICS ===
    
    # Context-answer overlap
    context_faithfulness = []
    for entry in eval_data:
        all_context_text = " ".join(entry['contexts']).lower()
        answer_text = entry['answer'].lower()
        
        # Count overlapping words
        context_words = set(all_context_text.split())
        answer_words = set(answer_text.split())
        
        if len(answer_words) > 0:
            overlap = len(context_words.intersection(answer_words))
            faithfulness = overlap / len(answer_words)
            context_faithfulness.append(faithfulness)
    
    results['context_faithfulness'] = {
        'mean': np.mean(context_faithfulness) if context_faithfulness else 0,
        'std': np.std(context_faithfulness) if context_faithfulness else 0
    }
    
    # === COMPLETENESS METRICS ===
    
    # Question answering completeness (simple heuristic)
    completeness_scores = []
    question_types = {'what': 0, 'how': 0, 'when': 0, 'where': 0, 'why': 0, 'who': 0}
    
    for entry in eval_data:
        question = entry['question'].lower()
        answer = entry['answer'].lower()
        
        # Identify question type
        q_type = 'other'
        for qword in question_types.keys():
            if qword in question:
                q_type = qword
                question_types[qword] += 1
                break
        
        # Simple completeness heuristic
        if 'what' in question and any(word in answer for word in ['is', 'are', 'definition', 'means']):
            completeness_scores.append(0.8)
        elif 'how' in question and any(word in answer for word in ['step', 'process', 'method', 'way']):
            completeness_scores.append(0.8)
        elif 'when' in question and any(word in answer for word in ['time', 'date', 'year', 'day']):
            completeness_scores.append(0.8)
        elif 'where' in question and any(word in answer for word in ['location', 'place', 'address', 'at', 'in']):
            completeness_scores.append(0.8)
        elif 'why' in question and any(word in answer for word in ['because', 'reason', 'due to', 'caused']):
            completeness_scores.append(0.8)
        elif 'who' in question and any(word in answer for word in ['person', 'people', 'name', 'individual']):
            completeness_scores.append(0.8)
        else:
            # General completeness based on answer length relative to question complexity
            q_complexity = len(entry['question'].split())
            a_length = len(entry['answer'].split())
            completeness_scores.append(min(1.0, a_length / max(1, q_complexity * 2)))
    
    results['completeness'] = {
        'mean': np.mean(completeness_scores) if completeness_scores else 0,
        'question_type_distribution': question_types
    }
    
    # === NLUE-style METRICS (if NLTK available) ===
    
    try:
        # BLEU scores (treating question as reference, answer as candidate)
        bleu_scores = []
        for entry in eval_data:
            reference = [entry['question'].split()]
            candidate = entry['answer'].split()
            if len(candidate) > 0:
                bleu = sentence_bleu(reference, candidate)
                bleu_scores.append(bleu)
        
        results['bleu_score'] = {
            'mean': np.mean(bleu_scores) if bleu_scores else 0,
            'std': np.std(bleu_scores) if bleu_scores else 0
        }
        
        # METEOR scores
        meteor_scores = []
        for entry in eval_data:
            try:
                meteor = meteor_score([entry['question']], entry['answer'])
                meteor_scores.append(meteor)
            except:
                continue
        
        if meteor_scores:
            results['meteor_score'] = {
                'mean': np.mean(meteor_scores),
                'std': np.std(meteor_scores)
            }
    
    except Exception as e:
        print(f"NLTK metrics not available: {e}")
    
    # === DIVERSITY METRICS ===
    
    # Answer diversity
    all_answers = [entry['answer'] for entry in eval_data]
    unique_answers = len(set(all_answers))
    results['answer_diversity'] = unique_answers / len(all_answers) if all_answers else 0
    
    # Vocabulary richness
    all_words = []
    for entry in eval_data:
        all_words.extend(entry['answer'].lower().split())
    
    if all_words:
        unique_words = len(set(all_words))
        total_words = len(all_words)
        results['vocabulary_richness'] = unique_words / total_words
        results['total_vocabulary_size'] = unique_words
    
    # === ERROR ANALYSIS ===
    
    # Empty/short answer detection
    empty_answers = sum(1 for entry in eval_data if len(entry['answer'].strip()) == 0)
    short_answers = sum(1 for entry in eval_data if len(entry['answer'].split()) < 3)
    
    results['error_analysis'] = {
        'empty_answers': empty_answers,
        'short_answers': short_answers,
        'empty_answer_rate': empty_answers / total_examples,
        'short_answer_rate': short_answers / total_examples
    }
    
    # === RETRIEVAL METRICS ===
    
    # Context relevance to question
    if semantic_model:
        context_relevance_scores = []
        for entry in eval_data:
            if entry['contexts']:
                q_emb = semantic_model.encode([entry['question']])
                for context in entry['contexts']:
                    c_emb = semantic_model.encode([context])
                    relevance = cosine_similarity(q_emb, c_emb)[0][0]
                    context_relevance_scores.append(relevance)
        
        if context_relevance_scores:
            results['context_relevance'] = {
                'mean': np.mean(context_relevance_scores),
                'std': np.std(context_relevance_scores)
            }
    
    return results

def setup_local_models():
    """Setup local models instead of OpenAI"""
    print("Setting up local models for RAGAS evaluation...")
    
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Use a lightweight local LLM
    try:
        # Try to use a small model that works well for evaluation
        model_name = "microsoft/DialoGPT-medium"  # Lightweight option
        
        # Create HuggingFace pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        print("✅ Local models loaded successfully")
        return embeddings, llm
        
    except Exception as e:
        print(f"❌ Error loading local models: {e}")
        print("Falling back to simpler evaluation...")
        return embeddings, None

def setup_ragas_with_local_models(embeddings, llm):
    """Configure RAGAS to use local models"""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    # Wrap the models for RAGAS
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    available_metrics = []
    
    if llm is not None:
        ragas_llm = LangchainLLMWrapper(llm)
        
        # Core RAGAS metrics
        answer_relevancy.embeddings = ragas_embeddings
        answer_relevancy.llm = ragas_llm
        available_metrics.append(answer_relevancy)
        
        faithfulness.llm = ragas_llm
        available_metrics.append(faithfulness)
        
        context_precision.llm = ragas_llm
        available_metrics.append(context_precision)
        
        # Additional RAGAS metrics
        try:
            context_recall.llm = ragas_llm
            available_metrics.append(context_recall)
        except:
            print("Context recall not available")
        
        try:
            answer_correctness.llm = ragas_llm
            answer_correctness.embeddings = ragas_embeddings
            available_metrics.append(answer_correctness)
        except:
            print("Answer correctness not available")
            
        try:
            answer_similarity.embeddings = ragas_embeddings
            available_metrics.append(answer_similarity)
        except:
            print("Answer similarity not available")
        
    else:
        # Use only embedding-based metrics if LLM fails
        answer_relevancy.embeddings = ragas_embeddings
        available_metrics.append(answer_relevancy)
    
    return available_metrics
    """Create a Dataset object compatible with RAGAS"""
    df = pd.DataFrame(eval_data)
    return Dataset.from_pandas(df)

def main():
    try:
        # Load evaluation outputs
        print("Loading eval_outputs.jsonl...")
        eval_data = load_eval_outputs("data/eval_outputs.jsonl")
        print(f"Loaded {len(eval_data)} evaluation entries")
        
        if not eval_data:
            print("❌ No evaluation data found or no entries with contexts!")
            return
        
        # Create dataset
        print("Creating RAGAS dataset...")
        dataset = create_ragas_dataset(eval_data)
        
        # Print sample to verify format
        print("\nSample entry:")
        print(f"Question: {dataset[0]['question']}")
        print(f"Answer: {dataset[0]['answer']}")
        print(f"Contexts: {len(dataset[0]['contexts'])} items")
        print(f"Ground Truth: {dataset[0]['ground_truth']}")
        
        # Setup local models
        embeddings, llm = setup_local_models()
        
        try:
            # Try RAGAS evaluation with local models
            print("\nAttempting RAGAS evaluation with local models...")
            metrics_to_use = setup_ragas_with_local_models(embeddings, llm)
            
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_use
            )
            
            print("\n" + "="*50)
            print("RAGAS Evaluation Results ✅")
            print("="*50)
            
            # Handle RAGAS result object properly
            if hasattr(result, 'to_pandas'):
                # Convert to pandas DataFrame and get mean scores
                df = result.to_pandas()
                print("Results DataFrame:")
                print(df.describe())
                
                # Get mean scores for each metric
                result_dict = {}
                for col in df.columns:
                    if col not in ['question', 'answer', 'contexts', 'ground_truth']:
                        mean_score = df[col].mean()
                        result_dict[col] = mean_score
                        print(f"{col}: {mean_score:.4f}")
                
            elif hasattr(result, '__dict__'):
                # Try to access as object attributes
                result_dict = {}
                for attr in dir(result):
                    if not attr.startswith('_') and not callable(getattr(result, attr)):
                        value = getattr(result, attr)
                        if isinstance(value, (int, float)):
                            result_dict[attr] = value
                            print(f"{attr}: {value:.4f}")
                        
            else:
                # Fallback: try to convert to dict or print as-is
                print("Raw result:", result)
                try:
                    result_dict = dict(result) if hasattr(result, 'keys') else {'result': str(result)}
                    for metric_name, score in result_dict.items():
                        print(f"{metric_name}: {score}")
                except:
                    result_dict = {'raw_result': str(result)}
            
            # Save results to file
            results_file = "ragas_evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
            print(f"\nResults saved to {results_file}")
            
        except Exception as ragas_error:
            print(f"⚠️ RAGAS evaluation failed: {ragas_error}")
            print("Running comprehensive evaluation instead...")
            
            # Run comprehensive evaluation instead
            result_dict = comprehensive_evaluation_metrics(eval_data)
            
            print("\n" + "="*50)
            print("Comprehensive Evaluation Results ✅")
            print("="*50)
            
            # Print results in organized sections
            def print_nested_dict(d, prefix=""):
                for key, value in d.items():
                    if isinstance(value, dict):
                        print(f"{prefix}{key}:")
                        print_nested_dict(value, prefix + "  ")
                    elif isinstance(value, (int, float)):
                        print(f"{prefix}{key}: {value:.4f}")
                    else:
                        print(f"{prefix}{key}: {value}")
            
            print_nested_dict(result_dict)
            
            # Save comprehensive results
            results_file = "comprehensive_evaluation_results.json"
            with open(results_file, 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                def convert_numpy_types(obj):
                    if isinstance(obj, dict):
                        return {key: convert_numpy_types(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                json.dump(convert_numpy_types(result_dict), f, indent=2)
            print(f"\nResults saved to {results_file}")
        
        # Create detailed report
        try:
            create_detailed_report(dataset, result_dict if 'result_dict' in locals() else {})
        except:
            create_detailed_report(dataset, {})
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Install required packages: pip install sentence-transformers transformers torch")
        print("2. Make sure eval_outputs.jsonl exists and has the correct format")
        print("3. Check available memory - local models require RAM")
        print("4. Try the simple evaluation fallback if RAGAS fails")
        
        # Try simple evaluation as last resort
        try:
            print("\nTrying comprehensive evaluation as fallback...")
            eval_data = load_eval_outputs("eval_outputs.jsonl")
            comprehensive_results = comprehensive_evaluation_metrics(eval_data)
            print("Comprehensive evaluation completed successfully!")
            
            # Print key metrics
            if 'semantic_similarity' in comprehensive_results:
                print(f"Semantic Similarity: {comprehensive_results['semantic_similarity']['mean']:.4f}")
            if 'context_faithfulness' in comprehensive_results:
                print(f"Context Faithfulness: {comprehensive_results['context_faithfulness']['mean']:.4f}")
            if 'completeness' in comprehensive_results:
                print(f"Answer Completeness: {comprehensive_results['completeness']['mean']:.4f}")
                
        except Exception as comprehensive_error:
            print(f"Even comprehensive evaluation failed: {comprehensive_error}")

def create_detailed_report(dataset, results):
    """Create a detailed evaluation report"""
    report_file = "evaluation_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RAGAS Evaluation Report\n")
        f.write("="*50 + "\n\n")
        
        f.write("Overall Scores:\n")
        f.write("-"*20 + "\n")
        for metric_name, score in results.items():
            f.write(f"{metric_name}: {score:.4f}\n")
        f.write("\n")
        
        f.write("Dataset Statistics:\n")
        f.write("-"*20 + "\n")
        f.write(f"Total examples: {len(dataset)}\n")
        
        # Calculate context statistics
        context_lengths = [len(example['contexts']) for example in dataset]
        f.write(f"Average contexts per example: {sum(context_lengths)/len(context_lengths):.2f}\n")
        f.write(f"Max contexts: {max(context_lengths)}\n")
        f.write(f"Min contexts: {min(context_lengths)}\n")
        
        # Calculate answer length statistics
        answer_lengths = [len(example['answer'].split()) for example in dataset]
        f.write(f"Average answer length: {sum(answer_lengths)/len(answer_lengths):.2f} words\n")
        
        f.write("\nSample Examples:\n")
        f.write("-"*20 + "\n")
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Question: {example['question']}\n")
            f.write(f"Answer: {example['answer'][:100]}...\n")
            f.write(f"Number of contexts: {len(example['contexts'])}\n")
    
    print(f"Detailed report saved to {report_file}")

if __name__ == "__main__":
    main()
