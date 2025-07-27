"""
Main script for CREAM training following official methodology
Replaces the previous main.py with proper CREAM implementation
"""

import torch
import json
import os
from datasets import load_dataset
from cream_trainer import CREAMTrainer, CREAMConfig

def prepare_sft_data(dataset_name="imdb", num_samples=50):
    """Prepare SFT data from a dataset"""
    print(f"Preparing SFT data from {dataset_name}...")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    
    sft_data = []
    for item in dataset:
        # Convert to instruction format
        prompt = f"Analyze this text and provide a thoughtful response: {item['text'][:200]}..."
        response = f"This text appears to be {'positive' if item['label'] == 1 else 'negative'} in sentiment. It discusses various themes and presents {'optimistic' if item['label'] == 1 else 'pessimistic'} viewpoints."
        
        sft_data.append({
            "prompt": prompt,
            "response": response
        })
    
    return sft_data

def prepare_prompts(dataset_name="imdb", num_samples=20):
    """Prepare prompts for CREAM training"""
    print(f"Preparing prompts from {dataset_name}...")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=f"test[:{num_samples}]")
    
    prompts = []
    for item in dataset:
        prompt = f"What can you tell me about this text: {item['text'][:150]}..."
        prompts.append(prompt)
    
    return prompts

def main():
    """Main CREAM training pipeline"""
    print("Starting CREAM Training Pipeline")
    print("=" * 50)
    
    # Configuration
    config = CREAMConfig(
        model_name="microsoft/DialoGPT-medium",  # Start with smaller model
        max_length=512,
        num_responses=4,
        temperature=0.8,
        top_p=0.9,
        consistency_method="consistency_avg",
        dpo_beta=0.1,
        learning_rate=1e-6,
        epochs=1,
        batch_size=2  # Small batch for CPU
    )
    
    print(f"Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Responses per prompt: {config.num_responses}")
    print(f"  Consistency method: {config.consistency_method}")
    
    # Initialize CREAM trainer
    cream_trainer = CREAMTrainer(config)
    
    # Prepare data
    print("\nPreparing training data...")
    sft_data = prepare_sft_data("imdb", num_samples=20)  # Small dataset for testing
    prompts = prepare_prompts("imdb", num_samples=10)    # Small prompt set
    
    print(f"SFT data: {len(sft_data)} examples")
    print(f"Prompts: {len(prompts)} examples")
    
    # Run CREAM iterations
    current_checkpoint = None
    reference_checkpoint = None
    consistency_history = []
    
    num_iterations = 2  # Start with 2 iterations
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"CREAM ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        try:
            current_checkpoint, consistency = cream_trainer.full_cream_iteration(
                prompts=prompts,
                sft_data=sft_data if iteration == 0 else None,  # SFT only on first iteration
                current_checkpoint=current_checkpoint,
                reference_checkpoint=reference_checkpoint if iteration > 0 else None
            )
            
            consistency_history.append(consistency)
            
            # Update reference for next iteration
            reference_checkpoint = current_checkpoint
            
            print(f"\nIteration {iteration + 1} Results:")
            print(f"  Checkpoint: {current_checkpoint}")
            print(f"  Consistency: {consistency:.4f}")
            
        except Exception as e:
            print(f"Error in iteration {iteration + 1}: {e}")
            break
    
    # Final results
    print(f"\n{'='*60}")
    print("CREAM TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Total iterations: {len(consistency_history)}")
    print(f"Final model: {current_checkpoint}")
    print(f"Consistency history: {consistency_history}")
    
    if consistency_history:
        print(f"Average consistency: {sum(consistency_history)/len(consistency_history):.4f}")
        print(f"Final consistency: {consistency_history[-1]:.4f}")
    
    # Save training summary
    summary = {
        "config": {
            "model_name": config.model_name,
            "num_responses": config.num_responses,
            "consistency_method": config.consistency_method,
            "iterations": len(consistency_history)
        },
        "results": {
            "final_checkpoint": current_checkpoint,
            "consistency_history": consistency_history,
            "average_consistency": sum(consistency_history)/len(consistency_history) if consistency_history else 0
        }
    }
    
    with open("outputs/cream_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to: outputs/cream_training_summary.json")

def run_single_step_demo():
    """Demo of running individual CREAM steps"""
    print("Running CREAM Single Step Demo")
    print("=" * 40)
    
    config = CREAMConfig(
        model_name="microsoft/DialoGPT-medium",
        num_responses=3,
        batch_size=1
    )
    
    cream_trainer = CREAMTrainer(config)
    
    # Sample data
    sft_data = [
        {"prompt": "What is AI?", "response": "AI is artificial intelligence that helps computers think."},
        {"prompt": "How do robots work?", "response": "Robots use sensors and motors to interact with the world."}
    ]
    
    prompts = ["Tell me about machine learning.", "Explain computer vision."]
    
    try:
        # Step 1: SFT Training
        print("\n1. Running SFT training...")
        sft_checkpoint = cream_trainer.sft_training(sft_data, "outputs/demo_sft")
        
        # Step 2: Response Sampling
        print("\n2. Sampling responses...")
        sampling_file = cream_trainer.response_sampling(prompts, sft_checkpoint, "outputs/demo_sampling.json")
        
        # Step 3: Ranking (current model)
        print("\n3. Ranking with current model...")
        current_ranking = cream_trainer.response_ranking(sampling_file, sft_checkpoint, False, "current")
        
        # Step 4: Ranking (reference model - using same for demo)
        print("\n4. Ranking with reference model...")
        ref_ranking = cream_trainer.response_ranking(sampling_file, sft_checkpoint, True, "reference")
        
        # Step 5: Calculate consistency
        print("\n5. Calculating consistency...")
        dpo_file, consistency = cream_trainer.calculate_consistency(current_ranking, ref_ranking)
        
        print(f"\nDemo completed successfully!")
        print(f"Consistency score: {consistency:.4f}")
        print(f"DPO data saved to: {dpo_file}")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_single_step_demo()
    else:
        main()