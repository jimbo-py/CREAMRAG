from datasets import load_dataset
import os


save_dir = "qa_data"
os.makedirs(save_dir, exist_ok=True)


datasets_to_download = [
    ("hotpot_qa", "hotpotqa/hotpot_qa", "distractor", "train"),
    ("natural_questions", "sentence-transformers/natural-questions", None, "train"),
]

for name, hf_id, config_name, split in datasets_to_download:
    print(f"📥 Downloading {split} split of {hf_id} from Hugging Face...")

    dataset = None
    try:
        
        if config_name:
            dataset = load_dataset(hf_id, config_name, split=split)
        else:
            dataset = load_dataset(hf_id, split=split)
    except Exception as e1:
        print(f"⚠️  First attempt failed: {e1}")
        try:
           
            if config_name:
                full_dataset = load_dataset(hf_id, config_name)
            else:
                full_dataset = load_dataset(hf_id)
            if split in full_dataset:
                dataset = full_dataset[split]
            else:
               
                first_split = list(full_dataset.keys())[0]
                print(f"⚠️  No '{split}' split found, using '{first_split}' split instead")
                dataset = full_dataset[first_split]
        except Exception as e2:
            print(f"❌ Failed to load {split} split for {hf_id}: {e2}")
            continue

    if dataset is None:
        print(f"❌ Could not load {hf_id}")
        continue

    
    save_path = os.path.join(save_dir, f"{name}_train.jsonl")
    dataset.to_json(save_path, orient="records", lines=True)
    print(f"✅ Saved {name} training split to {save_path}")

print("\n🎉 All available training splits downloaded successfully into qa_data/")
