
from datasets import load_dataset
import json

def main():
    dataset = load_dataset("HanningZhang/OpenGenAlign", split="train")

    with open("data/corpus.jsonl", "w", encoding="utf-8") as f_out:
        doc_id = 1
        for example in dataset:
            # Extract all 'content' strings inside the 'chosen' list
            chosen_list = example.get("chosen", [])
            for item in chosen_list:
                text = item.get("content", "").strip()
                if text:
                    line = {
                        "id": f"doc{doc_id}",
                        "text": text
                    }
                    f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
                    doc_id += 1

    print("corpus.jsonl generated successfully.")

if __name__ == "__main__":
    main()

