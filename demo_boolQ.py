
from QA_GEN import Pipeline, QADataset, QAPair
import json

def load_jsonl_data(filename: str, num_samples: int = None):
    """
    Load data from a .jsonl file and convert to QAPair format
    """
    qa_pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            item = json.loads(line)
            answer = "Yes" if item["answer"] else "No"
            qa_pair = QAPair(
                question=item["question"],
                answer=answer,
                context=item["passage"]
            )
            qa_pairs.append(qa_pair)
    print(f"Loaded {len(qa_pairs)} QA pairs from {filename}")
    return qa_pairs

def save_jsonl_data(filename: str, qa_pairs):
    """
    Save a list of QAPair objects to a .jsonl file in BoolQ format
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            # 將 answer 轉回 bool
            answer_bool = True if str(qa.answer).strip().lower() == "yes" else False
            item = {
                "question": qa.question,
                "passage": qa.context,
                "answer": answer_bool
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(qa_pairs)} QA pairs to {filename}")

def main():
    pipeline = Pipeline()
    dataset = QADataset(description="BoolQ jsonl for T5 question augmentation")
    # 載入 train_original.jsonl
    boolq_pairs = load_jsonl_data("train_original.jsonl")
    for qa_pair in boolq_pairs:
        dataset.add(qa_pair)
    print(f"\nInitial dataset size: {len(dataset)} pairs")
    print("\nSample of initial data:")
    for i, qa in enumerate(dataset):
        if i < 3:
            print(f"Q: {qa.question}")
            print(f"A: {qa.answer}")
            print(f"Context: {qa.context[:100]}...")
            print("-" * 50)
    config = {
        'auto_config': False,
        'methods_to_run': ["t5_question_augmentation"]
    }
    processed_dataset = pipeline.run(dataset, config)
    # 將擴增後的資料存成 result_augmented.jsonl
    save_jsonl_data("result_augmented.jsonl", processed_dataset)

if __name__ == '__main__':
    main()