from QA_GEN import Pipeline, QADataset, QAPair
from typing import List, Dict, Any

def main():
    # In main script
    pipeline = Pipeline()
    dataset = QADataset(description="A simple QA dataset, example: 'Q: What is AI?', 'A: Artificial Intelligence'")
    dataset.add(QAPair(question="What is AI?", answer="Artificial Intelligence"))
    dataset.add(QAPair(question="What is ML?", answer="Machine Learning"))
    dataset.add(QAPair(context="AI is a subset of Computer Science."))
    for qa in dataset:
        print(qa)
    config = {
        'auto_config': False,
        'methods_to_run': ["paraphrase_augmentation", "key_sentences_question"],
    }

    # Run pipeline
    processed_dataset = pipeline.run(dataset, config)
    for qa in processed_dataset:
        print(qa)

if __name__ == '__main__':
    main()