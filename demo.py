#demo.py
from QA_GEN import Pipeline, QADataset, QAPair
from typing import List, Dict, Any

def main():
    """
    Main demo function that replicates the original demo.py behavior
    with optional disk storage support
    """
    # Create pipeline
    pipeline = Pipeline()
    
    # Create dataset - set use_disk=True for large datasets, False for small ones
    dataset = QADataset(
        description="A simple QA dataset, example: 'Q: What is AI?', 'A: Artificial Intelligence'"
    )
    
    # Add data same as original
    dataset.add(QAPair(question="What is AI?", answer="Artificial Intelligence"))
    dataset.add(QAPair(question="What is ML?", answer="Machine Learning"))
    dataset.add(QAPair(context="AI is a subset of Computer Science."))
    
    # Print initial dataset
    for qa in dataset:
        print(qa)
    
    # Same config as original
    config = {
        'auto_config': False,
        'methods_to_run': ["paraphrase_augmentation", "key_sentences_question", "context_to_qa", "paq_answer_extraction"],
    }

    # Run pipeline - same as original
    processed_dataset = pipeline.run(dataset, config)
    
    # Print final results - same as original
    for qa in processed_dataset:
        print(qa)

if __name__ == '__main__':
    main()