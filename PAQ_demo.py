#demo.py
from QA_GEN import Pipeline, QADataset, QAPair
from typing import List, Dict, Any
import csv

def main():
    """
    Main demo function that replicates the original demo.py behavior
    with optional disk storage support
    """
    # Create pipeline
    pipeline = Pipeline(stages=["data_expansion"])
    
    # Create dataset - set use_disk=True for large datasets, False for small ones
    dataset = QADataset(
        description="A simple QA dataset, example: 'Q: What is AI?', 'A: Artificial Intelligence'"
    )
    
    # read data/psgs_w100.first_1000.tsv
    with open('data/psgs_w100.first_1000.tsv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            dataset.add(QAPair(context=row['text']))
    
    # Same config as original
    config = {
        'auto_config': False,
        'methods_to_run': ["paq_QA_generation"],
    }

    # Run pipeline - same as original
    processed_dataset = pipeline.run(dataset, config)
    
    # Print final results - same as original
    for qa in processed_dataset:
        print(qa)

if __name__ == '__main__':
    main()