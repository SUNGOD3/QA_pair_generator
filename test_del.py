from QA_GEN import Pipeline, QADataset, QAPair
from typing import List, Dict, Any

def test_delete_function():
    """
    Test the delete function to ensure it maintains data integrity.
    """
    # Create test dataset
    dataset = QADataset()
    
    # Add some QA pairs
    qa1 = QAPair(context="Context 1", question="Question 1", answer="Answer 1")
    qa2 = QAPair(context="Context 2", question="Question 2", answer="Answer 2")
    qa3 = QAPair(context="Context 3", question="Question 3", answer="Answer 3")
    qa4 = QAPair(context="Context 4", question="Question 4", answer="Answer 4")
    qa5 = QAPair(context="Context 5", question="Question 5", answer="Answer 5")
    
    dataset.add(qa1)  # ID: 1
    dataset.add(qa2)  # ID: 2
    dataset.add(qa3)  # ID: 3
    dataset.add(qa4)  # ID: 4
    dataset.add(qa5)  # ID: 5
    
    # Add some edges
    dataset.add_edge(1, 2, "related")
    dataset.add_edge(1, 3, "similar")
    dataset.add_edge(2, 1, "related")
    dataset.add_edge(2, 3, "follows")
    dataset.add_edge(3, 4, "related")
    dataset.add_edge(4, 5, "follows")
    dataset.add_edge(5, 1, "circular")
    
    print("Original dataset:")
    for qa in dataset:
        print(f"ID: {qa.id}, Edges: {qa.edges}")
    
    # Delete QA pairs with IDs 2 and 4
    dataset.delete([2, 4])
    
    print("\nAfter deletion:")
    for qa in dataset:
        print(f"ID: {qa.id}, Edges: {qa.edges}")
    
    # Check data integrity
    print("\nVerifying data integrity:")
    all_ids = set(qa.id for qa in dataset)
    print(f"All IDs in dataset: {all_ids}")
    
    # Check if IDs are sequential
    expected_ids = set(range(1, len(dataset) + 1))
    print(f"Expected IDs: {expected_ids}")
    print(f"IDs are sequential: {all_ids == expected_ids}")
    
    # Check if all edge targets exist
    all_targets = set()
    for qa in dataset:
        for _, target_id in qa.edges:
            all_targets.add(target_id)
    
    print(f"All edge targets: {all_targets}")
    print(f"All targets exist in dataset: {all_targets.issubset(all_ids)}")
    
    # Check content preservation
    print("\nContent verification:")
    for qa in dataset:
        print(f"ID: {qa.id}, Content: {qa.context[:10]}...")

# Example execution
if __name__ == "__main__":
    test_delete_function()