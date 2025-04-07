from QA_GEN import Pipeline, QADataset, QAPair
from typing import List, Dict, Any
from QA_GEN.edge_builder import EdgeBuilder

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
        'auto_config': True
    }

    # Run pipeline
    processed_dataset = pipeline.run(dataset, config)
    for qa in processed_dataset:
        print(qa)

    edge_builder = EdgeBuilder(processed_dataset)

    # Build edges automatically
    edge_builder.build_cosine_similarity_edges(threshold=0.5)
    edge_builder.build_keyword_overlap_edges(threshold=2)

    # Add custom edges
    edge_builder.add_edge(source_id=1, target_id=2, method_name="custom_method")
    # Visualize all graphs separately
    edge_builder.visualize_all_graphs()

    # Visualize a combined graph
    edge_builder.visualize_combined_graph()
    stats = edge_builder.get_graph_statistics()
    print("Graph Statistics:", stats)

if __name__ == '__main__':
    main()