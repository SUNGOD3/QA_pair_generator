from data import QADataset, QAPair

dataset = QADataset()
dataset.add(QAPair(question="What is AI?", answer="Artificial Intelligence"))
dataset.add(QAPair(question="What is ML?", answer="Machine Learning"))
dataset.add(QAPair(context="AI is a subset of Computer Science."))

print("Original dataset:")
for qa in dataset:
    print(qa)

print("\nExpanded dataset:")
config = {
    "init_setup": {"param1": True},
    "data_expansion": {"extract_key_sentences": True, "generate_fill_in_blank": True, "generate_summary_qa": True, "generate_causal_qa": True, "generate_logical_qa": True,
        "adjacency_matrix": { # All true by default
            1: [False, True, True, True, True, True, True, True],
            2: [True, False, True, True, True, True, True, True],
            3: [True, True, False, True, True, True, True, True],
            4: [True, True, True, False, True, True, True, True],
            5: [True, True, True, True, False, True, True, True],
            6: [True, True, True, True, True, False, True, True],
            7: [True, True, True, True, True, True, False, True],
            8: [True, True, True, True, True, True, True, False]
        }
    },
    "build_knowledge_graph": {"use_graph": True},
    "combination_segmentation": {"split_long_text": True},
    "filter": {"remove_duplicates": True},
    "data_augmentation": {"add_noise": True}
}

dataset.run_pipeline(config)

for qa in dataset:
    print(qa)
