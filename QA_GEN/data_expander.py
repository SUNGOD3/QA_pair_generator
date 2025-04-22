#data_expander.py
import copy
from typing import List, Dict, Any
from .base import QAPair, QADataset
from .methods import Method

class DataExpander:
    """
    Handles data expansion for QA pairs with minimal and efficient design.
    """

    def __init__(self):
        """
        Since the data expansion method has more detailed classification, another dictionary is needed to record
        {method_name: {pair_list [{from, to}, {from, to}, ...]}}
        "Document": 1,
        "Question": 2,
        "Ground Truth": 4,
        "Context + Question": 3,
        "Context + Ground Truth": 5,
        "QA Pair": 6,
        "Complete QA": 7
        """
        self.data_expansion_methods = {
            "context_to_qa": [(1, 7)],
            "key_sentences_question": [(1, 3)],
            "generate_summary_question": [(1, 3)],
            "generate_summary_qa": [(1, 7)],
            "generate_fill_in_blank": [(1, 7)],
        }

    def expand_data(self, dataset: QADataset, config) -> QADataset:
        """
        Expand dataset based on configuration.
        
        Args:
            dataset (QADataset): Original dataset
            config (Dict[str, Any]): Expansion configuration
        
        Returns:
            QADataset: Expanded dataset
        """
        expanded_dataset = dataset.copy()
        for qa_pair in dataset:
            current_type = qa_pair.classify_id()
            for target_type in range(1, 8):
                if current_type == target_type:
                    continue
                # Subset type: reduce
                if (current_type & target_type) == target_type:
                    reduced_pair = DataExpander._reduce_to_type(qa_pair, target_type)
                    expanded_dataset.add(reduced_pair)
                    continue
                
                edge_pair = (current_type, target_type)
                # Complex expansion
                for method_name in config:
                    if method_name in Method.get_methods() and self.data_expansion_methods.get(method_name):
                        method_info = Method.get_methods()[method_name]
                        pair_list = self.data_expansion_methods[method_name]
                        if edge_pair in pair_list:
                            print("Using Method name:", method_name)
                            expanded_entry = method_info['func']([qa_pair], config)
                            for entry in expanded_entry:
                                entry.add_edge(qa_pair.id, "expanded")
                                expanded_dataset.add(entry)
        
        return expanded_dataset

    def _reduce_to_type(qa_pair: QAPair, target_type: int) -> QAPair:
        """
        Reduce QA pair to target type.
        
        Args:
            qa_pair (QAPair): Original QA pair
            target_type (int): Target type to reduce to
        
        Returns:
            QAPair: Reduced QA pair
        """
        new_pair = copy.deepcopy(qa_pair)
        
        if not (target_type & 1):  # Context
            new_pair.context = None
        if not (target_type & 2):  # Question
            new_pair.question = None
        if not (target_type & 4):  # Answer
            new_pair.answer = None
        
        new_pair.add_edge(qa_pair.id, "reduced")
        return new_pair
    