#data_fusioner.py
from typing import List, Dict, Any, Callable, Set, Tuple
from collections import defaultdict
import random
from .base import QAPair, QADataset
from .methods import Method

class DataFusioner:
    """
    Handles data fusion for QA pairs using different retrieval methods and 
    registered fusion techniques.
    """

    def __init__(self):
        """
        Initialize the DataFusioner with registries for retrievers and fusion method types.
        """
        # Registry of retrievers for each fusion method
        # Format: {method_name: retriever_function}
        self.retrievers = {}
        
        # Registry to track whether a fusion method is combination or permutation
        # Format: {method_name: "combination" or "permutation"}
        self.method_types = {}
        
        # Track processed pair combinations to avoid duplicates
        self.processed_combinations = set()
        
        # Auto-register built-in retrievers
        self._register_default_retrievers()
        # Auto-register method types
        self._register_default_method_types()

    def _register_default_retrievers(self):
        """Register the built-in retrievers automatically"""
        self.register_retriever("cosine_similarity", self.cosine_similarity_retriever)
        self.register_retriever("keyword_overlap", self.keyword_overlap_retriever)
        self.register_retriever("weighted_edge", self.weighted_edge_retriever) 
        self.register_retriever("type_based", self.type_based_retriever)
        self.register_retriever("few_shot_fusion", self.few_shot_retriever)
        self.register_retriever("cross_domain_fusion", self.cross_domain_retriever)
        
    def _register_default_method_types(self):
        """Register the default method types (combination vs permutation)"""
        # Existing methods are all combination-based (order doesn't matter)
        self.register_method_type("few_shot_fusion", "combination")
        self.register_method_type("cross_domain_fusion", "combination")
        # Add more methods as they are created

    def register_retriever(self, method_name: str, retriever_function: Callable):
        """
        Register a retriever function for a specific fusion method.
        
        Args:
            method_name (str): Name of the fusion method
            retriever_function (Callable): Function that retrieves relevant QA pairs
        """
        self.retrievers[method_name] = retriever_function

    def register_method_type(self, method_name: str, method_type: str):
        """
        Register whether a fusion method is combination or permutation based.
        
        Args:
            method_name (str): Name of the fusion method
            method_type (str): Either "combination" or "permutation"
        """
        if method_type not in ["combination", "permutation"]:
            raise ValueError("Method type must be either 'combination' or 'permutation'")
        self.method_types[method_name] = method_type

    def get_combination_key(self, method_name: str, pair_ids: List[int]) -> str:
        """
        Generate a unique key for a combination of pair IDs based on method type.
        
        Args:
            method_name (str): Name of the fusion method
            pair_ids (List[int]): List of pair IDs in the combination
            
        Returns:
            str: Unique key for this combination
        """
        # Check if the method is registered as combination or permutation
        method_type = self.method_types.get(method_name, "combination")  # Default to combination
        
        if method_type == "combination":
            # For combination methods, order doesn't matter, so sort IDs
            return f"{method_name}-" + "-".join(map(str, sorted(pair_ids)))
        else:
            # For permutation methods, order matters, so preserve order
            return f"{method_name}-" + "-".join(map(str, pair_ids))

    def retrieve_relevant_pairs(self, dataset: QADataset, source_id: int, method_name: str) -> List[QAPair]:
        """
        Retrieve relevant QA pairs for fusion based on the registered retriever.
        
        Args:
            dataset (QADataset): The dataset to search in
            source_id (int): ID of the source QA pair
            method_name (str): Name of the fusion method
            max_pairs (int): Maximum number of pairs to retrieve
            
        Returns:
            List[QAPair]: List of relevant QA pairs
        """
        if method_name not in self.retrievers:
            print(f"No retriever registered for method: {method_name}")
            return []
            
        retriever = self.retrievers[method_name]
        return retriever(dataset, source_id)

    def fuse_data(self, dataset: QADataset, fusion_methods: List[str], 
                  config: Dict[str, Any] = None) -> QADataset:
        """
        Fuse data using registered methods and retrievers.
        
        Args:
            dataset (QADataset): Original dataset
            fusion_methods (List[str]): List of fusion methods to use
            config (Dict[str, Any]): Configuration parameters
            
        Returns:
            QADataset: Dataset with fused pairs added
        """
        if config is None:
            config = {}
            
        # Create a new dataset to avoid modifying the original
        fused_dataset = dataset.copy()
        self.processed_combinations = set()
        
        # Filter methods to only those registered in Method registry
        available_methods = {
            method_name: Method.get_methods()[method_name]
            for method_name in fusion_methods
            if method_name in Method.get_methods() and method_name in self.retrievers
        }
        
        if not available_methods:
            print("No valid fusion methods found.")
            return fused_dataset
            
        # For each QA pair in the dataset, try each fusion method
        for source_pair in dataset:
            for method_name, method_info in available_methods.items():
                print(f"Applying fusion method: {method_name} to pair {source_pair.id}")
                
                # Get relevant pairs using the retriever for this method
                relevant_pairs = self.retrieve_relevant_pairs(
                    dataset, source_pair.id, method_name
                )
                
                if not relevant_pairs:
                    continue
                    
                # Add the source pair to ensure it's included in fusion
                fusion_input = [source_pair] + relevant_pairs
                
                # Create a unique combination ID to avoid duplicates
                pair_ids = [pair.id for pair in fusion_input]
                combination_key = self.get_combination_key(method_name, pair_ids)
                
                if combination_key in self.processed_combinations:
                    print(f"Skipping duplicate combination: {combination_key}")
                    continue
                    
                self.processed_combinations.add(combination_key)
                
                # Apply the fusion method
                fused_pairs = method_info['func'](fusion_input, config)
                
                # Add edge information and add to dataset
                for fused_pair in fused_pairs:
                    # Add edges from all source pairs to the fused pair
                    for input_pair in fusion_input:
                        fused_pair.add_edge(input_pair.id, f"fused")
                    
                    fused_dataset.add(fused_pair)
        
        return fused_dataset
    
    def _register_default_retrievers(self):
        """Register the built-in retrievers automatically"""
        self.register_retriever("cosine_similarity", self.cosine_similarity_retriever)
        self.register_retriever("keyword_overlap", self.keyword_overlap_retriever)
        self.register_retriever("weighted_edge", self.weighted_edge_retriever) 
        self.register_retriever("type_based", self.type_based_retriever)
        self.register_retriever("few_shot_fusion", self.few_shot_retriever)
        self.register_retriever("cross_domain_fusion", self.cross_domain_retriever) 
    
    # All the existing methods...
    def cosine_similarity_retriever(self, dataset: QADataset, source_id: int, max_pairs: int = 5) -> List[QAPair]:
        """
        Retrieve pairs using cosine similarity edges.
        """
        source_pair = dataset.get(source_id)
        if not source_pair:
            return []
            
        # Get all target IDs connected via cosine similarity
        target_ids = source_pair.get_edges_by_method("cosine_similarity")
        
        # Limit to max_pairs
        if len(target_ids) > max_pairs:
            target_ids = random.sample(target_ids, max_pairs)
            
        return [dataset.get(tid) for tid in target_ids if dataset.get(tid)]

    def keyword_overlap_retriever(self, dataset: QADataset, source_id: int, max_pairs: int = 5) -> List[QAPair]:
        """
        Retrieve pairs using keyword overlap edges.
        """
        source_pair = dataset.get(source_id)
        if not source_pair:
            return []
            
        # Get all target IDs connected via keyword overlap
        target_ids = source_pair.get_edges_by_method("keyword_overlap")
        
        # Limit to max_pairs
        if len(target_ids) > max_pairs:
            target_ids = random.sample(target_ids, max_pairs)
            
        return [dataset.get(tid) for tid in target_ids if dataset.get(tid)]

    def weighted_edge_retriever(self, dataset: QADataset, source_id: int, max_pairs: int = 5,
                              edge_weights: Dict[str, float] = None) -> List[QAPair]:
        """
        Retrieve pairs using a weighted combination of different edge types.
        """
        # Implementation as before
        source_pair = dataset.get(source_id)
        if not source_pair:
            return []
            
        if edge_weights is None:
            # Default weights
            edge_weights = {
                "cosine_similarity": 0.5,
                "keyword_overlap": 0.3,
                "expanded": 0.1,
                "reduced": 0.1
            }
            
        # Track scores for each target ID
        scores = defaultdict(float)
        
        # Calculate scores based on edge weights
        for edge_type, weight in edge_weights.items():
            target_ids = source_pair.get_edges_by_method(edge_type)
            for tid in target_ids:
                scores[tid] += weight
                
        # Sort by score and take top max_pairs
        sorted_targets = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_pairs]
        
        # Return the QA pairs
        return [dataset.get(tid) for tid, _ in sorted_targets if dataset.get(tid)]

    def type_based_retriever(self, dataset: QADataset, source_id: int, max_pairs: int = 5,
                            target_types: List[int] = None) -> List[QAPair]:
        """
        Retrieve pairs of specific QA types.
        """
        # Implementation as before
        if target_types is None:
            target_types = [7]  # Default to Complete QA pairs
            
        # Get all pairs of the target types
        candidates = []
        for qa_pair in dataset:
            if qa_pair.id != source_id and qa_pair.classify_id() in target_types:
                candidates.append(qa_pair)
                
        # Randomly select up to max_pairs
        if len(candidates) > max_pairs:
            return random.sample(candidates, max_pairs)
        return candidates

    def few_shot_retriever(self, dataset: QADataset, source_id: int, max_pairs: int = 5) -> List[QAPair]:
        """
        Retrieve pairs with a similar question/answer structure for few-shot learning.
        
        Args:
            dataset (QADataset): The dataset to search in
            source_id (int): ID of the source QA pair
            max_pairs (int): Maximum number of pairs to retrieve
            
        Returns:
            List[QAPair]: List of QA pairs for few-shot learning
        """
        source_pair = dataset.get(source_id)
        if not source_pair:
            return []
        
        # First priority: pairs connected by cosine similarity or keyword overlap
        target_ids = set()
        target_ids.update(source_pair.get_edges_by_method("cosine_similarity"))
        target_ids.update(source_pair.get_edges_by_method("keyword_overlap"))
        
        # If we don't have enough, find pairs with similar question types
        if len(target_ids) < max_pairs:
            # Try to find QA pairs with similar structure
            source_type = source_pair.classify_id()
            for qa_pair in dataset:
                if qa_pair.id != source_id and qa_pair.classify_id() == source_type:
                    target_ids.add(qa_pair.id)
                    if len(target_ids) >= max_pairs:
                        break
        
        # Convert to list and limit to max_pairs
        target_ids_list = list(target_ids)
        if len(target_ids_list) > max_pairs:
            target_ids_list = random.sample(target_ids_list, max_pairs)
        
        return [dataset.get(tid) for tid in target_ids_list if dataset.get(tid)]
    
    def cross_domain_retriever(self, dataset: QADataset, source_id: int, max_pairs: int = 1) -> List[QAPair]:
        """
        Retrieve a QA pair that is somewhat related but from a different domain
        or context compared to the source pair.
        
        Args:
            dataset (QADataset): The dataset to search in
            source_id (int): ID of the source QA pair
            max_pairs (int): Maximum number of pairs to retrieve (usually 1 for this method)
            
        Returns:
            List[QAPair]: List of moderately related QA pairs
        """
        source_pair = dataset.get(source_id)
        if not source_pair:
            return []
        
        # Create a set of keywords from the source pair
        source_keywords = set()
        if source_pair.context_keywords:
            source_keywords.update(source_pair.context_keywords)
        if source_pair.question_keywords:
            source_keywords.update(source_pair.question_keywords)
        if source_pair.answer_keywords:
            source_keywords.update(source_pair.answer_keywords)
        
        # Find candidates with some keyword overlap, but not too much
        candidates = []
        for qa_pair in dataset:
            if qa_pair.id == source_id:
                continue
                
            # Skip pairs with missing components
            if not qa_pair.context or not qa_pair.question or not qa_pair.answer:
                continue
                
            # Create a set of keywords from the candidate pair
            candidate_keywords = set()
            if qa_pair.context_keywords:
                candidate_keywords.update(qa_pair.context_keywords)
            if qa_pair.question_keywords:
                candidate_keywords.update(qa_pair.question_keywords)
            if qa_pair.answer_keywords:
                candidate_keywords.update(qa_pair.answer_keywords)
                
            # Calculate overlap
            overlap = len(source_keywords.intersection(candidate_keywords))
            
            # We want some overlap, but not too much
            # Adjust these thresholds based on your data
            if 1 <= overlap <= 2:  # Moderate overlap: 1-2 keywords in common
                candidates.append((qa_pair, overlap))
        
        # Sort by overlap (ascending, to get less similar pairs first)
        candidates.sort(key=lambda x: x[1])
        
        # Select candidates
        selected = [pair for pair, _ in candidates[:max_pairs]]
        
        # If we couldn't find moderately related pairs, fall back to random selection
        if not selected and len(dataset) > 1:
            all_pairs = [p for p in dataset if p.id != source_id and p.context and p.question and p.answer]
            if all_pairs:
                selected = random.sample(all_pairs, min(max_pairs, len(all_pairs)))
        
        return selected