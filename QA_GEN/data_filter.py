#data_filter.py
from typing import List, Dict, Any, Set
from collections import deque
from .base import QADataset, QAPair
from .methods import Method
from .docker_manager import DockerMethodManager

class DataFilter:
    """
    Implements the filtering stage of the QA dataset pipeline.
    Uses registered filter methods to process connected subgraphs in the dataset.
    Supports both deletion and refinement filter methods.
    """
    
    def __init__(self):
        """
        Initialize the DataFilter with predefined method types.
        Methods are categorized as either deletion or refinement at initialization.
        """
        # Define method types
        self.DELETION_TYPE = "deletion"     # Methods that remove QA pairs
        self.REFINEMENT_TYPE = "refinement" # Methods that modify QA pairs
        
        # Predefined categorization of filter methods
        self.filter_method_types = {
            # Existing method
            "filter_similar_pairs": self.DELETION_TYPE,
            
            # New methods
            "filter_exact_substring": self.DELETION_TYPE,
            "refine_repeated_content": self.REFINEMENT_TYPE,
            
            # Add new methods here with their types
        }

        self.filter_data_types = [
            # Define the types of data that can be filtered first
            0, 1, 2, 3, 4, 5
            # Add more data types as needed
        ]
        self.docker_manager = DockerMethodManager()
    
    def filter_data(self, dataset: QADataset, registered_methods: Dict[str, bool], config: Dict[str, Any]) -> QADataset:
        """
        Filter the dataset by applying registered filter methods to connected subgraphs.
        Applies refinement methods first, then deletion methods.
        
        Args:
            dataset (QADataset): The dataset to filter
            registered_methods (Dict[str, bool]): Dictionary of method names and their registration status
            config (Dict[str, Any]): Configuration parameters for filtering
            
        Returns:
            QADataset: The filtered dataset
        """
        print("Starting the filtering process...")

        # Delete data that is still incomplete at this stage
        delete_ids = []
        for qa_pair in dataset:
            if qa_pair.classify_id() in self.filter_data_types:
                delete_ids.append(qa_pair.id)
        dataset.delete(delete_ids)
        print(f"Deleted {len(delete_ids)} incomplete QAPairs from the dataset")

        # Get all registered filter methods
        filter_methods = []
        for method_name, method_info in Method.get_methods().items():
            if 'data_filter' in method_info['applicable_stages'] and registered_methods.get(method_name, False):
                filter_methods.append(method_name)
                # Get method type (default to deletion for backward compatibility)
                method_type = self.filter_method_types.get(method_name, self.DELETION_TYPE)
                print(f"Using filter method: {method_name} (Type: {method_type})")
        
        if not filter_methods:
            print("No filter methods registered. Skipping filtering stage.")
            return dataset
            
        # Find connected components (subgraphs) using BFS
        subgraphs = self._find_connected_components(dataset)
        print(f"Found {len(subgraphs)} connected subgraphs in the dataset")
        
        # First apply refinement methods (which modify QA pairs)
        print("Applying refinement methods...")
        refinement_methods = [m for m in filter_methods if self.filter_method_types.get(m, self.DELETION_TYPE) == self.REFINEMENT_TYPE]
        for method_name in refinement_methods:
            method_func = Method.get_methods()[method_name]['func']
            print(f"Applying refinement method: {method_name}")
            
            # Process each subgraph with the refinement method
            for i, subgraph_ids in enumerate(subgraphs):
                print(f"  Processing subgraph {i+1}/{len(subgraphs)} with {len(subgraph_ids)} nodes")
                
                # Convert IDs to QAPairs
                subgraph_pairs = [dataset.get(id) for id in subgraph_ids]
                
                # Apply the refinement method
                if config.get('use_docker', True):
                    refined_pairs = self.docker_manager.execute_method_in_docker(method_name, subgraph_pairs, config)
                else:
                    refined_pairs = method_func(subgraph_pairs, config)
                
                # Update the QA pairs in the dataset
                for qa_pair in refined_pairs:
                    if qa_pair.id in dataset.data:
                        dataset.edit(qa_pair.id, qa_pair=qa_pair)
        
        # Then apply deletion methods (which remove QA pairs)
        print("Applying deletion methods...")
        deletion_methods = [m for m in filter_methods if self.filter_method_types.get(m, self.DELETION_TYPE) == self.DELETION_TYPE]
        
        # IDs to delete from the dataset
        ids_to_delete = set()
        
        # Process each subgraph with the deletion methods
        for i, subgraph_ids in enumerate(subgraphs):
            print(f"Processing subgraph {i+1}/{len(subgraphs)} with {len(subgraph_ids)} nodes")
            
            # Convert IDs to QAPairs
            subgraph_pairs = [dataset.get(id) for id in subgraph_ids]
            
            # Apply each deletion method to the subgraph
            for method_name in deletion_methods:
                method_func = Method.get_methods()[method_name]['func']
                print(f"  Applying {method_name} to subgraph {i+1}")
                
                # Apply the deletion method
                if config.get('use_docker', True):
                    filtered_pairs = self.docker_manager.execute_method_in_docker(method_name, subgraph_pairs, config)
                else:
                    filtered_pairs = method_func(subgraph_pairs, config)
                
                # Determine which pairs were removed
                removed_pairs = set(subgraph_pairs) - set(filtered_pairs)
                removed_ids = {pair.id for pair in removed_pairs}
                
                # Add removed IDs to the deletion set
                ids_to_delete.update(removed_ids)
                
                # Update subgraph_pairs for the next filter method
                subgraph_pairs = filtered_pairs
                
                print(f"  {method_name} removed {len(removed_ids)} pairs from subgraph {i+1}")
        
        # If there are any IDs to delete, apply the deletion
        if ids_to_delete:
            print(f"Deleting {len(ids_to_delete)} QAPairs from the dataset")
            print(f"IDs to delete: {ids_to_delete}")  
            dataset.delete(list(ids_to_delete))
        else:
            print("No QAPairs were deleted during filtering")
            
        return dataset
    
    def _find_connected_components(self, dataset: QADataset) -> List[Set[int]]:
        """
        Find all connected components (subgraphs) in the dataset using BFS.
        
        Args:
            dataset (QADataset): The dataset to analyze
            
        Returns:
            List[Set[int]]: A list of sets, where each set contains the IDs of nodes in a connected component
        """
        # Initialize a set to track visited nodes
        visited = set()
        components = []
        
        # Process each node in the dataset
        for qa_id in dataset.data.keys():
            # Skip if already visited
            if qa_id in visited:
                continue
                
            # Start a new component
            component = set()
            queue = deque([qa_id])
            visited.add(qa_id)
            
            # BFS to find all connected nodes
            while queue:
                current_id = queue.popleft()
                component.add(current_id)
                
                # Get current QAPair
                current_qa = dataset.get(current_id)
                if not current_qa:
                    continue
                    
                # Process all edges of the current node
                for _, target_id in current_qa.edges:
                    if target_id not in visited and target_id in dataset.data:
                        visited.add(target_id)
                        queue.append(target_id)
            
            # Add the component to the list of components
            components.append(component)
            
        return components

    def __del__(self):
        """
        Cleanup Docker resources when the object is destroyed
        """
        if hasattr(self, 'docker_manager'):
            self.docker_manager.cleanup()