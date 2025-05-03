#data_filter.py
from typing import List, Dict, Any, Set
from collections import deque
from .base import QADataset, QAPair
from .methods import Method

class DataFilter:
    """
    Implements the filtering stage of the QA dataset pipeline.
    Uses registered filter methods to process connected subgraphs in the dataset.
    """
    
    def __init__(self):
        """Initialize the DataFilter."""
        pass
        
    def filter_data(self, dataset: QADataset, registered_methods: Dict[str, bool], config: Dict[str, Any]) -> QADataset:
        """
        Filter the dataset by applying registered filter methods to connected subgraphs.
        
        Args:
            dataset (QADataset): The dataset to filter
            registered_methods (Dict[str, bool]): Dictionary of method names and their registration status
            config (Dict[str, Any]): Configuration parameters for filtering
            
        Returns:
            QADataset: The filtered dataset
        """
        print("Starting the filtering process...")

        # Get all registered filter methods
        filter_methods = []
        for method_name, method_info in Method.get_methods().items():
            if 'data_filter' in method_info['applicable_stages'] and registered_methods.get(method_name, False):
                filter_methods.append(method_name)
                print(f"Using filter method: {method_name}")
        
        if not filter_methods:
            print("No filter methods registered. Skipping filtering stage.")
            return dataset
            
        # Find connected components (subgraphs) using BFS
        subgraphs = self._find_connected_components(dataset)
        print(f"Found {len(subgraphs)} connected subgraphs in the dataset")
        
        # IDs to delete from the dataset
        ids_to_delete = set()
        
        # Process each subgraph with the filter methods
        for i, subgraph_ids in enumerate(subgraphs):
            print(f"Processing subgraph {i+1}/{len(subgraphs)} with {len(subgraph_ids)} nodes")
            
            # Convert IDs to QAPairs
            subgraph_pairs = [dataset.get(id) for id in subgraph_ids]
            
            # Apply each filter method to the subgraph
            for method_name in filter_methods:
                method_func = Method.get_methods()[method_name]['func']
                print(f"  Applying {method_name} to subgraph {i+1}")
                
                # Apply the filter method
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