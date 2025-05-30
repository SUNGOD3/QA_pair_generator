#data_augmenter.py
from typing import List, Dict, Any
from .base import QAPair, QADataset
from .methods import Method
from .docker_manager import DockerMethodManager

class DataAugmenter:
    """
    Handles data augmentation for QA pairs with Docker environment support.
    """

    def __init__(self):
        """
        Initialize the DataAugmenter class.
        
        Args:
            use_docker: Whether to use Docker for method execution
        """
        self.docker_manager = DockerMethodManager()

    def augment_data(self, dataset: QADataset, methods_registered: Dict[str, bool], config: Dict[str, Any]) -> QADataset:
        """
        Augment dataset using registered augmentation methods.
        
        Args:
            dataset (QADataset): Original dataset
            methods_registered (Dict[str, bool]): Dictionary of registered methods
            config (Dict[str, Any]): Augmentation configuration
        
        Returns:
            QADataset: Augmented dataset
        """
        # Create a copy of the original dataset to add augmented data to
        augmented_dataset = dataset.copy()
        
        # Get all methods that apply to the data_augmentation stage
        augmentation_methods = Method.get_methods_for_stage("data_augmentation")
        
        # For each applicable method that is registered
        for method_name, method_info in augmentation_methods.items():
            if methods_registered.get(method_name, False):
                print(f"Applying augmentation method: {method_name}")
                
                try:
                    if method_info['use_docker'] == True:
                        # Execute method in Docker environment
                        augmented_pairs = self.docker_manager.execute_method_in_docker(
                            method_name, 
                            list(dataset.data.values()), 
                            config
                        )
                    else:
                        # Execute method directly (fallback)
                        augmented_pairs = method_info['func'](list(dataset.data.values()), config)
                    
                    # Add the augmented pairs to the dataset
                    for augmented_pair in augmented_pairs:
                        augmented_dataset.add(augmented_pair)
                        
                    print(f"  Added {len(augmented_pairs)} augmented pairs using {method_name}")
                    
                except Exception as e:
                    print(f"  Error executing method {method_name}: {str(e)}")
                    if method_info['use_docker'] == False:
                        # If not using Docker, re-raise the exception
                        raise
                    else:
                        # If using Docker, continue with other methods
                        print(f"  Skipping method {method_name} due to error")
                        continue
        
        return augmented_dataset
    
    def __del__(self):
        """
        Cleanup Docker resources when the object is destroyed
        """
        if hasattr(self, 'docker_manager'):
            self.docker_manager.cleanup()