from typing import List, Dict, Any
from .base import QAPair, QADataset
from .methods import Method

class DataAugmenter:
    """
    Handles data augmentation for QA pairs with a simple design.
    """

    def __init__(self):
        """
        Initialize the DataAugmenter class.
        """
        pass

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
                
                # Call the method function with the dataset QA pairs and the configuration
                augmented_pairs = method_info['func'](list(dataset.data.values()), config)
                
                # Add the augmented pairs to the dataset, linking back to original source
                for augmented_pair in augmented_pairs:
                    augmented_dataset.add(augmented_pair)
                    
                print(f"  Added {len(augmented_pairs)} augmented pairs using {method_name}")
        
        return augmented_dataset