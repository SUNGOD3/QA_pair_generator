#data_augmenter.py
from typing import List, Dict, Any, Optional
from .base import QAPair, QADataset
from .method_register import Method
from .docker_manager import DockerMethodManager
from .edge_builder import EdgeBuilder

class DataAugmenter:
    """
    Handles data augmentation for QA pairs with Docker environment support.
    Supports targeted augmentation for rare data.
    """

    def __init__(self):
        """
        Initialize the DataAugmenter class.
        """
        self.docker_manager = DockerMethodManager()

    def augment_data(self, dataset: QADataset, methods_registered: Dict[str, bool], config: Dict[str, Any],
                    augment_rare_only: bool = True, rare_threshold: int = 3, 
                    edge_method: Optional[str] = None) -> QADataset:
        """
        Augment dataset using registered augmentation methods.
        
        Args:
            dataset (QADataset): Original dataset
            methods_registered (Dict[str, bool]): Dictionary of registered methods
            config (Dict[str, Any]): Augmentation configuration
            augment_rare_only (bool): If True, only augment rare data points
            rare_threshold (int): Maximum component size to consider as rare
            edge_method (Optional[str]): Edge method to use for rare data detection
        
        Returns:
            QADataset: Augmented dataset
        """
        augmented_dataset = dataset.copy()
        
        # Determine which QA pairs to augment
        if augment_rare_only:
            edge_builder = EdgeBuilder(dataset)
            rare_qa_pairs = edge_builder.get_rare_data_qa_pairs(k=rare_threshold, method_name=edge_method)
            target_qa_pairs = rare_qa_pairs
        else:
            target_qa_pairs = list(dataset.data.values())
        
        if not target_qa_pairs:
            return augmented_dataset
        print("Size of target QA pairs for augmentation:", len(target_qa_pairs))
        augmentation_methods = Method.get_methods_for_stage("data_augmentation")
        
        for method_name, method_info in augmentation_methods.items():
            if methods_registered.get(method_name, False):
                try:
                    if method_info['use_docker'] == True:
                        augmented_pairs = self.docker_manager.execute_method_in_docker(
                            method_name, target_qa_pairs, config
                        )
                    else:
                        augmented_pairs = method_info['func'](target_qa_pairs, config)
                    
                    for augmented_pair in augmented_pairs:
                        augmented_dataset.add(augmented_pair)
                        
                except Exception as e:
                    if method_info['use_docker'] == False:
                        raise
                    else:
                        continue
        
        return augmented_dataset
    
    def __del__(self):
        """
        Cleanup Docker resources when the object is destroyed
        """
        if hasattr(self, 'docker_manager'):
            self.docker_manager.cleanup()