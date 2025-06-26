#pipeline.py
from typing import List, Dict, Callable, Optional, Any
from .base import QAPair, QADataset
from .methods import Method
from .docker_manager import DockerMethodManager
from .data_expander import DataExpander
from .llms.oai_chat import OpenAIChat
from .edge_builder import EdgeBuilder
from .data_fusioner import DataFusioner
from .data_filter import DataFilter
from .data_augmenter import DataAugmenter
import logging

class Pipeline:
    """
    Flexible data processing pipeline for QA datasets.
    Discovers and manages methods from methods.py dynamically.
    Supports disk-based storage for large-scale dataset processing.
    """
    def __init__(self):
        """
        Automatically discover and register methods from methods module
        """
        # Predefined stages
        self.stages = [
            "data_expansion", 
            "build_knowledge_graph", 
            "data_fusion", 
            "data_filter",
            "data_augmentation"
        ]
        self.methods_registered = {method: False for method in Method.get_methods().keys()}

    def run(self, dataset: QADataset, config: Dict[str, Any]) -> QADataset:
        """
        Execute the pipeline stages with discovered methods.
        Supports disk-based processing for large datasets.
        
        Args:
            dataset (QADataset): Initial dataset to process
            config (Dict[str, Any]): Configuration for processing
        
        Returns:
            QADataset: Processed dataset
        """
        # Initialize methods
        print(f"Running init_setup stage...")
        dataset = self.init_setup(dataset, config)
        
        # Process each stage with disk storage support
        for stage in self.stages:
            print(f"Running stage: {stage}")
            
            # Load previous stage results if using disk storage
            if dataset.use_disk:
                dataset.load_stage_input(stage)
                print(f"Loaded dataset from disk for stage {stage}, size: {len(dataset)}")
            
            # Execute the stage
            dataset = getattr(self, stage)(dataset, config)
            
            # Save stage results if using disk storage
            if dataset.use_disk:
                dataset.save_stage_result(stage)
                print(f"Saved dataset to disk after stage {stage}, size: {len(dataset)}")
            else:
                print(f"Size of dataset after {stage}: {len(dataset)}")
        
        print("Processing completed.")
        return dataset

    def init_setup(self, dataset: QADataset, params: dict):
        """
        Initialize the dataset and setup methods.
        Handles disk storage initialization if enabled.
        """
        print("Initializing dataset...")
        
        # Initialize dataset structure
        dataset._initialize_dataset(list(dataset.data.values()))
        
        # Save initial state if using disk storage
        if dataset.use_disk:
            dataset.save_stage_result("init_setup")
        
        # Auto-configure methods or use manual configuration
        if params.get('auto_config', True):
            dataset_description = dataset.description
            for name, method in Method.get_methods().items():
                # llm = OpenAIChat()
                # LLM: Decide whether to use this method
                method_description = method['description']
                prompt = f"Given the dataset description '{dataset_description}' and method description '{method_description}', should we use this method? (yes/no)"
                #response_text, response_info = llm(prompt=prompt)
                #response_text = response_text.strip()
                #response_text = response_text.strip('.,!?')
                response_text = 'yes'  # Simulate LLM response for testing
                if len(response_text) > 3:
                    response_text = response_text[:3]
                if response_text.lower() == 'yes' or response_text.lower() == 'y' or response_text.lower() == 'true':
                    self.methods_registered[name] = True
                    print(f"  Method '{name}' added to dataset.")
                else:
                    self.methods_registered[name] = False
                    print(f"  Method '{name}' not applicable.")
        
        # Manual method configuration
        method = params.get('methods_to_run', [])
        if method:
            for method_name in method:
                if method_name in self.methods_registered:
                    self.methods_registered[method_name] = True
                    print(f"  Method '{method_name}' added to dataset.")
                else:
                    print(f"  Method '{method_name}' not found or not applicable.")
        
        return dataset

    def data_expansion(self, dataset: QADataset, config: Dict[str, Any]) -> QADataset:
        """
        Data expansion stage in the pipeline.
        
        Args:
            dataset (QADataset): Input dataset
            config (Dict[str, Any]): Expansion configuration
        
        Returns:
            QADataset: Expanded dataset
        """
        print("Expanding data...")
        
        # Get expansion methods from registered Methods
        expansion_methods = []
        for method_name, method_info in Method.get_methods().items():
            if 'data_expansion' in method_info['applicable_stages'] and self.methods_registered.get(method_name, True):
                expansion_methods.append(method_name)
        
        # Expand data using DataExpander
        data_expander = DataExpander()
        expanded_pairs = data_expander.expand_data(dataset, expansion_methods)
        
        return expanded_pairs

    def build_knowledge_graph(self, dataset: QADataset, params: dict):
        """
        Build knowledge graph stage with disk storage support.
        """
        print("Building knowledge graph...")
        
        edge_builder = EdgeBuilder(dataset)

        # Build edges automatically
        edge_builder.build_cosine_similarity_edges()
        edge_builder.build_keyword_overlap_edges()

        # Visualize graphs (only if dataset is not too large or if explicitly requested)
        if len(dataset) < params.get('max_visualization_size', 1000):
            edge_builder.visualize_all_graphs()
            edge_builder.visualize_combined_graph()
        else:
            print("Dataset too large for visualization, skipping graph visualization")

        stats = edge_builder.get_graph_statistics()
        print("Graph Statistics:", stats)
        
        return dataset

    def data_fusion(self, dataset: QADataset, params: dict):
        """
        Data fusion stage with disk storage support.
        """
        print("Running data_fusion...")
        
        # Get fusion methods from registered Methods
        fusion_methods = []
        for method_name, method_info in Method.get_methods().items():
            if 'data_fusion' in method_info['applicable_stages'] and self.methods_registered.get(method_name, True):
                fusion_methods.append(method_name)
        
        # Initialize DataFusioner
        fusioner = DataFusioner()
        
        # Fuse data using the registered methods
        fused_dataset = fusioner.fuse_data(dataset, fusion_methods, params)
        
        return fused_dataset

    def data_filter(self, dataset: QADataset, params: dict):
        """
        Filter data stage in the pipeline with disk storage support.
        
        Args:
            dataset (QADataset): Input dataset
            params (dict): Configuration parameters
        
        Returns:
            QADataset: Filtered dataset
        """
        print("Filtering data...")
        
        # For debugging purposes (only print first few items for large datasets)
        print("Original dataset size:", len(dataset))
        if len(dataset) <= 10:
            for qa in dataset:
                print(qa)
        else:
            print("Dataset too large to print all items, showing first 3:")
            for i, qa in enumerate(dataset):
                if i >= 3:
                    break
                print(qa)
        
        # Initialize and run the DataFilter
        data_filter = DataFilter()
        filtered_dataset = data_filter.filter_data(dataset, self.methods_registered, params)
        
        return filtered_dataset

    def data_augmentation(self, dataset: QADataset, params: dict):
        """
        Data augmentation stage in the pipeline with disk storage support.
        
        Args:
            dataset (QADataset): Input dataset
            params (dict): Configuration parameters
            
        Returns:
            QADataset: Augmented dataset
        """
        print("Running data augmentation...")
        
        # Initialize DataAugmenter
        data_augmenter = DataAugmenter()
        
        # Apply augmentation methods to the dataset
        augmented_dataset = data_augmenter.augment_data(dataset, self.methods_registered, params)
        
        print(f"Dataset size after augmentation: {len(augmented_dataset)}")
        
        return augmented_dataset

    def cleanup_disk_storage(self, dataset: QADataset):
        """
        Clean up disk storage files for the dataset.
        
        Args:
            dataset (QADataset): Dataset whose disk storage should be cleaned up
        """
        if dataset.use_disk and dataset.disk_path:
            import shutil
            try:
                shutil.rmtree(dataset.disk_path)
                print(f"Cleaned up disk storage at: {dataset.disk_path}")
            except Exception as e:
                print(f"Warning: Could not clean up disk storage: {e}")

    def get_disk_usage(self, dataset: QADataset) -> dict:
        """
        Get disk usage information for the dataset.
        
        Args:
            dataset (QADataset): Dataset to check disk usage for
            
        Returns:
            dict: Dictionary containing disk usage information
        """
        if not dataset.use_disk or not dataset.disk_path:
            return {"disk_enabled": False}
        
        import os
        from pathlib import Path
        
        disk_path = Path(dataset.disk_path)
        if not disk_path.exists():
            return {"disk_enabled": True, "path": str(disk_path), "exists": False}
        
        total_size = 0
        file_info = {}
        
        for file_path in disk_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                file_info[file_path.name] = {
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2)
                }
        
        return {
            "disk_enabled": True,
            "path": str(disk_path),
            "exists": True,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": file_info
        }