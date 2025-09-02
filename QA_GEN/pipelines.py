# pipeline.py
from typing import List, Dict, Optional, Any
from .base import QADataset
from .method_register import Method, MethodRegistry, load_all_methods
from .data_expander import DataExpander
from .llms.oai_chat import OpenAIChat
from .llms.ollama_chat import OllamaChat
from .edge_builder import EdgeBuilder
from .data_fusioner import DataFusioner
from .data_filter import DataFilter
from .data_augmenter import DataAugmenter


class Pipeline:
    """
    Flexible data processing pipeline for QA datasets.
    Discovers and manages methods from methods/ directory dynamically.
    Supports disk-based storage for large-scale dataset processing.
    """
    def __init__(self, stages: Optional[List[str]] = None, methods_dir: str = "methods"):
        """
        Initialize Pipeline with customizable stages.
        
        Args:
            stages (Optional[List[str]]): List of stages to execute in the pipeline.
                                        Must be subset of available stages:
                                        ["data_expansion", "build_knowledge_graph", 
                                        "data_fusion", "data_filter", "data_augmentation"]
                                        If None, uses all default stages.
            methods_dir (str): Directory containing method files. Defaults to "methods".
        
        Raises:
            ValueError: If any stage in the provided list is not available.
        """
        # Define available stages
        self.available_stages = [
            "data_expansion", 
            "build_knowledge_graph", 
            "data_fusion", 
            "data_filter",
            "data_augmentation"
        ]
        
        # Set stages - use provided stages or default to all available stages
        if stages is None:
            self.stages = self.available_stages.copy()
        else:
            # Validate that all provided stages are available
            invalid_stages = [stage for stage in stages if stage not in self.available_stages]
            if invalid_stages:
                raise ValueError(f"Invalid stages provided: {invalid_stages}. "
                            f"Available stages are: {self.available_stages}")
            
            self.stages = stages.copy()
        
        # Initialize method registry and load methods
        self.method_registry = MethodRegistry(methods_dir)
        self.loaded_methods = {}
        self.methods_registered = {}
        
        # Load all methods from the methods directory
        self._load_methods()

    def _load_methods(self):
        """Load all methods from the methods directory."""
        try:
            print("Loading methods from methods directory...")
            self.loaded_methods = self.method_registry.discover_and_load_methods()
            
            # Initialize registration status
            self.methods_registered = {method: False for method in self.loaded_methods.keys()}
            
            # Validate loaded methods
            validation_results = self.method_registry.validate_methods()
            
            if validation_results["errors"]:
                print("Method validation errors:")
                for error in validation_results["errors"]:
                    print(f"  ERROR: {error}")
            
            if validation_results["warnings"]:
                print("Method validation warnings:")
                for warning in validation_results["warnings"]:
                    print(f"  WARNING: {warning}")
            
            print(f"Successfully validated {len(validation_results['valid'])} methods")
            
        except Exception as e:
            print(f"Error loading methods: {e}")
            self.loaded_methods = {}
            self.methods_registered = {}

    def reload_methods(self):
        """Reload all methods (useful for development)."""
        print("Reloading methods...")
        self._load_methods()

    def list_available_methods(self, stage: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        List available methods, optionally filtered by stage.
        
        Args:
            stage (Optional[str]): Filter methods by stage
        
        Returns:
            Dict[str, Dict[str, Any]]: Available methods
        """
        if stage:
            return Method.get_methods_for_stage(stage)
        return Method.get_methods()

    def get_method_info(self, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific method.
        
        Args:
            method_name (str): Name of the method
        
        Returns:
            Optional[Dict[str, Any]]: Method information or None if not found
        """
        return self.method_registry.get_method_info(method_name)

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
            for name, method in self.loaded_methods.items():
                if params.get('use_llm_for_method_selection', False):
                    # Use LLM to decide whether to use this method
                    llm = OllamaChat()
                    method_description = method['description']
                    prompt = f"Given the dataset description '{dataset_description}' and method description '{method_description}', should we use this method? Answer with 'yes' or 'no' only."
                    
                    try:
                        response_text, response_info = llm(prompt=prompt)
                        response_text = response_text.strip().lower()
                        response_text = response_text.strip('.,!?')
                        
                        if len(response_text) > 3:
                            response_text = response_text[:3]
                            
                        use_method = response_text in ['yes', 'y', 'true']
                    except Exception as e:
                        print(f"  Error in LLM method selection for '{name}': {e}")
                        use_method = True  # Default to using the method if LLM fails
                else:
                    # Default to using all methods if not using LLM selection
                    use_method = True
                
                self.methods_registered[name] = use_method
                status = "added" if use_method else "not applicable"
                print(f"  Method '{name}' {status}.")
        
        # Manual method configuration (overrides auto-config)
        methods_to_run = params.get('methods_to_run', [])
        if methods_to_run:
            # First, disable all methods
            for method_name in self.methods_registered:
                self.methods_registered[method_name] = False
            
            # Then enable only specified methods
            for method_name in methods_to_run:
                if method_name in self.methods_registered:
                    self.methods_registered[method_name] = True
                    print(f"  Method '{method_name}' manually enabled.")
                else:
                    print(f"  Method '{method_name}' not found in loaded methods.")
        
        # Print summary of enabled methods
        enabled_methods = [name for name, enabled in self.methods_registered.items() if enabled]
        print(f"Pipeline initialized with {len(enabled_methods)} enabled methods: {enabled_methods}")
        
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
        stage_methods = Method.get_methods_for_stage('data_expansion')
        
        for method_name in stage_methods:
            if self.methods_registered.get(method_name, False):
                expansion_methods.append(method_name)
        
        print(f"Using expansion methods: {expansion_methods}")
        
        # If we have a DataExpander class, use it
        if hasattr(self, '_use_data_expander') and self._use_data_expander:
            data_expander = DataExpander()
            expanded_pairs = data_expander.expand_data(dataset, expansion_methods)
        else:
            # Direct method execution
            expanded_pairs = []
            for method_name in expansion_methods:
                method_info = Method.get_methods()[method_name]
                method_func = method_info['func']
                
                try:
                    result = method_func(list(dataset.data.values()), config)
                    if isinstance(result, list):
                        expanded_pairs.extend(result)
                    print(f"  Applied method '{method_name}': +{len(result) if isinstance(result, list) else 0} pairs")
                except Exception as e:
                    print(f"  Error applying method '{method_name}': {e}")
            
            # Create new dataset with expanded data
            if expanded_pairs:
                # Add expanded pairs to original dataset
                all_pairs = list(dataset.data.values()) + expanded_pairs
                dataset._initialize_dataset(all_pairs)
        
        return dataset

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
        stage_methods = Method.get_methods_for_stage('data_fusion')
        
        for method_name in stage_methods:
            if self.methods_registered.get(method_name, False):
                fusion_methods.append(method_name)
        
        print(f"Using fusion methods: {fusion_methods}")
        
        # Initialize DataFusioner if available
        if hasattr(self, '_use_data_fusioner') and self._use_data_fusioner:
            fusioner = DataFusioner()
            fused_dataset = fusioner.fuse_data(dataset, fusion_methods, params)
        else:
            # Direct method execution for fusion methods
            fused_dataset = dataset
            for method_name in fusion_methods:
                method_info = Method.get_methods()[method_name]
                method_func = method_info['func']
                
                try:
                    result = method_func(fused_dataset, params)
                    if result:
                        fused_dataset = result
                    print(f"  Applied fusion method '{method_name}'")
                except Exception as e:
                    print(f"  Error applying fusion method '{method_name}': {e}")
        
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