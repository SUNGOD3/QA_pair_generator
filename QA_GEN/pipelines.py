#pipeline.py
from typing import List, Dict, Callable, Optional, Any
from .base import QAPair, QADataset
from .methods import Method
from .data_expander import DataExpander
from .llms.oai_chat import OpenAIChat
from .edge_builder import EdgeBuilder
from .data_fusioner import DataFusioner
from .data_filter import DataFilter

class Pipeline:
    """
    Flexible data processing pipeline for QA datasets.
    Discovers and manages methods from methods.py dynamically.
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
        
        Args:
            dataset (QADataset): Initial dataset to process
            config (Dict[str, Any]): Configuration for processing
        
        Returns:
            QADataset: Processed dataset
        """
        # Initialize methods
        print(f"Running init_setup stage...")
        self.init_setup(dataset, config)

        for stage in self.stages:
            print(f"Running stage: {stage}")
            dataset = getattr(self, stage)(dataset, config)
            print(f"Size of dataset after {stage}: {len(dataset)}")
        
        print("Processing completed.")
        return dataset


    def init_setup(self, dataset: QADataset, params: dict):
        print("Initializing dataset...")
        dataset._initialize_dataset(list(dataset.data.values()))
        if params.get('auto_config', True):
            dataset_description = dataset.description
            for name, method in Method.get_methods().items():
                llm = OpenAIChat()
                # LLM: Decide whether to use this method
                method_description = method['description']
                prompt = f"Given the dataset description '{dataset_description}' and method description '{method_description}', should we use this method? (yes/no)"
                response_text, response_info = llm(prompt=prompt)
                response_text = response_text.strip()
                response_text = response_text.strip('.,!?')
                if len(response_text) > 3:
                    response_text = response_text[:3]
                if response_text.lower() == 'yes' or response_text.lower() == 'y' or response_text.lower() == 'true':
                    self.methods_registered[name] = True
                    print(f"  Method '{name}' added to dataset.")
                else:
                    self.methods_registered[name] = False
                    print(f"  Method '{name}' not applicable.")
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
        # Add methods to the configuration
        expansin_methods = []
        for method_name, method_info in Method.get_methods().items():
            if 'data_expansion' in method_info['applicable_stages'] and self.methods_registered.get(method_name, True):
                expansin_methods.append(method_name)
        
        # Expand data using DataExpander
        data_expander = DataExpander()
        expanded_pairs = data_expander.expand_data(dataset, expansin_methods)
        
        return expanded_pairs

    def build_knowledge_graph(self, dataset: QADataset, params: dict):
        print("Building knowledge graph...")
        edge_builder = EdgeBuilder(dataset)

        # Build edges automatically
        edge_builder.build_cosine_similarity_edges()
        edge_builder.build_keyword_overlap_edges()

        # Visualize all graphs separately
        edge_builder.visualize_all_graphs()

        # Visualize a combined graph
        edge_builder.visualize_combined_graph()
        stats = edge_builder.get_graph_statistics()
        print("Graph Statistics:", stats)
        return dataset

    def data_fusion(self, dataset: QADataset, params: dict):
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
        Filter data stage in the pipeline. Applies registered filter methods to the dataset.
        
        Args:
            dataset (QADataset): Input dataset
            params (dict): Configuration parameters
        
        Returns:
            QADataset: Filtered dataset
        """
        print("Filtering data...")
        
        # For debugging purposes, print the original dataset
        print("Original dataset size:", len(dataset))
        for qa in dataset:
            print(qa)
        
        # Initialize and run the DataFilter
        
        data_filter = DataFilter()
        filtered_dataset = data_filter.filter_data(dataset, self.methods_registered, params)
        
        return filtered_dataset

    def data_augmentation(self, dataset: QADataset, params: dict):
        print("Running data augmentation...")
        return dataset