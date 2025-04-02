from typing import List, Dict, Callable, Optional, Any
from .base import QAPair, QADataset
from .methods import Method
from .data_expander import DataExpander
from .llms.oai_chat import OpenAIChat

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
            "combination_segmentation", 
            "filter_data",
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
        return dataset

    def combination_segmentation(self, dataset: QADataset, params: dict):
        print("Running combination & segmentation...")
        return dataset

    def filter_data(self, dataset: QADataset, params: dict):
        print("Filtering data...")
        return dataset

    def data_augmentation(self, dataset: QADataset, params: dict):
        print("Running data augmentation...")
        return dataset