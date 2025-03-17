from typing import Dict
from .data_expansion import expand_data


def init_setup(dataset, config):
    print("Init setup: Applying initial configurations.")
    # TODO:
    return dataset

def data_expansion(dataset, config: Dict):
    """Expands dataset based on the adjacency matrix and selected methods in config."""
    new_data = expand_data(dataset, config)
    return dataset


def build_knowledge_graph(dataset, config):
    print("Building Knowledge Graph: Creating relationships between QA pairs.")

    for source_id, qa1 in dataset.items():  
        for target_id, qa2 in dataset.items():
            if source_id != target_id and qa1.answer and qa2.question:
                if qa1.answer in qa2.question:
                    qa1.add_link(target_id)  

    return dataset 

def combination_segmentation(dataset, config):
    """組合與切割"""
    print("Combination & Segmentation: Merging or splitting QA pairs.")
    # TODO: 根據需求進行長文本拆分或合併相似的 QA pair
    return dataset

def filter_data(dataset, config):
    """數據過濾"""
    print("Filtering: Removing unwanted QA pairs.")
    # TODO: 根據 config 過濾掉不需要的數據，例如刪除重複 QA 或過短的答案
    return dataset

def data_augmentation(dataset, config):
    """數據增強"""
    print("Data Augmentation: Generating new variations of QA pairs.")
    # TODO: 例如透過 NLP 技術生成語法變體，擴展數據量
    return dataset
