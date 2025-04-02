from typing import Any, Callable, Dict, List
from .base import QAPair
from .llms.oai_chat import OpenAIChat
import re

class Method:
    """
    Decorator class to automatically register methods with metadata.
    Stores all discovered methods with their associated information.
    """
    # Class-level dictionary to store all discovered methods
    _methods: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self, 
        name: str, 
        description: str, 
        applicable_stages: List[str], # List[str] = ["init_setup", "data_expansion", "build_knowledge_graph", "combination_segmentation", "filter", "data_augmentation"]
        use_LLM: bool = False
    ):
        """
        Initialize a method with its metadata.
        
        Args:
            name (str): Unique name of the method
            description (str): Detailed description of the method
            applicable_stages (List[str]): Stages where this method can be applied
            use_LLM (bool, optional): Whether the method uses LLM. Defaults to False.
        """
        self.name = name
        self.description = description
        self.applicable_stages = applicable_stages
        self.use_LLM = use_LLM

    def __call__(self, func: Callable):
        """
        Register the method when the decorator is applied.
        
        Args:
            func (Callable): The method function to be registered
        
        Returns:
            Callable: The original function
        """
        # Store method information
        Method._methods[self.name] = {
            'func': func,
            'name': self.name,
            'description': self.description,
            'applicable_stages': self.applicable_stages,
            'use_LLM': self.use_LLM
        }
        return func

    @classmethod
    def get_methods(cls) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all registered methods.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of registered methods
        """
        return cls._methods

    @classmethod
    def get_methods_for_stage(cls, stage: str) -> Dict[str, Dict[str, Any]]:
        """
        Get methods applicable to a specific stage.
        
        Args:
            stage (str): Pipeline stage to filter methods for
        
        Returns:
            Dict[str, Dict[str, Any]]: Methods applicable to the stage
        """
        return {
            name: method for name, method in cls._methods.items()
            if stage in method['applicable_stages']
        }

@Method(
    name="context_to_qa", 
    description="Expand context-only QA pairs to full QA pairs",
    applicable_stages=["data_expansion"],
    use_LLM=True
)
def context_to_qa(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Generate full QA pairs from context-only entries.
    
    Args:
        qa_pairs (List[QAPair]): Input QA pairs
        config : Configuration for this method
    
    Returns:
        List[QAPair]: Expanded QA pairs
    """
    expanded_pairs = []

    for context_pair in qa_pairs:
        if context_pair.classify_id() == 1:  # Context-only type
            llm = OpenAIChat()
            # Generate full QA pairs using LLM
            prompt = f"""
You are an AI specialized in generating QA pairs from a given context.  
Generate a well-structured question and answer from the provided context.  
Output **only** in the following format:  

Q: <your question>  
A: <your answer>  

Context: {context_pair.context}  
"""
            response_text, response_info = llm(prompt=prompt)
            match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", response_text, re.DOTALL)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                new_pair = QAPair(
                    question=question,
                    answer=answer,
                    context=context_pair.context
                )
                expanded_pairs.append(new_pair)
            else:
                print("Parsing failed:", response_text)
    return expanded_pairs


print("Methods registered:")
for name, method in Method.get_methods().items():
    print(f" - {name}: {method['description']}")