#methods.py
from typing import Any, Callable, Dict, List
from .base import QAPair
from .llms.oai_chat import OpenAIChat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random

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
        applicable_stages: List[str], # List[str] = ["init_setup", "data_expansion", "build_knowledge_graph", "data_fusion", "filter", "data_augmentation"]
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


@Method(
    name="key_sentences_question",
    description="Generate only questions based on key sentences",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def key_sentences_question(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        sentences =  qa_pair.context_keywords
        if not sentences:
            return []
        key_sentence = random.choice(sentences)
        question = f"Based on the content, what does '{key_sentence}' mean?"
        qa_pair.set_question(question)
        rt_pairs.append(qa_pair)
    return rt_pairs

@Method(
    name="generate_summary_question",
    description="Generate summary questions, where the question is about the summary of the content",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def generate_summary_question(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        question = f"Based on the content, what is the key information summarized?"
        qa_pair.set_question(question)
        rt_pairs.append(qa_pair)
    return rt_pairs

@Method(
    name="generate_summary_qa",
    description="Generate summary-type QA pairs, where the question is about the summary of the content",
    applicable_stages=["data_expansion"],
    use_LLM=True
)
def generate_summary_qa(qa_pairs, config):
    rt_pairs = []
    llm = OpenAIChat()
    question = f"Based on the content, what is the key information summarized?"
    for qa_pair in qa_pairs:
        qa_pair.set_question(question)
        response_text, response_info = llm(prompt=qa_pair.context + '\n' + question)
        summarized_text = response_text.strip()
        qa_pair.set_answer(summarized_text)
        rt_pairs.append(qa_pair)
    return rt_pairs

@Method(
    name="generate_fill_in_blank",
    description="Generate fill-in-the-blank type QA pairs",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def generate_fill_in_blank(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        context = qa_pair.context
        words = word_tokenize(context)
        words = [w for w in words if w.isalnum() and w.lower() not in stopwords.words("english")]
        if not words:
            return []
        blank_word = random.choice(words)
        question = context.replace(blank_word, "____")
        qa_pair.set_question(question)
        qa_pair.set_answer(blank_word)
        rt_pairs.append(qa_pair)
    return rt_pairs

print("Methods registered:")
for name, method in Method.get_methods().items():
    print(f" - {name}: {method['description']}")

@Method(
    name="few_shot_fusion",
    description="Generate new QA pairs by providing a few similar examples to the LLM",
    applicable_stages=["data_fusion"],
    use_LLM=True
)
def few_shot_fusion(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Create new QA pairs by showing a few examples to the LLM and asking it to generate
    new content following the same pattern or structure.
    
    Args:
        qa_pairs (List[QAPair]): List of QA pairs to use as examples
        config: Configuration parameters
        
    Returns:
        List[QAPair]: Newly generated QA pairs
    """
    if len(qa_pairs) < 2:
        print("Need at least 2 QA pairs for few-shot fusion")
        return []
    
    # Get the source pair and example pairs
    source_pair = qa_pairs[0]
    example_pairs = qa_pairs[1:]
    
    # Prepare examples for few-shot learning
    examples_text = ""
    for i, example in enumerate(example_pairs, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Context: {example.context}\n"
        examples_text += f"Question: {example.question}\n"
        examples_text += f"Answer: {example.answer}\n\n"
    
    # Create prompt for LLM
    llm = OpenAIChat()
    prompt = f"""
You are an expert at creating educational question-answer pairs.
I'll show you some example QA pairs, and I want you to create a new, original QA pair that follows the same pattern, style, and difficulty level.

Here are some examples:

{examples_text}

Now, based on these examples and using the following context as inspiration, create a new, original QA pair:

Context: {source_pair.context}

Please provide your response in this format:
Context: (a variation of the provided context, modified to be original)
Question: (your original question)
Answer: (your original answer)
"""

    # Generate new QA pair
    response_text, response_info = llm(prompt=prompt)
    
    # Parse the response
    context_match = re.search(r"Context:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    question_match = re.search(r"Question:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    answer_match = re.search(r"Answer:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    
    if context_match and question_match and answer_match:
        new_context = context_match.group(1).strip()
        new_question = question_match.group(1).strip()
        new_answer = answer_match.group(1).strip()
        
        # Create a new QA pair
        new_pair = QAPair(
            question=new_question,
            answer=new_answer,
            context=new_context
        )
        
        return [new_pair]
    else:
        print("Failed to parse the LLM response for few-shot fusion")
        return []
    
@Method(
    name="cross_domain_fusion",
    description="Fuse two moderately related QA pairs to create a new hybrid QA pair that bridges both domains",
    applicable_stages=["data_fusion"],
    use_LLM=True
)
def cross_domain_fusion(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Create a new QA pair by fusing knowledge from two moderately related QA pairs.
    This method works best when the input pairs have some conceptual overlap but
    are from different domains or contexts.
    
    Args:
        qa_pairs (List[QAPair]): List of QA pairs to fuse (ideally 2 pairs)
        config: Configuration parameters
        
    Returns:
        List[QAPair]: Newly generated hybrid QA pair
    """
    # Need exactly 2 QA pairs for this fusion method
    if len(qa_pairs) != 2:
        print("Cross-domain fusion requires exactly 2 QA pairs")
        return []
    
    pair1, pair2 = qa_pairs
    
    # Create prompt for LLM
    llm = OpenAIChat()
    prompt = f"""
You are an expert at creating content that bridges multiple domains or topics.
I'll show you two different QA pairs that have some moderate relationship, and I want you to create a new, 
original QA pair that creatively combines elements from both sources.

First QA Pair:
Context: {pair1.context}
Question: {pair1.question}
Answer: {pair1.answer}

Second QA Pair:
Context: {pair2.context}
Question: {pair2.question}
Answer: {pair2.answer}

Create a new QA pair that bridges these two topics or domains. The new QA pair should:
1. Identify a conceptual link or relationship between these two areas
2. Create something novel that couldn't be derived from either source alone
3. Be factually accurate

Please provide your response in this format:
Context: (a bridge context that connects elements from both sources)
Question: (your original question that spans both domains)
Answer: (your original answer that incorporates knowledge from both sources)
"""

    # Generate new QA pair
    response_text, response_info = llm(prompt=prompt)
    
    # Parse the response
    context_match = re.search(r"Context:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    question_match = re.search(r"Question:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    answer_match = re.search(r"Answer:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    
    if context_match and question_match and answer_match:
        new_context = context_match.group(1).strip()
        new_question = question_match.group(1).strip()
        new_answer = answer_match.group(1).strip()
        
        # Create a new QA pair
        new_pair = QAPair(
            question=new_question,
            answer=new_answer,
            context=new_context
        )
        
        return [new_pair]
    else:
        print("Failed to parse the LLM response for cross-domain fusion")
        return []