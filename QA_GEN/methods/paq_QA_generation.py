from typing import List
from QA_GEN.base import QAPair
from QA_GEN.method_register import Method
from QA_GEN.methods.paq_answer_extraction import paq_answer_extraction
from QA_GEN.methods.paq_question_generation import paq_question_generation

@Method(
    name="paq_QA_generation", 
    description="paq_answer_extraction + paq_question_generation",
    applicable_stages=["data_expansion"],
    use_LLM=True,
)
def paq_QA_generation(qa_pairs: List[QAPair], config, k: int = 8, llm = None) -> List[QAPair]:
    """
    Combine PAQ-style answer extraction and question generation into a single method.
    
    Args:
        qa_pairs: List of QAPair objects (should contain context-only pairs)
        config: Configuration object
        k: Number of candidate answers to extract per context (default: 3)
    Returns:
        List of complete QAPair objects with generated questions and extracted answers
    """
    extracted_pairs = paq_answer_extraction(qa_pairs, config, k=k, llm=llm)
    complete_pairs = paq_question_generation(extracted_pairs, config, llm=llm)
    return complete_pairs