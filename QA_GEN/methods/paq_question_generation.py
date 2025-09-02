from typing import List
import re
from QA_GEN.base import QAPair
from QA_GEN.llms.ollama_chat import OllamaChat
from QA_GEN.method_register import Method

@Method(
    name="paq_question_generation", 
    description="Generate questions for QAPairs that have context and answer but missing questions (PAQ-style question generation). Converts Context+Answer pairs into complete QA pairs.",
    applicable_stages=["data_expansion"],
    use_LLM=True,
    complexity="O(n) where n is the number of context+answer pairs; LLM calls may impact performance"
)
def paq_question_generation(qa_pairs: List[QAPair], config, llm = None) -> List[QAPair]:
    """
    Generate questions for QAPairs that have context and answer but no question.
    
    Args:
        qa_pairs: List of QAPair objects (should contain Context+Answer pairs)
        config: Configuration object
    
    Returns:
        List of complete QAPair objects with generated questions
    """
    generated_pairs = []
    
    for qa_pair in qa_pairs:
        if qa_pair.classify_id() == 5:  # Context + Ground Truth type
            if llm is None:
                llm = OllamaChat()
            
            # Prompt for generating a question given context and answer
            prompt = f"""
You are an AI specialized in generating natural questions given a context and a target answer.
Your task is to create a clear, natural question that would logically lead to the given answer based on the provided context.

Instructions:
1. Generate exactly ONE question that would have the given answer
2. The question should be natural and grammatically correct
3. The question should be answerable from the context provided
4. The question should specifically target the given answer
5. Avoid overly complex or ambiguous questions
6. The question should be complete and standalone

Output format:
Q: <your generated question>

Context: {qa_pair.context[:2000]}

Target Answer: {qa_pair.answer}
"""
            
            try:
                response_text, response_info = llm(prompt=prompt)
                
                # Extract the question from the response
                match = re.search(r"Q:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
                if match:
                    generated_question = match.group(1).strip()
                    
                    # Create a new complete QAPair
                    new_pair = QAPair(
                        context=qa_pair.context,
                        question=generated_question,
                        answer=qa_pair.answer
                    )
                    
                    # Copy original metadata and add generation info
                    new_pair.metadata = qa_pair.metadata.copy()
                    generated_pairs.append(new_pair)
                    
            except Exception as e:
                continue
    
    return generated_pairs